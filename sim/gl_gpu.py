"""Production GL relaxation for the DMI bilayer (same functional/conventions as params.py).

Backend: CuPy if importable and a device exists (env GL_BACKEND=numpy forces CPU), else NumPy.
Scheme (identical to gl_bilayer_reference.Bilayer): dm/dt = -dF/dm, semi-implicit,
    (1 + 2 dt L_q) m^{n+1} = m^n - dt N(m^n)
linear part L_q (6x6 per q) implicit, nonlinear part (4u m^2 m - H e_z) explicit.
Implementation differences w.r.t. the reference: vectorised kernel, rfft2 (real fields), batched matmul.
"""
import os
import numpy as np
from params import landau_to_model

_want = os.environ.get('GL_BACKEND', 'auto')
xp = np
ON_GPU = False
if _want in ('auto', 'cupy'):
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        xp, ON_GPU = cp, True
    except Exception:
        pass


def to_np(a):
    return xp.asnumpy(a) if ON_GPU else np.asarray(a)


def kernel_field(p, qx, qy, nyq_fix=True):
    """Vectorised version of params.kernel: array (...,6,6) for arrays qx,qy (numpy).
    nyq_fix: odd (DMI) derivative terms are set to zero on the Nyquist lines |q|=pi, where a real field's
    mode is real and i*D*q is not Hermitian-consistent (this is the standard spectral treatment of odd derivatives)."""
    sh = qx.shape
    if nyq_fix:
        dqx = np.where(np.isclose(np.abs(qx), np.pi), 0.0, qx)
        dqy = np.where(np.isclose(np.abs(qy), np.pi), 0.0, qy)
    else:
        dqx, dqy = qx, qy
    L = np.zeros(sh + (6, 6), complex)
    q2 = qx * qx + qy * qy
    for i, (r, D) in enumerate([(p['r1'], p['D1']), (p['r2'], p['D2'])]):
        o = 3 * i
        for a in range(3):
            L[..., o + a, o + a] += r + p['A'] * q2 + p['c']
        L[..., o + 2, o + 0] += 1j * D * dqx
        L[..., o + 0, o + 2] += -1j * D * dqx
        L[..., o + 2, o + 1] += 1j * D * dqy
        L[..., o + 1, o + 2] += -1j * D * dqy
    for a in range(3):
        L[..., a, 3 + a] += -p['c']
        L[..., 3 + a, a] += -p['c']
    return L


class GLBilayer:
    def __init__(s, p, L=128, dt=0.5, seed=0, init='noise', stab=0.0):
        """stab: S in the stabilised splitting (1+2dt(L+S)) m' = m - dt (N(m) - 2 S m); fixed points are unchanged
        (S=0 is the reference scheme). Use S ~ kappa*|r| to allow dt >> 1/(12|r|)."""
        s.p, s.L, s.dt, s.stab = dict(p), L, dt, float(stab)
        k = 2 * np.pi * np.fft.fftfreq(L)
        kr = 2 * np.pi * np.fft.rfftfreq(L)
        s.qx, s.qy = np.meshgrid(k, kr, indexing='ij')       # (L, L//2+1)
        s.build()
        rng = np.random.default_rng(seed)
        m = 1e-3 * rng.normal(size=(6, L, L))
        if init == 'noise':
            m[2] += p['M_target']; m[5] += p['M_target']
        s.m = xp.asarray(m)

    def build(s):
        p = s.p
        Lq = kernel_field(p, s.qx, s.qy)
        Ai = np.linalg.inv(np.eye(6)[None, None] * (1 + 2 * s.dt * s.stab) + 2 * s.dt * Lq)
        s.Lq = xp.asarray(Lq)
        s.Ainv = xp.asarray(Ai)
        # weights for rfft energy sum (columns 0 and L/2 counted once, others twice)
        w = np.full(s.qx.shape[1], 2.0); w[0] = 1.0
        if s.L % 2 == 0: w[-1] = 1.0
        s.w = xp.asarray(w)[None, :]

    def set_fields(s, H1, H2):
        s.p['H1'], s.p['H2'] = float(H1), float(H2)

    def nonlin(s, m):
        g = xp.empty_like(m)
        for o, H in ((0, s.p['H1']), (3, s.p['H2'])):
            m2 = m[o] ** 2 + m[o + 1] ** 2 + m[o + 2] ** 2
            g[o:o + 3] = 4 * s.p['u'] * m2 * m[o:o + 3]
            g[o + 2] -= H
        return g

    def step(s, n=1):
        L = s.L
        for _ in range(n):
            mh = xp.fft.rfft2(s.m, axes=(1, 2))
            nh = xp.fft.rfft2(s.nonlin(s.m), axes=(1, 2))
            rhs = xp.moveaxis(mh * (1 + 2 * s.dt * s.stab) - s.dt * nh, 0, -1)
            out = xp.matmul(s.Ainv, rhs[..., None])[..., 0]
            s.m = xp.fft.irfft2(xp.moveaxis(out, -1, 0), s=(L, L), axes=(1, 2))

    def energy(s):
        N = s.L * s.L
        mh = xp.moveaxis(xp.fft.rfft2(s.m, axes=(1, 2)), 0, -1)
        Lm = xp.matmul(s.Lq, mh[..., None])[..., 0]
        e2 = float(xp.real((xp.conj(mh) * Lm).sum(-1) * s.w).sum()) / N
        e4 = 0.0
        for o, H in ((0, s.p['H1']), (3, s.p['H2'])):
            m2 = s.m[o] ** 2 + s.m[o + 1] ** 2 + s.m[o + 2] ** 2
            e4 += s.p['u'] * float((m2 ** 2).sum()) - H * float(s.m[o + 2].sum())
        return (e2 + e4) / N

    def uniform_energy(s, iters=60):
        """energy per site of the uniform (field-polarised, unmodulated) state at the current (H1,H2): Newton on the
        two coupled cubic equations 2 r_i m_i + 4u m_i^3 + 2c(m_i-m_j) = H_i. The modulated-state gain is E - E_uniform
        (this is the quantity that corresponds to the Landau F at fixed M)."""
        p = s.p
        m = np.array([p['M_target'], p['M_target']], float)
        r = np.array([p['r1'], p['r2']]); H = np.array([p['H1'], p['H2']]); c, u = p['c'], p['u']
        for _ in range(iters):
            f = 2 * r * m + 4 * u * m ** 3 + 2 * c * (m - m[::-1]) - H
            J = np.array([[2 * r[0] + 12 * u * m[0] ** 2 + 2 * c, -2 * c], [-2 * c, 2 * r[1] + 12 * u * m[1] ** 2 + 2 * c]])
            m = m - np.linalg.solve(J, f)
        E = float((r * m ** 2 + u * m ** 4 - H * m).sum() + c * (m[0] - m[1]) ** 2)
        return E, m

    def mz_mean(s):
        return float(s.m[2].mean()), float(s.m[5].mean())

    def structure(s, layer):
        """full (L,L) S_i(q) = sum_alpha |m^_ia(q)|^2 on the fftfreq grid (numpy, k=0 removed)."""
        o = 3 * layer
        S = 0
        for a in range(3):
            f = s.m[o + a]
            S = S + xp.abs(xp.fft.fft2(f - f.mean())) ** 2
        return to_np(S)

    def mhat(s, layer, comp=2):
        return to_np(xp.fft.fft2(s.m[3 * layer + comp]))

    def state(s):
        return to_np(s.m)


def relax(sim, nmax, chunk=200, tol=1e-9, verbose=True, min_chunks=3):
    """Relax until relative energy change per chunk < tol. Returns energy history (checks monotonic decrease)."""
    E = [sim.energy()]
    for c in range(nmax // chunk):
        sim.step(chunk)
        E.append(sim.energy())
        rel = abs(E[-1] - E[-2]) / max(abs(E[-1]), 1e-300)
        if verbose and (c % 10 == 0):
            print(f'  step {(c+1)*chunk:7d}  E/r^2 = {E[-1]/sim.p["r_abs"]**2: .6f}  rel dE = {rel:.2e}', flush=True)
        if c + 1 >= min_chunks and rel < tol:
            break
    return np.array(E)


def calibrate_fields(sim, M_target, iters=6, chunk=400, nrelax=4000, tol_m=1e-3, verbose=True, gain=1.0):
    """Step 1: iterate H_i so that layer-resolved <m_z> -> M_target (relax, measure, update H)."""
    p = sim.p
    hist = []
    for it in range(iters):
        relax(sim, nrelax, chunk=chunk, tol=1e-8, verbose=False)
        mz = sim.mz_mean()
        hist.append((p['H1'], p['H2'], mz[0], mz[1]))
        err = max(abs(mz[0] - M_target), abs(mz[1] - M_target)) / M_target
        if verbose:
            print(f' calib {it}: H=({p["H1"]:.4e},{p["H2"]:.4e}) <mz>=({mz[0]:.5f},{mz[1]:.5f}) M*={M_target:.5f} err={err:.2e}', flush=True)
        if err < tol_m:
            break
        # secant-like update: dH/dM ~ 2 r + 12 u M^2 (uniform-state susceptibility) -- fall back to proportional
        p['H1'] += gain * (2 * p['r1'] + 12 * p['u'] * M_target ** 2 + 2 * p['c']) * (M_target - mz[0]) * 0.5
        p['H2'] += gain * (2 * p['r2'] + 12 * p['u'] * M_target ** 2 + 2 * p['c']) * (M_target - mz[1]) * 0.5
    return hist


def g_eff(p, mz):
    """g_eff = 4 u <m_z> / sqrt(u |r|), |r| = r_abs (band depth)."""
    return 4 * p['u'] * mz / np.sqrt(p['u'] * p['r_abs'])
