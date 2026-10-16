"""Fixed-length (|m_i| = 1) two-layer DMI model, relaxed by preconditioned projected gradient descent
(overdamped LLG / steepest descent on the sphere product).

E = sum_i  int [ A |grad m_i|^2 + 2 D_i m_iz div(m_i,par) - Ku m_iz^2 - H m_iz ] + c int |m_1 - m_2|^2 ,   c = J_int/2
(|m|=1:  c|m1-m2|^2 = J_int (1 - m1.m2)).  Same Neel-DMI sign convention and spectral derivatives as gl_gpu/params
(a single cycloid in layer i has q_i = D_i/(2A), E = -D_i^2/(4A) per site, exactly, for |m|=1).

Scheme: g = dE/dm (spectral, quadratic part 2 L_q m) ; g_t = g - (g.m) m (tangent projection, per site) ;
        m <- normalise( m - dt (1 + 2 dt (L+S))^{-1} g_t )   with a positive shift S >= -min eig(L_q) so the
preconditioner is positive definite.  Fixed points are exactly the constrained critical points (g_t = 0).
The box is (Lx, Ly) points of physical lengths (Bx, By), so any wavevector 2 pi n / B is commensurate.
Backend: NumPy, or CuPy if available (GL_BACKEND=numpy forces CPU), as in gl_gpu.
"""
import os
import numpy as np

_want = os.environ.get('GL_BACKEND', 'auto')
xp = np; ON_GPU = False
if _want in ('auto', 'cupy'):
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount(); xp, ON_GPU = cp, True
    except Exception:
        pass


def to_np(a): return xp.asnumpy(a) if ON_GPU else np.asarray(a)


def kernel6(A, D1, D2, c, qx, qy):
    """6x6 Hermitian kernel of the quadratic part (exchange, DMI, interlayer coupling); arrays qx,qy (...)"""
    sh = qx.shape
    L = np.zeros(sh + (6, 6), complex)
    q2 = qx * qx + qy * qy
    for i, D in enumerate((D1, D2)):
        o = 3 * i
        for a in range(3): L[..., o + a, o + a] += A * q2 + c
        L[..., o + 2, o + 0] += 1j * D * qx; L[..., o + 0, o + 2] += -1j * D * qx
        L[..., o + 2, o + 1] += 1j * D * qy; L[..., o + 1, o + 2] += -1j * D * qy
    for a in range(3):
        L[..., a, 3 + a] += -c; L[..., 3 + a, a] += -c
    return L


class LLG:
    def __init__(s, A=1.0, D1=0.4, D2=0.8, c=0.0, H=0.0, Ku=0.0, L=(64, 64), B=None, dt=None, shift_pad=0.05, seed=0):
        s.A, s.D1, s.D2, s.c, s.H, s.Ku = A, D1, D2, c, H, Ku
        s.Lx, s.Ly = (L, L) if np.isscalar(L) else tuple(L)
        s.Bx, s.By = (float(s.Lx), float(s.Ly)) if B is None else tuple(map(float, B))
        kx = 2 * np.pi * np.fft.fftfreq(s.Lx, s.Bx / s.Lx); ky = 2 * np.pi * np.fft.rfftfreq(s.Ly, s.By / s.Ly)
        s.qx, s.qy = np.meshgrid(kx, ky, indexing='ij')
        # odd-derivative (DMI) terms vanish on Nyquist lines (as in gl_gpu)
        nyx = np.isclose(np.abs(s.qx), np.pi * s.Lx / s.Bx) & (s.Lx % 2 == 0)
        nyy = np.isclose(np.abs(s.qy), np.pi * s.Ly / s.By) & (s.Ly % 2 == 0)
        s.Lq_np = kernel6(A, D1, D2, c, np.where(nyx, 0.0, s.qx), np.where(nyy, 0.0, s.qy))
        # diagonal (exchange, coupling) must keep the true q^2: rebuild with true q for the diagonal part
        q2 = s.qx ** 2 + s.qy ** 2
        for i in range(3):
            for o in (0, 3):
                s.Lq_np[..., o + i, o + i] = A * q2 + c
        lam = np.linalg.eigvalsh(s.Lq_np)
        s.S = float(max(0.0, -lam.min()) + shift_pad * abs(lam.min()) + 1e-12)
        # time step: preconditioned step; dt ~ 1/S-ish works as an inverse-iteration-like step (tuned)
        s.dt = dt if dt is not None else 1.0 / (2 * s.S)
        w = np.full(s.qx.shape[1], 2.0); w[0] = 1.0
        if s.Ly % 2 == 0: w[-1] = 1.0
        s.w = xp.asarray(w)[None, :]
        s.Lq = xp.asarray(s.Lq_np)
        s.Pinv = xp.asarray(np.linalg.inv(np.eye(6)[None, None] * (1 + 2 * s.dt * s.S) + 2 * s.dt * s.Lq_np))
        rng = np.random.default_rng(seed)
        m = rng.normal(size=(6, s.Lx, s.Ly)); s.m = xp.asarray(m); s.normalise()

    # ---------------------------------------------------------------- basics
    def normalise(s):
        for o in (0, 3):
            n = xp.sqrt((s.m[o:o + 3] ** 2).sum(0)); s.m[o:o + 3] /= n

    def set_m(s, m):
        s.m = xp.asarray(np.array(m, float)); s.normalise()

    def grad(s, m):
        mh = xp.moveaxis(xp.fft.rfft2(m, axes=(1, 2)), 0, -1)
        gh = 2 * xp.matmul(s.Lq, mh[..., None])[..., 0]
        g = xp.fft.irfft2(xp.moveaxis(gh, -1, 0), s=(s.Lx, s.Ly), axes=(1, 2))
        for o in (0, 3):
            g[o + 2] += -s.H - 2 * s.Ku * m[o + 2]
        return g

    def energy(s):
        """energy per site"""
        N = s.Lx * s.Ly
        mh = xp.moveaxis(xp.fft.rfft2(s.m, axes=(1, 2)), 0, -1)
        Lm = xp.matmul(s.Lq, mh[..., None])[..., 0]
        e2 = float(xp.real((xp.conj(mh) * Lm).sum(-1) * s.w).sum()) / N
        e1 = sum(-s.H * float(s.m[o + 2].sum()) - s.Ku * float((s.m[o + 2] ** 2).sum()) for o in (0, 3))
        return (e2 + e1) / N

    def step(s, n=1):
        for _ in range(n):
            g = s.grad(s.m)
            for o in (0, 3):
                mo = s.m[o:o + 3]; go = g[o:o + 3]
                go -= (go * mo).sum(0) * mo
            gh = xp.moveaxis(xp.fft.rfft2(g, axes=(1, 2)), 0, -1)
            dh = xp.matmul(s.Pinv, gh[..., None])[..., 0]
            d = xp.fft.irfft2(xp.moveaxis(dh, -1, 0), s=(s.Lx, s.Ly), axes=(1, 2))
            s.m = s.m - s.dt * d
            s.normalise()

    def residual(s):
        """max tangent-gradient norm (convergence measure)"""
        g = s.grad(s.m)
        r = 0.0
        for o in (0, 3):
            mo = s.m[o:o + 3]; go = g[o:o + 3]
            go = go - (go * mo).sum(0) * mo
            r = max(r, float(xp.sqrt((go ** 2).sum(0)).max()))
        return r

    def relax(s, nmax=20000, chunk=100, tol=1e-9, min_chunks=3, verbose=False):
        E = [s.energy()]
        for c in range(nmax // chunk):
            s.step(chunk); E.append(s.energy())
            rel = abs(E[-1] - E[-2]) / max(abs(E[-1]), 1e-30)
            if verbose and c % 10 == 0: print(f'   step {(c+1)*chunk} E={E[-1]:.8f} rel={rel:.1e} res={s.residual():.1e}', flush=True)
            if c + 1 >= min_chunks and rel < tol: break
        E = np.array(E)
        return E, bool(np.all(np.diff(E) <= 1e-12 * np.abs(E[:-1])))

    def mz_mean(s): return float(s.m[2].mean()), float(s.m[5].mean())
    def state(s): return to_np(s.m)


# ---------------------------------------------------------------- 1D band-minimum calibration (fixed length)
def spiral_energy_1d(q, A, D1, D2, c, H=0.0, N=48, nmax=6000, seed=0, init=None):
    """energy per site of the best |m|=1 two-layer spiral of wavevector q along x (one wavelength per box)"""
    B = 2 * np.pi / q
    sim = LLG(A, D1, D2, c, H, L=(N, 1), B=(B, 1.0), seed=seed)
    if init is None:
        x = np.arange(N) * B / N
        m = np.zeros((6, N, 1))
        for i, D in enumerate((D1, D2)):
            qi = D / (2 * A)
            m[3 * i + 0, :, 0] = -np.sin(q * x); m[3 * i + 2, :, 0] = np.cos(q * x)      # cycloid at the box wavevector (helicity selected by D>0)
        init = m + 1e-3 * np.random.default_rng(seed).normal(size=m.shape)
    sim.set_m(init)
    E, mono = sim.relax(nmax=nmax, chunk=200, tol=1e-12)
    return sim.energy(), sim, mono


def collinear_pair_state(na, nb, B, N, D1=None, D2=None, phase=0.0, noise=1e-3, seed=0):
    """seed: layer-1 cycloid with na periods per box, layer-2 cycloid with nb periods (along x)"""
    x = np.arange(N) * B / N
    m = np.zeros((6, N, 1))
    for o, n, ph in ((0, na, 0.0), (3, nb, phase)):
        th = 2 * np.pi * n * x / B + ph
        m[o + 0, :, 0] = -np.sin(th); m[o + 2, :, 0] = np.cos(th)
    return m + noise * np.random.default_rng(seed).normal(size=m.shape)


def best_pair_1d(A, D1, D2, c, H=0.0, B=4096.0, N=4096, start=None, nmax=4000, verbose=False):
    """integer coordinate descent over (na, nb) (periods per box of the layer-1 / layer-2 dominant cycloid) minimising the
    relaxed energy per site; returns q_minus, q_plus (rad/site), energies dict"""
    q1 = D1 / (2 * A); q2 = D2 / (2 * A)
    na0 = int(round(q1 * B / (2 * np.pi))); nb0 = int(round(q2 * B / (2 * np.pi)))
    cur = start or (na0, nb0); cache = {}
    def ev(na, nb):
        if (na, nb) not in cache:
            sim = LLG(A, D1, D2, c, H, L=(N, 1), B=(B, 1.0))
            sim.set_m(collinear_pair_state(na, nb, B, N)); sim.relax(nmax=nmax, chunk=200, tol=1e-12)
            cache[(na, nb)] = sim.energy()
        return cache[(na, nb)]
    improved = True
    while improved:
        improved = False
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            cand = (cur[0] + d[0], cur[1] + d[1])
            if cand[0] < 2 or cand[1] < 2: continue
            if ev(*cand) < ev(*cur) - 1e-12: cur = cand; improved = True; break
    return 2 * np.pi * cur[0] / B, 2 * np.pi * cur[1] / B, cur, cache


# ---------------------------------------------------------------- general chiral C6 triad seeds (box-commensurate)
def triad_modes(qm, qp, Bx, By, width=0.10, fourth=False, amps=(0.6, 0.15, 0.5, 0.0), weights=((0.8, 0.2), (0.2, 0.8))):
    """General chiral C6 state (REQUEST5 L2): q- hexagonal lattice A  (v1,v2,v3: sum 0, |v|~q-), a second q- hexagon B
    (w_i ~ v_i rotated by phi = 2 arccos(R/2), R = q+/q-) and the q+ hexagon u_i = v_i + w_i (|u|~q+, at phi/2).
    fourth=True adds the 4th orbit w_i + v_{i+1}-type sums used by the 12-fold lock (only resonant near R = 2cos 15deg).
    Wavevectors are integer vectors of the box (Bx, By).  Returns (modes, info) with modes = [(n, amp, layer_weights)]."""
    R = qp / qm; phi = 2 * np.arccos(min(R / 2, 1.0))
    sx, sy = 2 * np.pi / Bx, 2 * np.pi / By
    kk = lambda n: np.array([n[0] * sx, n[1] * sy])
    rmax = int(np.ceil(qm * (1 + width + 0.05) / min(sx, sy)))
    V = [(a, b) for a in range(-rmax, rmax + 1) for b in range(-rmax, rmax + 1) if abs(np.hypot(*kk((a, b))) / qm - 1) < width]
    rot = lambda v, t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ v
    best = None
    for i1, v1 in enumerate(V):
        k1 = kk(v1)
        for v2 in V[i1 + 1:]:
            k2 = kk(v2); c = k1 @ k2 / (np.linalg.norm(k1) * np.linalg.norm(k2))
            if not (-0.58 < c < -0.42): continue
            v3 = (-(v1[0] + v2[0]), -(v1[1] + v2[1])); vs = [v1, v2, v3]
            if abs(np.hypot(*kk(v3)) / qm - 1) > width: continue
            cost = sum((np.linalg.norm(kk(v)) / qm - 1) ** 2 for v in vs)
            ws = []; us = []
            for v in vs:
                tgt = rot(kk(v), phi); wn = (int(round(tgt[0] / sx)), int(round(tgt[1] / sy)))
                bw = None
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        w = (wn[0] + da, wn[1] + db); u = (v[0] + w[0], v[1] + w[1])
                        cc = (np.linalg.norm(kk(w)) / qm - 1) ** 2 + (np.linalg.norm(kk(u)) / qp - 1) ** 2
                        if bw is None or cc < bw[0]: bw = (cc, w, u)
                cost += bw[0]; ws.append(bw[1]); us.append(bw[2])
            if best is None or cost < best[0]: best = (cost, vs, ws, us)
    cost, vs, ws, us = best
    A_, B_, C_, D_ = amps
    modes = [(v, A_, weights[0]) for v in vs] + [(w, B_, weights[0]) for w in ws] + [(u, C_, weights[1]) for u in us]
    if fourth:
        u4 = [(w[0] + vs[(i + 1) % 3][0], w[1] + vs[(i + 1) % 3][1]) for i, w in enumerate(ws)]
        modes += [(u, D_, weights[1]) for u in u4]
    meas = lambda a, b: np.degrees(np.arccos(np.clip(kk(a) @ kk(b) / np.linalg.norm(kk(a)) / np.linalg.norm(kk(b)), -1, 1)))
    info = dict(cost=float(cost), phi_target=float(np.degrees(phi)), phi_box=float(meas(vs[0], ws[0])),
                R_target=float(R), R_box=[float(np.linalg.norm(kk(u)) / np.linalg.norm(kk(v))) for u, v in zip(us, vs)],
                v=vs, w=ws, u=us)
    return modes, info


def modes_to_field(modes, Lx, Ly, Bx, By, m0=0.0, sign=-1.0):
    """unit-length seed: m_i = normalise( (0,0,m0) + sum_j sign*a_j*sqrt(w_ij)*cycloid_j ) (phases 0, amplitudes negative: triad phase sum pi)"""
    X, Y = np.meshgrid(np.arange(Lx) * Bx / Lx, np.arange(Ly) * By / Ly, indexing='ij')
    m = np.zeros((6, Lx, Ly)); m[2] = m0; m[5] = m0
    for n, a, w in modes:
        k = np.array([2 * np.pi * n[0] / Bx, 2 * np.pi * n[1] / By]); kn = np.hypot(*k)
        if kn == 0: continue
        ph = k[0] * X + k[1] * Y
        for i in range(2):
            amp = sign * a * np.sqrt(w[i])
            m[3 * i + 2] += amp * np.cos(ph); m[3 * i + 0] -= amp * k[0] / kn * np.sin(ph); m[3 * i + 1] -= amp * k[1] / kn * np.sin(ph)
    return m
