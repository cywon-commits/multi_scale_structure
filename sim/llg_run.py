"""2D fixed-length runs for REQUEST5: seeds, relaxation, peak/orbit measurement.  Importable helpers + CLI in llg_tasks.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import llg

A = 1.0; q1 = 2 * np.pi / 32; D1 = 2 * A * q1


def find_peaks_llg(m, Bx, By, rel=1e-4, nmax=60):
    """local maxima of S(q) = sum_alpha |m^_alpha|^2 (both layers); amplitude of the z Fourier coefficient per layer:
    a_i = 2|m^_iz|/N.  n = integer box vector (one of +-n), k physical, angle in [0,180)"""
    Lx, Ly = m.shape[1:]; N = Lx * Ly
    mh = np.fft.fft2(m, axes=(1, 2)); S = (np.abs(mh) ** 2).sum(0); S[0, 0] = 0
    loc = np.ones_like(S, bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy: loc &= S >= np.roll(np.roll(S, dx, 0), dy, 1)
    idx = np.argwhere(loc & (S > rel * S.max()))
    nx = np.fft.fftfreq(Lx, 1.0 / Lx).astype(int); ny = np.fft.fftfreq(Ly, 1.0 / Ly).astype(int)
    seen = set(); out = []
    for ix, iy in idx:
        n = (int(nx[ix]), int(ny[iy])); key = n if (n[0] > 0 or (n[0] == 0 and n[1] > 0)) else (-n[0], -n[1])
        if key in seen: continue
        seen.add(key)
        a = [2 * abs(mh[3 * i + 2, ix, iy]) / N for i in range(2)]
        k = np.array([2 * np.pi * key[0] / Bx, 2 * np.pi * key[1] / By])
        out.append(dict(n=key, k=float(np.hypot(*k)), angle=float(np.degrees(np.arctan2(k[1], k[0])) % 180), a1=a[0], a2=a[1], a=float(np.hypot(*a)),
                        S=float(S[ix, iy])))
    out.sort(key=lambda d: -d['a'])
    return out[:nmax]


def ring_orbits(peaks, kref, tol=0.08):
    """peaks near |k| = kref grouped into the two C6 orbits (angles mod 60): returns dict(big, small, offset...) """
    sel = [p for p in peaks if abs(p['k'] / kref - 1) < tol]
    if len(sel) < 3: return None
    a = np.array([p['a'] for p in sel]); ang = np.array([p['angle'] for p in sel])
    top = ang[np.argsort(a)[::-1][:3]]
    ph = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(top) * 6)))) / 6
    mod = lambda x: ((x - ph + 30) % 60) - 30
    maj = np.abs(mod(ang)) < 15
    return dict(angle_mod60=float(ph % 60), major=float(a[maj].mean()), minor=float(a[~maj].mean()) if (~maj).any() else 0.0, n=len(sel))


def measure_triad(peaks, qm, qp):
    """phi (angle between the two q- orbits mod 60, in (-30,30]), q+ orbit angle relative to the big q- orbit, amplitudes"""
    r1 = ring_orbits(peaks, qm); r2 = ring_orbits(peaks, qp)
    out = dict(qm=r1, qp=r2)
    if r1:
        sel = [p for p in peaks if abs(p['k'] / qm - 1) < 0.08]
        a = np.array([p['a'] for p in sel]); ang = np.array([p['angle'] for p in sel])
        ph = r1['angle_mod60']; mod = lambda x: ((x - ph + 30) % 60) - 30
        minor = ang[np.abs(mod(ang)) >= 15]
        if len(minor):
            amin = a[np.abs(mod(ang)) >= 15]; mm = np.degrees(np.angle(np.mean(amin[:, None] * 0 + np.exp(1j * np.radians(minor) * 6)))) / 6
            out['phi_mod60'] = float(((mm - ph + 30) % 60) - 30)
    if r1 and r2:
        out['qp_rel_angle_mod60'] = float(((r2['angle_mod60'] - r1['angle_mod60'] + 30) % 60) - 30)
    return out


def relax_state(A_, D1_, D2_, c, H, m0, Bx, By, nmax=8000, chunk=200, tol=1e-9, Ku=0.0, verbose=False):
    Lx, Ly = m0.shape[1:]
    sim = llg.LLG(A_, D1_, D2_, c, H, Ku, L=(Lx, Ly), B=(Bx, By))
    sim.set_m(m0)
    E, mono = sim.relax(nmax=nmax, chunk=chunk, tol=tol, verbose=verbose)
    return sim, E, mono


def _amp_at(peaks, kref, angle, tol_k=0.08, tol_ang=6.0):
    """largest peak amplitude near (|k| = kref, line angle = angle mod 60 pattern: any of angle + 60 j)"""
    best = 0.0
    for p in peaks:
        if abs(p['k'] / kref - 1) > tol_k: continue
        d = ((p['angle'] - angle + 30) % 60) - 30
        if abs(d) < tol_ang: best = max(best, p['a'])
    return best


def measure_triad2(peaks, qm, qp, phi_deg):
    """REQUEST5 addendum: read the q- peak amplitudes at +phi and -phi from the big q- orbit (chi = +1 if +phi is occupied),
    and the q+ orbit angle relative to the big q- orbit (expected +-phi/2 mod 60)."""
    r1 = ring_orbits(peaks, qm)
    if not r1: return None
    th0 = r1['angle_mod60']
    A0 = _amp_at(peaks, qm, th0); Ap = _amp_at(peaks, qm, th0 + phi_deg); Am = _amp_at(peaks, qm, th0 - phi_deg)
    out = dict(theta0=th0, A_big=A0, A_plus_phi=Ap, A_minus_phi=Am)
    out['chi'] = int(np.sign(Ap - Am)) if max(Ap, Am) > 2 * min(Ap, Am) + 1e-12 else 0
    r2 = ring_orbits(peaks, qp)
    if r2:
        rel = ((r2['angle_mod60'] - th0 + 30) % 60) - 30
        out['qp_orbit_rel_mod60'] = float(rel); out['qp_amp'] = r2['major']
        out['expected_phi_half_mod60'] = [float(((phi_deg / 2 + 30) % 60) - 30), float(((-phi_deg / 2 + 30) % 60) - 30)]
    return out
