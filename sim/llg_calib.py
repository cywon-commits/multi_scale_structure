"""Fixed-length calibration: preferred wavevectors q-, q+ of the coupled layers (collinear 1D pair states), R = q+/q-,
Lifshitz-type merging coupling J_L, and the saturation field H_sat.  Units: A = 1, q1 = 2 pi/32, D1 = 2 A q1, rho = D2/D1,
J_int = 2c.   usage: python llg_calib.py rho"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import llg
A = 1.0; q1 = 2 * np.pi / 32; D1 = 2 * A * q1
B, N = 1024.0, 512


def measure(rho, J, H=0.0):
    qm, qp, cur, cache = llg.best_pair_1d(A, D1, rho * D1, J / 2, H, B=B, N=N, nmax=3000)
    return qm / q1, qp / q1, cur


def locked_energy(rho, J, H=0.0):
    """best energy of the locked (na = nb) collinear state, scanning n near the mid wavevector"""
    D2 = rho * D1; qbar = 0.5 * (D1 + D2) / (2 * A); n0 = int(round(qbar * B / (2 * np.pi))); best = None
    for n in range(n0 - 4, n0 + 5):
        sim = llg.LLG(A, D1, D2, J / 2, H, L=(N, 1), B=(B, 1.0)); sim.set_m(llg.collinear_pair_state(n, n, B, N))
        sim.relax(nmax=3000, chunk=200, tol=1e-11); e = sim.energy()
        if best is None or e < best[0]: best = (e, n)
    return best


def pair_energy(rho, J, H=0.0):
    qm, qp, cur, cache = llg.best_pair_1d(A, D1, rho * D1, J / 2, H, B=B, N=N, nmax=3000)
    return cache[cur], qm / q1, qp / q1, cur


def find_JL(rho, lo=0.0, hi=0.4, iters=7):
    """J at which the locked single-wavevector state becomes lower than the incommensurate collinear pair (first-order merge)"""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if locked_energy(rho, mid)[0] < pair_energy(rho, mid)[0]: hi = mid
        else: lo = mid
    return hi


def H_sat(rho, J, hi=None, iters=10):
    """field above which the relaxed state (started from the zero-field pair state) has <m_z> = 1 in both layers"""
    D2 = rho * D1
    hi = hi or 3 * np.pi ** 2 * (max(D1, D2) ** 2) / (16 * A); lo = 0.0
    qm, qp, cur = measure(rho, J)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        sim = llg.LLG(A, D1, D2, J / 2, mid, L=(N, 1), B=(B, 1.0))
        sim.set_m(llg.collinear_pair_state(cur[0], cur[1], B, N)); sim.relax(nmax=6000, chunk=200, tol=1e-12)
        mz = sim.mz_mean()
        if min(mz) > 0.999: hi = mid
        else: lo = mid
    return hi


if __name__ == '__main__':
    rho = float(sys.argv[1]); out = dict(rho=rho)
    t = time.time()
    JL = find_JL(rho); out['J_L'] = JL; print('rho', rho, 'J_L', JL, round(time.time() - t), 's', flush=True)
    rows = []
    for fr in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        ep, qm, qp, cur = pair_energy(rho, fr * JL); el, nl = locked_energy(rho, fr * JL)
        rows.append(dict(J_over_JL=fr, J=fr * JL, q_minus=qm, q_plus=qp, R=qp / qm, n=cur, E_pair=ep, E_locked=el, n_locked=nl))
        print(f'J/J_L={fr:.2f} J={fr*JL:.4f} q-={qm:.3f} q+={qp:.3f} R={qp/qm:.3f} n={cur} E_pair={ep:.5f} E_locked={el:.5f}', flush=True)
    out['rows'] = rows
    out['H_sat_J0'] = H_sat(rho, 0.0); print('H_sat(J=0)', out['H_sat_J0'], flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'runs', f'llg_calib_rho{rho}.json'), 'w'), indent=1, default=float)
