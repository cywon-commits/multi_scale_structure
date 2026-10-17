"""REQUEST5b: rotation lock of the q+ hexagon relative to the q- hexagon (fixed-length model).
   python llg_m.py M1 delta_deg        # q- SkX + q+ SkX rotated by theta = phi/2 + delta
   python llg_m.py M2                  # converge the report5 (a),(b) runs to residual < 1e-7
   python llg_m.py M3 seed             # q- SkX only + random layer 2: which rotation does q+ nucleate with?
Point: rho = 1.5, J = 0.0022 (R = 1.5, phi/2 = 41.4 deg), H = 0.3 H_sat = 0.0694 (as report5)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import llg, llg_run as lr
A = 1.0; q1 = 2 * np.pi / 32; D1 = 2 * A * q1
RUNS = os.path.join(os.path.dirname(__file__), '..', 'runs')
RHO, J, HS = 1.5, 0.0022, 0.2314
HFRAC = float(os.environ.get('HFRAC', '0.3'))
H = HFRAC * HS
QM, QP = q1, 1.5 * q1
R = QP / QM; PHI = np.degrees(2 * np.arccos(R / 2)); PHI2 = PHI / 2
NPER = 12
BX = NPER * 2 * np.pi / QM; BY = np.sqrt(3) * BX
LX = int(round(BX / 1.5)); LY = int(round(BY / 1.5)); LX += LX % 2; LY += LY % 2


def kk(n): return np.array([2 * np.pi * n[0] / BX, 2 * np.pi * n[1] / BY])


def qm_lattice():
    n = NPER
    return [(n, 0), (-n // 2, 3 * n // 2), (-n // 2, -3 * n // 2)]      # hexagon: exact in the sqrt3 box


def plus_lattice(theta_deg):
    """integer approximant of the q+ hexagon rotated by theta from the q- hexagon (u1+u2+u3 = 0); returns vectors and errors"""
    t = np.radians(theta_deg); rot = lambda v, a: np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])
    v1 = kk(qm_lattice()[0]); tgt1 = rot(v1, t) * R; tgt2 = rot(tgt1, 2 * np.pi / 3)
    n1 = np.rint(tgt1 / (2 * np.pi / np.array([BX, BY]))).astype(int); n2 = np.rint(tgt2 / (2 * np.pi / np.array([BX, BY]))).astype(int)
    best = None
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                for d in range(-3, 4):
                    u1 = (n1[0] + a, n1[1] + b); u2 = (n2[0] + c, n2[1] + d); u3 = (-(u1[0] + u2[0]), -(u1[1] + u2[1]))
                    ks = [kk(u) for u in (u1, u2, u3)]
                    cost = sum((np.linalg.norm(k) / QP - 1) ** 2 for k in ks)
                    ang = np.degrees(np.arctan2(ks[0][1], ks[0][0]) - np.arctan2(v1[1], v1[0]))
                    cost += (np.radians(((ang - theta_deg + 180) % 360) - 180)) ** 2
                    if best is None or cost < best[0]: best = (cost, [u1, u2, u3], ang)
    cost, us, ang = best
    lens = [float(np.linalg.norm(kk(u)) / QP - 1) for u in us]
    return us, dict(theta_target=theta_deg, theta_box=float(((ang + 180) % 360) - 180), len_err=lens, cost=float(cost))


def seed_field(theta_deg, with_plus=True):
    us, info = plus_lattice(theta_deg) if with_plus else ([], {})
    modes = [(v, 0.6, (0.8, 0.2)) for v in qm_lattice()] + [(u, 0.5, (0.2, 0.8)) for u in us]
    return llg.modes_to_field(modes, LX, LY, BX, BY, m0=0.3), info


def measure(sim):
    m = sim.state(); pk = lr.find_peaks_llg(m, BX, BY)
    r1 = lr.ring_orbits(pk, QM); r2 = lr.ring_orbits(pk, QP)
    rel = None
    if r1 and r2: rel = float(((r2['angle_mod60'] - r1['angle_mod60'] + 30) % 60) - 30)
    t = lr.measure_triad2(pk, QM, QP, PHI)
    return dict(E=sim.energy(), res=sim.residual(), rel=rel, A_big=(t or {}).get('A_big'), A_plus=(t or {}).get('A_plus_phi'), A_minus=(t or {}).get('A_minus_phi'),
                chi=(t or {}).get('chi'), qp_amp=(t or {}).get('qp_amp'), top=[(list(p['n']), round(p['k'] / QM, 3), round(p['angle'], 1), round(p['a'], 3)) for p in pk[:10]]), m


def run(name, m0, nmax=40000, chunk=2500, tol_res=1e-6, extra=None):
    t0 = time.time()
    sim = llg.LLG(A, D1, RHO * D1, J / 2, H, 0.0, L=(m0.shape[1], m0.shape[2]), B=(BX, BY)); sim.set_m(m0)
    hist = []
    for c in range(nmax // chunk):
        sim.step(chunk); r, m = measure(sim); r['step'] = (c + 1) * chunk; hist.append(r)
        print(f"  {name} step {r['step']}: E={r['E']:.7f} res={r['res']:.1e} rel_angle={r['rel']} A0={r['A_big']} A+phi={r['A_plus']} A-phi={r['A_minus']}", flush=True)
        if r['res'] < tol_res: break
        if c >= 3 and abs(hist[-1]['E'] - hist[-2]['E']) < 1e-10 and r['res'] < 1e-5: break
    np.save(os.path.join(RUNS, 'fields', f'llgM_{name}_m.npy'), m.astype(np.float32))
    out = dict(name=name, box=[BX, BY, LX, LY], rho=RHO, J=J, H=H, wall=time.time() - t0, hist=hist, final=hist[-1], **(extra or {}))
    json.dump(out, open(os.path.join(RUNS, f'llgM_{name}.json'), 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    t = sys.argv[1]
    print(f'point: rho={RHO} J={J} H={H:.4f} R={R:.3f} phi={PHI:.2f} phi/2={PHI2:.2f}; box {LX}x{LY} ({BX:.0f}x{BY:.0f})', flush=True)
    if t == 'M1':
        d = float(sys.argv[2]); theta = PHI2 + d
        m0, info = seed_field(theta); print('seed', info, flush=True)
        run(f'M1_d{d:+.0f}', m0, extra=dict(delta=d, theta=theta, seed_info=info))
    elif t == 'M4':
        # near-saturation rotation scan: E relative to the saturated state (-2H per site), same box/delta set as M1
        d = float(sys.argv[2]); theta = PHI2 + d; f = (1 - HFRAC) / 0.7
        us, info = plus_lattice(theta)
        modes = [(v, 0.6 * f, (0.8, 0.2)) for v in qm_lattice()] + [(u, 0.5 * f, (0.2, 0.8)) for u in us]
        m0 = llg.modes_to_field(modes, LX, LY, BX, BY, m0=0.85)
        name = f'M4_h{HFRAC}_d{d:+.0f}'; t0 = time.time()
        sim = llg.LLG(A, D1, RHO * D1, J / 2, H, 0.0, L=(LX, LY), B=(BX, BY)); sim.set_m(m0)
        Esat = -2 * H; hist = []
        for c in range(12):
            sim.step(2500); r, m = measure(sim); r['step'] = (c + 1) * 2500; r['dE_sat'] = r['E'] - Esat; r['mz'] = sim.mz_mean(); hist.append(r)
            print(f"  {name} step {r['step']}: E-Esat={r['dE_sat']:.3e} res={r['res']:.1e} mz={np.round(r['mz'],4)} rel={r['rel']} A0={r['A_big']} A+phi={r['A_plus']} A-phi={r['A_minus']}", flush=True)
            if c >= 2 and abs(hist[-1]['E'] - hist[-2]['E']) < 1e-11 and r['res'] < 1e-7: break
        np.save(os.path.join(RUNS, 'fields', f'llgM_{name}_m.npy'), m.astype(np.float32))
        json.dump(dict(name=name, hfrac=HFRAC, H=H, delta=d, theta=theta, seed_info=info, box=[BX, BY, LX, LY], hist=hist, final=hist[-1], Esat=Esat, wall=time.time() - t0),
                  open(os.path.join(RUNS, f'llgM_{name}.json'), 'w'), indent=1, default=float)
    elif t == 'M2':
        for nm in ('a_triad', 'b_skx2'):
            m0 = np.load(os.path.join(RUNS, 'fields', f'llg_L2_rho1.5_J0.0022_H0.3_{nm}_m.npy')).astype(float)
            # report5 box: nper=10 -> different box; reuse its own box
            jj = json.load(open(os.path.join(RUNS, 'llg_L2_rho1.5_J0.0022_H0.3.json')))
            bx, by, lx, ly = jj['box']
            sim = llg.LLG(A, D1, RHO * D1, J / 2, H, 0.0, L=(lx, ly) if False else (m0.shape[1], m0.shape[2]), B=(bx, by)); sim.set_m(m0)
            hist = []
            for c in range(40):
                sim.step(2500); mm = sim.state(); pk = lr.find_peaks_llg(mm, bx, by)
                qm0 = jj['calib_row']['q_minus'] * q1; qp0 = jj['calib_row']['q_plus'] * q1
                t_ = lr.measure_triad2(pk, qm0, qp0, jj['box_info']['phi_box']); r1 = lr.ring_orbits(pk, qm0); r2 = lr.ring_orbits(pk, qp0)
                rel = float(((r2['angle_mod60'] - r1['angle_mod60'] + 30) % 60) - 30) if r1 and r2 else None
                h = dict(step=(c + 1) * 2500, E=sim.energy(), res=sim.residual(), rel=rel, A0=(t_ or {}).get('A_big'), Ap=(t_ or {}).get('A_plus_phi'), Am=(t_ or {}).get('A_minus_phi'), qp=(t_ or {}).get('qp_amp'))
                hist.append(h); print(f"  {nm} step {h['step']}: E={h['E']:.7f} res={h['res']:.1e} rel={rel} A0={h['A0']} A+={h['Ap']} A-={h['Am']}", flush=True)
                if h['res'] < 1e-7: break
            json.dump(dict(name=nm, hist=hist, box=jj['box']), open(os.path.join(RUNS, f'llgM_M2_{nm}.json'), 'w'), indent=1, default=float)
            np.save(os.path.join(RUNS, 'fields', f'llgM_M2_{nm}_m.npy'), mm.astype(np.float32))
    elif t == 'M3':
        s = int(sys.argv[2]); m0, _ = seed_field(0.0, with_plus=False)
        rng = np.random.default_rng(500 + s); m0[3:6] = rng.normal(size=(3, LX, LY)); m0[5] += 0.5
        run(f'M3_s{s}', m0, extra=dict(seed=s))

