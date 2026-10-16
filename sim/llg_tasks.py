"""REQUEST5 tasks on the fixed-length model.  usage:
   python llg_tasks.py L2 rho J_over_JL H_over_Hsat nper [seeds]
   (J_L, H_sat are read from runs/llg_calib_rho<rho>.json written by llg_calib.py)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import llg, llg_run as lr

A = 1.0; q1 = 2 * np.pi / 32; D1 = 2 * A * q1
RUNS = os.path.join(os.path.dirname(__file__), '..', 'runs')


def calib(rho):
    return json.load(open(os.path.join(RUNS, f'llg_calib_rho{rho}.json')))


def setup(rho, nper, gridh=1.5):
    qm = q1; qp = rho * q1
    Bx = nper * 2 * np.pi / qm; By = np.sqrt(3) * Bx
    Lx = int(round(Bx / gridh)); Ly = int(round(By / gridh))
    Lx += Lx % 2; Ly += Ly % 2
    return qm, qp, Bx, By, Lx, Ly


def energy_summary(sim, qm, qp, phi_deg, Bx, By):
    m = sim.state()
    peaks = lr.find_peaks_llg(m, Bx, By)
    return dict(E=sim.energy(), mz=sim.mz_mean(), top=[(list(p['n']), round(p['k'] / qm, 3), round(p['angle'], 1), round(p['a'], 3)) for p in peaks[:16]],
                triad=lr.measure_triad2(peaks, qm, qp, phi_deg)), m


def L2(rho, J, hfrac, nper, nseeds=6, nmax=20000, tag=''):
    """J is the absolute interlayer coupling; the preferred wavevectors q-, q+ are taken from the 1D calibration row nearest to J"""
    cal = calib(rho); JL = cal['J_L']; Hs = cal['H_sat_J0']; jfrac = round(J, 5)
    row = min(cal['rows'], key=lambda r: abs(r['J'] - J))
    H = hfrac * Hs; c = J / 2
    qm0, qp0, Bx, By, Lx, Ly = setup(rho, nper)
    qm = row['q_minus'] * q1; qp = row['q_plus'] * q1
    Bx = nper * 2 * np.pi / qm; By = np.sqrt(3) * Bx
    Lx = int(round(Bx / 1.5)); Ly = int(round(By / 1.5)); Lx += Lx % 2; Ly += Ly % 2
    R = qp / qm; phi = np.degrees(2 * np.arccos(R / 2))
    out = dict(rho=rho, J=J, J_L=JL, H=H, H_sat=Hs, box=[Bx, By, Lx, Ly], R=R, phi=phi, calib_row=row, runs={})
    print(f'== rho={rho} (R={R:.3f}, phi={phi:.2f}) J={J:.4f} (J_L(bisect)={JL:.4f}) H={H:.4f} (H/Hsat={hfrac}) box {Lx}x{Ly} ({Bx:.0f}x{By:.0f})', flush=True)
    fn = os.path.join(RUNS, f'llg_L2_rho{rho}_J{jfrac}_H{hfrac}{tag}.json')
    seeds = {}
    for name, amps in (('a_triad', (0.6, 0.15, 0.5, 0.0)), ('b_skx2', (0.6, 0.0, 0.5, 0.0))):
        modes, info = llg.triad_modes(qm, qp, Bx, By, amps=amps)
        m0 = llg.modes_to_field(modes, Lx, Ly, Bx, By, m0=0.3 if H > 0 else 0.0)
        seeds[name] = (m0, info)
    out['box_info'] = {k: v for k, v in seeds['a_triad'][1].items() if k in ('phi_target', 'phi_box', 'R_target', 'R_box', 'cost', 'v', 'w', 'u')}
    for i in range(nseeds):
        rng = np.random.default_rng(100 + i); m = rng.normal(size=(6, Lx, Ly))     # fully random spins (a near-uniform start collapses to the saturated state)
        seeds[f'noise{i}'] = (m, None)
    only = os.environ.get('ONLY')
    for name, (m0, info) in seeds.items():
        if only and not name.startswith(only): continue
        t0 = time.time()
        sim, E, mono = lr.relax_state(A, D1, rho * D1, c, H, m0, Bx, By, nmax=nmax, chunk=250, tol=1e-9)
        r, m = energy_summary(sim, qm, qp, out['box_info']['phi_box'], Bx, By)
        r.update(steps=int((len(E) - 1) * 250), monotone=mono, wall=time.time() - t0)
        np.save(os.path.join(RUNS, 'fields', f'llg_L2_rho{rho}_J{jfrac}_H{hfrac}{tag}_{name}_m.npy'), m.astype(np.float32))
        out['runs'][name] = r
        t = r['triad'] or {}
        print(f"  {name}: E={r['E']:.6f} <mz>={np.round(r['mz'],3)} chi={t.get('chi')} A0={t.get('A_big',0):.3f} A+phi={t.get('A_plus_phi',0):.3f} A-phi={t.get('A_minus_phi',0):.3f} qp_rel={t.get('qp_orbit_rel_mod60')} steps={r['steps']} mono={mono} {r['wall']:.0f}s", flush=True)
        json.dump(out, open(fn, 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    t = sys.argv[1]
    if t == 'L2':
        L2(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 6)
