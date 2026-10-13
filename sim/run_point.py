"""Run one parameter point: calibrate H (step 1), relax from several seeds at the same H (step 2), save npz + json.

usage: python run_point.py configs/P1.json [--L 512] [--seeds noise,qc,hex] [--tag x]
Config keys: name, c_rel, rho, K, g, (L, tau, nmax, tol, calib_iters, nrelax_calib, seeds)
dt = tau/|r|  (semi-implicit scheme: dt limited by the explicit nonlinear part ~ |r|)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from params import landau_to_model
import gl_gpu as G
import analysis as an

ROOT = os.path.join(os.path.dirname(__file__), '..')


def make_sim(p, L, tau, seed_kind, seed=0, m0=None, kappa=12.0):
    dt = tau / p['r_abs']
    if seed_kind.startswith('noise') and seed_kind[5:].isdigit():
        seed = int(seed_kind[5:])          # 'noise3' = random small-noise seed #3
    sim = G.GLBilayer(p, L=L, dt=dt, seed=seed, stab=kappa * p['r_abs'])
    cr = p['landau']['c_rel']
    noise = 1e-4 * np.sqrt(p['r_abs']) * np.random.default_rng(seed).normal(size=(6, L, L))
    if seed_kind in ('qc', 'hex'):        # box-commensurate (integer-vector) approximant seeds, triads close exactly
        modes, info = (an.qc_int_modes if seed_kind == 'qc' else an.hex_int_modes)(p, L)
        sim.seed_info = info
        m = an.int_seed(modes, L, p, cr) + noise
        m[2] += p['M_target']; m[5] += p['M_target']
        sim.m = G.xp.asarray(m)
    elif seed_kind == 'qcr':              # old: continuous wavevectors rounded independently
        m = an.qc_seed(p, L) + 1e-4 * np.sqrt(p['r_abs']) * np.random.default_rng(seed).normal(size=(6, L, L))
        m[2] += p['M_target']; m[5] += p['M_target']
        sim.m = G.xp.asarray(m)
    elif seed_kind == 'hexr':
        m = an.hex_seed(p, L) + 1e-4 * np.sqrt(p['r_abs']) * np.random.default_rng(seed).normal(size=(6, L, L))
        m[2] += p['M_target']; m[5] += p['M_target']
        sim.m = G.xp.asarray(m)
    elif seed_kind in ('sp1', 'sp2', 'sp2c'):   # g=0 benchmarks: single spiral / two orthogonal / two collinear (exact kernel eigenvectors)
        nm = int(round(p['q_minus'] * L / (2 * np.pi))); npl = int(round(p['q_plus'] * L / (2 * np.pi)))
        specs = {'sp1': [((nm, 0), 1.0)], 'sp2': [((nm, 0), 1.0), ((0, npl), 1.0)], 'sp2c': [((nm, 0), 1.0), ((npl, 0), 1.0)]}[seed_kind]
        sim.m = G.xp.asarray(an.eig_spiral_seed(p, L, specs) + noise)
    elif seed_kind == 'file':
        sim.m = G.xp.asarray(m0)
    return sim


def relax_checked(sim, nmax, chunk, tol, p, verbose=True, min_chunks=6):
    """relax until the energy gain per chunk, measured against the modulation energy E-E_uniform, is < tol AND the
    leading Fourier amplitude changes by < sqrt(tol) (relative); returns (E history, monotone flag)"""
    E = [sim.energy()]
    Eu = sim.uniform_energy()[0]
    amp_old = None
    for c in range(nmax // chunk):
        sim.step(chunk)
        E.append(sim.energy())
        rel = abs(E[-1] - E[-2]) / max(abs(E[-1] - Eu), 1e-300)
        mz = G.xp.fft.rfft2(sim.m[2] + sim.m[5]); amp = float(G.xp.abs(mz).max())
        damp = abs(amp - amp_old) / amp if amp_old else 1.0
        amp_old = amp
        if verbose and c % 5 == 0:
            print(f'    step {(c+1)*chunk:7d} dE/r^2={(E[-1]-Eu)/p["r_abs"]**2: .6f} rel={rel:.1e} damp={damp:.1e} <mz>={sim.mz_mean()[0]:.5f},{sim.mz_mean()[1]:.5f}', flush=True)
        if c + 1 >= min_chunks and rel < tol and damp < np.sqrt(tol):
            break
    E = np.array(E)
    return E, bool(np.all(np.diff(E) <= 1e-12 * np.abs(E[:-1])))


def calibrate(sim, p, iters, nrelax, chunk, verbose=True):
    """Step 1: iterate (H1,H2) so layer-resolved <m_z> = M_target. Secant update per layer on warm-started relaxations."""
    M = p['M_target']
    hist = []
    prevH = [None, None]; prevm = [None, None]
    for it in range(iters):
        E, mono = relax_checked(sim, nrelax, chunk, 1e-7, p, verbose=False)
        mz = list(sim.mz_mean())
        H = [sim.p['H1'], sim.p['H2']]
        err = max(abs(mz[0] - M), abs(mz[1] - M)) / M
        hist.append(dict(it=it, H=H, mz=mz, err=err, E=float(E[-1]) / p['r_abs'] ** 2, monotone=mono))
        if verbose:
            print(f'  calib {it}: H=({H[0]:.5e},{H[1]:.5e}) <mz>=({mz[0]:.5f},{mz[1]:.5f}) target {M:.5f} err={err:.2e} mono={mono}', flush=True)
        if err < 2e-3:
            break
        newH = []
        for i in range(2):
            if prevH[i] is not None and abs(mz[i] - prevm[i]) > 1e-12:
                chi = (H[i] - prevH[i]) / (mz[i] - prevm[i])
            else:
                chi = (2 * sim.p['r1' if i == 0 else 'r2'] + 2 * sim.p['c'] + 12 * sim.p['u'] * M * M)
                chi = abs(chi)
            prevH[i], prevm[i] = H[i], mz[i]
            newH.append(H[i] + chi * (M - mz[i]))
        sim.set_fields(*newH)
    return hist


def analyse(sim, p, L):
    m = sim.state()
    peaks = an.find_peaks(m, p)
    S = an.structure_factors(m)
    rad = an.radial_profile(S)
    mz = sim.mz_mean()
    Eu, mu = sim.uniform_energy()
    res = dict(E_over_r2=sim.energy() / p['r_abs'] ** 2, dE_over_r2=(sim.energy() - Eu) / p['r_abs'] ** 2, E_uniform_over_r2=Eu / p['r_abs'] ** 2, mz=list(mz), g_eff=[G.g_eff(p, v) for v in mz],
               peaks=peaks[:60], orbits=an.orbit_summary(peaks, p))
    return res, S, rad, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('--L', type=int); ap.add_argument('--seeds'); ap.add_argument('--tag', default='')
    ap.add_argument('--tau', type=float); ap.add_argument('--nmax', type=int); ap.add_argument('--outdir', default=os.path.join(ROOT, 'runs'))
    ap.add_argument('--tol', type=float); ap.add_argument('--no-calib', action='store_true'); ap.add_argument('--g_target_only', action='store_true')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    L = a.L or cfg.get('L', 512)
    tau = a.tau or cfg.get('tau', 1.0); kappa = cfg.get('kappa', 12.0)
    nmax = a.nmax or cfg.get('nmax', 20000)
    seeds = (a.seeds or ','.join(cfg.get('seeds', ['noise', 'qc', 'hex']))).split(',')
    p = landau_to_model(cfg['c_rel'], cfg['rho'], cfg['K'], cfg['g'], q1=2 * np.pi / cfg.get('lambda1', 32))
    os.makedirs(a.outdir, exist_ok=True)
    name = f"{cfg['name']}_c{cfg['c_rel']}_rho{cfg['rho']}_K{cfg['K']}_g{cfg['g']}_L{L}{a.tag}"
    print('==', name, ' r_abs=%.3e dt=%.1f' % (p['r_abs'], tau / p['r_abs']), flush=True)
    out = dict(name=name, cfg=cfg, L=L, tau=tau, params={k: v for k, v in p.items() if k != 'landau'}, runs={})
    # ---- step 1: calibrate fields with the first seed that has g>0 (noise seed for g=0 -> fields all zero)
    Hcal = None
    if cfg['g'] == 0 or a.no_calib:
        pass   # g=0: M=0, H1=H2=0 exactly (landau_to_model)
    else:
        ck = 'qc' if 'qc' in seeds else seeds[0]
        sim = make_sim(p, L, tau, ck, kappa=kappa)
        out['calib'] = calibrate(sim, p, cfg.get('calib_iters', 8), cfg.get('nrelax_calib', 3000), 250)
        p['H1'], p['H2'] = sim.p['H1'], sim.p['H2']
    for sk in seeds:
        t0 = time.time()
        print(' seed', sk, flush=True)
        sim = make_sim(p, L, tau, sk, kappa=kappa)
        sim.set_fields(p['H1'], p['H2'])
        E, mono = relax_checked(sim, nmax, 250, a.tol or cfg.get('tol', 2e-9), p)
        res, S, rad, m = analyse(sim, p, L)
        res.update(monotone=mono, steps=int((len(E) - 1) * 250), dE_last=float(abs(E[-1] - E[-2]) / abs(E[-1])), wall=time.time() - t0,
                   E_hist=(E[::max(1, len(E) // 200)] / p['r_abs'] ** 2).tolist())
        out['runs'][sk] = res
        print(f"  -> dE/r^2={res['dE_over_r2']:.6f} g_eff={np.round(res['g_eff'],3)} steps={res['steps']} mono={mono} wall={res['wall']:.0f}s", flush=True)
        np.savez_compressed(os.path.join(a.outdir, f'{name}_{sk}.npz'), S=S.astype(np.float32), radial=rad)
        os.makedirs(os.path.join(a.outdir, 'fields'), exist_ok=True)
        np.save(os.path.join(a.outdir, 'fields', f'{name}_{sk}_m.npy'), m.astype(np.float32))   # large; git-ignored
    json.dump(out, open(os.path.join(a.outdir, name + '.json'), 'w'), indent=1, default=float)
    best = min(out['runs'], key=lambda k: out['runs'][k]['dE_over_r2'])
    print(' WINNER (lowest E):', best, {k: round(v['dE_over_r2'], 6) for k, v in out['runs'].items()})


if __name__ == '__main__':
    main()
