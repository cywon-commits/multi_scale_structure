"""REQUEST3: chirality of the two-scale 6-fold quasicrystal.
  python chiral.py Hcal P1|P2 L            # calibrate (H1,H2) once, store runs/chiral_H_<pt>_L<L>.json
  python chiral.py A P1|P2 L i0 i1         # noise seeds i0..i1-1 at the stored H (task A)
  python chiral.py mirror P1|P2 L          # y -> -y mirror of the stored reference state, relaxed (control)
  python chiral.py B ...                   # domain wall (task B)
  python chiral.py C ...                   # eta(c) scan (task C)
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'landau'))
from params import landau_to_model
import gl_gpu as G, analysis as an, run_point as rp, canonical as CN
import chirality as CH

ROOT = rp.ROOT; RUNS = os.path.join(ROOT, 'runs')
PT = {'P1': dict(c=0.3, rho=2.0, K=150, g=4.0), 'P2': dict(c=0.6, rho=2.318, K=1000, g=4.0)}
TAU, KAPPA = 2.0, 12.0


def model(pt):
    d = PT[pt]; return landau_to_model(d['c'], d['rho'], d['K'], d['g'])


def Hpath(pt, L): return os.path.join(RUNS, f'chiral_H_{pt}_L{L}.json')


def measure_state(m, p, box=None):
    """peaks -> chirality.measure.  For a rectangular box the integer vectors are rescaled to a square-box metric."""
    pk = an.find_peaks(m, p, nmax=60, rel=1e-4)
    Lx, Ly = m.shape[1], m.shape[2]
    pl = [{'n': [d['n'][0] * Ly / Lx * 0 + d['n'][0], d['n'][1] * Lx / Ly], 'a': d['a_landau']} for d in pk] if Lx != Ly else [{'n': list(d['n']), 'a': d['a_landau']} for d in pk]
    try:
        r = CH.measure(pl)
    except Exception as e:
        r = dict(error=str(e))
    return r, pk


def summarize(m, p, extra=None):
    r, pk = measure_state(m, p)
    dF, M = CN.dF(m, p)
    out = dict(dF=float(dF), g_eff=[G.g_eff(p, v) for v in M], meas=r, top=[(list(d['n']), round(d['a_landau'], 3)) for d in pk[:14]])
    q1, q2 = r.get('Q1'), r.get('Q2')
    out['eta1'] = q1['eta'] if q1 else None; out['eta2'] = q2['eta'] if q2 else None
    out['offset'] = r.get('offset_deg'); out['chi'] = r.get('chi')
    out['is_qc'] = bool(q1 and q2 and q1['eta'] > 0.2 and q2['eta'] > 0.2 and r.get('chi', 0) != 0)
    if extra: out.update(extra)
    return out


def getH(pt, L):
    f = Hpath(pt, L)
    if os.path.exists(f): return json.load(open(f))
    p = model(pt)
    sim = rp.make_sim(p, L, TAU, 'qc', kappa=KAPPA)
    hist = rp.calibrate(sim, p, 10, 1500, 250, verbose=True)
    H = dict(H1=sim.p['H1'], H2=sim.p['H2'], hist=hist)
    json.dump(H, open(f, 'w'), indent=1, default=float); return H


def relax_from(p, L, m0, H, nmax=12000, tol=1e-7):
    sim = G.GLBilayer(p, L=L, dt=TAU / p['r_abs'], stab=KAPPA * p['r_abs'])
    sim.set_fields(H['H1'], H['H2']); sim.m = G.xp.asarray(m0)
    E, mono = rp.relax_checked(sim, nmax, 250, tol, p, verbose=False)
    return sim, E, mono


def taskA(pt, L, i0, i1, nmax=12000):
    H = getH(pt, L); out = []
    fn = os.path.join(RUNS, f'chiral_A_{pt}_L{L}_{i0}_{i1}.json')
    for i in range(i0, i1):
        t0 = time.time(); p = model(pt)
        sim = rp.make_sim(p, L, TAU, f'noise{i}', kappa=KAPPA)
        sim.set_fields(H['H1'], H['H2'])
        E, mono = rp.relax_checked(sim, nmax, 250, 1e-7, p, verbose=False)
        m = sim.state(); np.save(os.path.join(RUNS, 'fields', f'chiral_{pt}_L{L}_s{i}_m.npy'), m.astype(np.float32))
        r = summarize(m, p, dict(seed=i, steps=int((len(E) - 1) * 250), monotone=mono, wall=time.time() - t0))
        out.append(r); json.dump(out, open(fn, 'w'), indent=1, default=float)
        print(f"{pt} L{L} seed {i}: dF={r['dF']:.4f} qc={r['is_qc']} chi={r['chi']} off={r['offset']} eta1={r['eta1']} eta2={r['eta2']} steps={r['steps']} {r['wall']:.0f}s", flush=True)


def mirror_y(m):
    """y -> -y (index j -> (-j) mod Ly) with the in-plane y components flipped"""
    mm = np.roll(m[:, :, ::-1], 1, axis=2).copy()
    mm[1] *= -1; mm[4] *= -1
    return mm


def taskMirror(pt, L, ref):
    H = getH(pt, L); p = model(pt)
    m = np.load(ref).astype(float)
    r0 = summarize(m, p, dict(tag='reference'))
    sim, E, mono = relax_from(p, L, m, H)
    r1 = summarize(sim.state(), p, dict(tag='reference relaxed', steps=len(E) * 250))
    mm = mirror_y(m)
    sim2, E2, mono2 = relax_from(p, L, mm, H)
    r2 = summarize(sim2.state(), p, dict(tag='mirror relaxed', steps=len(E2) * 250))
    r3 = summarize(mm, p, dict(tag='mirror raw'))
    res = [r0, r1, r3, r2]
    for r in res: print(r['tag'], 'dF=%.5f' % r['dF'], 'chi', r['chi'], 'off', r['offset'], 'eta', r['eta1'], r['eta2'])
    json.dump(res, open(os.path.join(RUNS, f'chiral_mirror_{pt}_L{L}.json'), 'w'), indent=1, default=float)



# ------------------------------------------------------------------ task B: mirror-domain wall
def _peaks_big(m, p, nbig=6):
    pk = an.find_peaks(m, p, nmax=40, rel=1e-4)
    return pk


def local_chirality(m, big_plus, big_minus, w=160, step=24):
    """chi(x) = (W+ - W-)/(W+ + W-) with W+- = sum of windowed amplitudes at the big-orbit vectors of the +/- domain.
    Windows: Hann in x of width w, full extent in y. Returns x centres, chi, W+, W-."""
    Lx, Ly = m.shape[1], m.shape[2]
    mz = m[2] + m[5]
    xs, chis, Wp, Wm = [], [], [], []
    xx = np.arange(Lx)
    for x0 in range(0, Lx, step):
        idx = (np.arange(x0 - w // 2, x0 + w // 2)) % Lx
        win = np.hanning(w)
        sub = mz[idx, :] * win[:, None]
        F = np.fft.fft2(sub, s=(w * 4, Ly))             # zero-pad in x for finer sampling
        norm = win.sum() * Ly
        def amp(n):
            kx = n[0] * (w * 4) / Lx                    # wavevector in the padded grid (units of 2pi/(4w))
            ix = int(round(kx)) % (w * 4); iy = n[1] % Ly
            blk = np.abs(F[(np.arange(ix - 3, ix + 4)) % (w * 4), iy]); return 2 * blk.max() / norm
        wp = sum(amp(n) ** 2 for n in big_plus); wm = sum(amp(n) ** 2 for n in big_minus)
        xs.append(x0); Wp.append(wp); Wm.append(wm); chis.append((wp - wm) / (wp + wm + 1e-30))
    return np.array(xs), np.array(chis), np.array(Wp), np.array(Wm)


def taskB(pt, L, ref, nmax=12000, tag=''):
    H = getH(pt, L); p = model(pt)
    S = np.load(ref).astype(float)
    r_ref = summarize(S, p)
    sim0, E0, _ = relax_from(p, L, S, H)                 # relaxed uniform reference (should be stationary)
    S = sim0.state(); r_ref = summarize(S, p, dict(tag='uniform ref'))
    M = mirror_y(S)
    big = lambda m: [tuple(d['n']) for d in an.find_peaks(m, p, nmax=40, rel=1e-4)[:12]]   # 12 strongest = big Q1+Q2 orbits (6+6)
    bp, bm = big(S), big(M)
    m0 = np.concatenate([S, M], axis=1)                  # (6, 2L, L): wall at x=L and x=2L=0
    Lx, Ly = m0.shape[1], m0.shape[2]
    snaps = []
    sim = G.GLBilayer(p, L=(Lx, Ly), dt=TAU / p['r_abs'], stab=KAPPA * p['r_abs'])
    sim.set_fields(H['H1'], H['H2']); sim.m = G.xp.asarray(m0)
    Eu = sim.uniform_energy()[0]
    hist = []
    for blk in range(nmax // 250):
        sim.step(250)
        if blk % 4 == 0:
            m = sim.state(); dF, _ = CN.dF(m, p)
            sigma = (dF - r_ref['dF']) * Lx / 2
            xs, chi, Wp, Wm = local_chirality(m, bp, bm)
            hist.append(dict(step=(blk + 1) * 250, dF=float(dF), sigma=float(sigma), chi=chi.tolist()))
            print(f'step {(blk+1)*250}: dF={dF:.5f} sigma={sigma:.4f} |r|^2 a  chi(x) min/max {chi.min():.2f}/{chi.max():.2f}', flush=True)
            if blk >= 12 and abs(hist[-1]['dF'] - hist[-2]['dF']) < 1e-6: break
    m = sim.state(); np.save(os.path.join(RUNS, 'fields', f'chiral_wall_{pt}_L{L}{tag}_m.npy'), m.astype(np.float32))
    xs, chi, Wp, Wm = local_chirality(m, bp, bm)
    dF, _ = CN.dF(m, p)
    res = dict(point=pt, L=[Lx, Ly], dF_uniform=r_ref['dF'], dF_wall=float(dF), sigma=float((dF - r_ref['dF']) * Lx / 2),
               sigma_per_Q1=float((dF - r_ref['dF']) * Lx / 2 * p['q_minus'] / (2 * np.pi)), x=xs.tolist(), chi=chi.tolist(), hist=hist,
               big_plus=bp, big_minus=bm, ref=r_ref)
    json.dump(res, open(os.path.join(RUNS, f'chiral_B_{pt}{tag}.json'), 'w'), indent=1, default=float)
    print('sigma', res['sigma'], 'per Q1 wavelength', res['sigma_per_Q1'])


# ------------------------------------------------------------------ task C: eta(c) scan
R12 = 2 * np.cos(np.pi / 12)


def rho_qc(c):
    s = np.sqrt(1 - c * c); x = (R12 - 1) / (R12 + 1)
    return (s + x) / (s - x)


def taskC(cs, L, nmax=12000, K=1000):
    for c in cs:
        t0 = time.time()
        rho = rho_qc(c); p = landau_to_model(c, rho, K, 4.0)
        sfx = '' if K == 1000 else f'_K{K}'
        pr = CH.predict(4.0, [c])[0]
        A, B, C_, D = pr['A'], pr['B'], pr['C'], pr['D']
        a1, a2 = 0.5 * (A + B), 0.5 * (C_ + D); eps = 0.01
        seeds = {'a_landau': (A, B, C_, D), 'b_sym_pert': (a1 * (1 + eps), a1 * (1 - eps), a2 * (1 + eps), a2 * (1 - eps))}
        modes0, info = an.qc_int_modes(p, L, amps=seeds['a_landau'])
        print(f'== c={c} rho={rho:.4f} Landau A,B,C,D={A:.3f},{B:.3f},{C_:.3f},{D:.3f} eta1,eta2={pr["eta1"]:.3f},{pr["eta2"]:.3f}  box {L} approximant {info}', flush=True)
        # H calibration with the Landau seed
        res = []
        H = None
        for sk, amps in seeds.items():
            modes, _ = an.qc_int_modes(p, L, amps=amps)
            m0 = an.int_seed(modes, L, p, c) + 1e-4 * np.sqrt(p['r_abs']) * np.random.default_rng(0).normal(size=(6, L, L))
            m0[2] += p['M_target']; m0[5] += p['M_target']
            sim = G.GLBilayer(p, L=L, dt=TAU / p['r_abs'], stab=KAPPA * p['r_abs']); sim.m = G.xp.asarray(m0)
            if H is None:
                rp.calibrate(sim, p, 4, 1500, 250, verbose=False); H = (sim.p['H1'], sim.p['H2'])
            else:
                sim.set_fields(*H)
            E, mono = rp.relax_checked(sim, nmax, 250, 1e-7, p, verbose=False)
            m = sim.state(); np.save(os.path.join(RUNS, 'fields', f'chiral_C{sfx}_c{c}_{sk}_m.npy'), m.astype(np.float32))
            r = summarize(m, p, dict(seed=sk, c=c, rho=rho, landau=dict(A=A, B=B, C=C_, D=D, eta1=pr['eta1'], eta2=pr['eta2'], F=pr['F']),
                                    steps=int((len(E) - 1) * 250), monotone=mono, approximant=info, H=H))
            res.append(r)
            print(f"  {sk}: dF={r['dF']:.4f} (Landau K=inf {pr['F']:.4f}) eta1={r['eta1']} eta2={r['eta2']} chi={r['chi']} off={r['offset']} g_eff={np.round(r['g_eff'],3)} steps={r['steps']}", flush=True)
        json.dump(res, open(os.path.join(RUNS, f'chiral_C{sfx}_c{c}.json'), 'w'), indent=1, default=float)
        print(f'  wall {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    t = sys.argv[1]
    if t == 'Hcal': getH(sys.argv[2], int(sys.argv[3]))
    elif t == 'A': taskA(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif t == 'mirror': taskMirror(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif t == 'B': taskB(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif t == 'C': taskC([float(x) for x in sys.argv[3].split(',')], int(sys.argv[2]), K=float(sys.argv[4]) if len(sys.argv) > 4 else 1000)
