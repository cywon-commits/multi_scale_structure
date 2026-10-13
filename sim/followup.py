"""Follow-up request (REQUEST2): minimal resonant states (V, T+, T-) and sqrt(3)-box hexagonal phases.
All runs use rectangular boxes (Lx, Ly) chosen so that the seed wavevectors are integer vectors of the box.
usage: python followup.py A [--cs 0.5,0.55,...] | B | C | D
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from params import landau_to_model, kernel
import gl_gpu as G
import analysis as an
import run_point as rp

ROOT = rp.ROOT
RUNS = os.path.join(ROOT, 'runs')
TAU, KAPPA = 2.0, 12.0


# ------------------------------------------------------------------ geometry helpers
def ratio_R(p):
    return p['q_plus'] / p['q_minus']


def _near_int_err(x):
    return abs(x - round(x)) / x


def box_V(p, lo=300, hi=560):
    """k- = (n-,0), k+ = (n-, +-m) with |k+| = R q-  (cos(theta) = 1/R).  Lx = 2 pi n-/q-, Ly = 2 pi m / (q- sqrt(R^2-1))."""
    R = ratio_R(p); q = p['q_minus']; best = None
    for n in range(10, 40):
        Lx = 2 * np.pi * n / q
        if not lo <= Lx <= hi: continue
        for m in range(1, 80):
            Ly = 2 * np.pi * m / (q * np.sqrt(R * R - 1))
            if not lo <= Ly <= hi: continue
            e = max(_near_int_err(Lx), _near_int_err(Ly))
            if best is None or e < best[0]:
                best = (e, n, m, int(round(Lx)), int(round(Ly)))
    return best


def box_perp(p, lo=300, hi=560):
    """k- = (n-,0), k+ = (0, n+): Lx = 2 pi n-/q-, Ly = 2 pi n+/q+."""
    best = None
    for n in range(10, 40):
        Lx = 2 * np.pi * n / p['q_minus']
        if not lo <= Lx <= hi: continue
        for m in range(10, 60):
            Ly = 2 * np.pi * m / p['q_plus']
            if not lo <= Ly <= hi: continue
            e = max(_near_int_err(Lx), _near_int_err(Ly))
            if best is None or e < best[0]:
                best = (e, n, m, int(round(Lx)), int(round(Ly)))
    return best


def box_T(p, kind='+', lo=300, hi=600):
    """T+: k- = (n,0), k+ = (-n/2, +-m) (n even), |k+| = R q-.  T-: k+ = (n,0)... handled with swapped roles."""
    R = ratio_R(p); q = p['q_minus']; best = None
    for n in range(10, 50, 2):
        Lx = 2 * np.pi * n / q
        if not lo <= Lx <= hi: continue
        ky = R * q * np.sqrt(1 - 1 / (4 * R * R))
        for m in range(1, 90):
            Ly = 2 * np.pi * m / ky
            if not lo <= Ly <= hi: continue
            e = max(_near_int_err(Lx), _near_int_err(Ly))
            if best is None or e < best[0]:
                best = (e, n, m, int(round(Lx)), int(round(Ly)))
    return best


def V_amps(c):
    """Landau mode amplitudes (k-, k+, k+') (Landau units) interpolated/extrapolated from the two values in REQUEST2"""
    a6 = np.array([0.696, 0.630, -0.190]); a8 = np.array([0.745, 0.432, -0.432])
    return a6 + (a8 - a6) * (c - 0.6) / 0.2


# ------------------------------------------------------------------ generic run
def run_state(p, box, specs, g=0.0, nmax=8000, tol=1e-7, tag='', noise=1e-4, seed=0, calib=False, verbose=False):
    t0 = time.time()
    Lx, Ly = box
    m = an.eig_spiral_seed(p, (Lx, Ly), specs) if specs else np.zeros((6, Lx, Ly))
    m = m + noise * np.sqrt(p['r_abs']) * np.random.default_rng(seed).normal(size=(6, Lx, Ly))
    m[2] += p['M_target']; m[5] += p['M_target']
    sim = G.GLBilayer(p, L=(Lx, Ly), dt=TAU / p['r_abs'], stab=KAPPA * p['r_abs'])
    sim.m = G.xp.asarray(m)
    hist = None
    if calib and g != 0:
        hist = rp.calibrate(sim, p, 8, 3000, 250, verbose=verbose)
        p['H1'], p['H2'] = sim.p['H1'], sim.p['H2']
    E, mono = rp.relax_checked(sim, nmax, 250, tol, p, verbose=verbose)
    Eu = sim.uniform_energy()[0]
    mf = sim.state()
    peaks = an.find_peaks(mf, p, nmax=40, rel=1e-4)
    import canonical as _C
    dFv = _C.dF(mf, p)[0] if g != 0 else None
    res = dict(tag=tag, box=[Lx, Ly], dE=(sim.energy() - Eu) / p['r_abs'] ** 2, dF=dFv, monotone=mono, steps=int((len(E) - 1) * 250),
               g_eff=[G.g_eff(p, v) for v in sim.mz_mean()], wall=time.time() - t0,
               peaks=[dict(n=list(d['n']), k_over_qm=d['k'] / p['q_minus'], angle=d['angle'], a1=d['a1'] / np.sqrt(p['r_abs']),
                           a2=d['a2'] / np.sqrt(p['r_abs']), a=d['a_landau']) for d in peaks[:16]])
    if hist: res['calib'] = hist
    return res, mf


def phase_of(mf, n):
    """phase phi of m_z(total) = a cos(k.r + phi) at integer wavevector n=(nx,ny)"""
    h = np.fft.fft2(mf[2] + mf[5])
    return float(np.angle(h[n[0] % mf.shape[1], n[1] % mf.shape[2]]))


def save(name, obj):
    os.makedirs(RUNS, exist_ok=True)
    json.dump(obj, open(os.path.join(RUNS, name), 'w'), indent=1, default=float)


# ------------------------------------------------------------------ Task A
def taskA(cs, nmax):
    out = []
    for c in cs:
        p0 = landau_to_model(c, 2.0, 50, 0.0)
        R = ratio_R(p0)
        eV, nV, mV, LxV, LyV = box_V(p0)
        eP, nP, mP, LxP, LyP = box_perp(p0)
        info = dict(c=c, R=R, q_minus=p0['q_minus'], q_plus=p0['q_plus'], theta_deg=float(np.degrees(np.arccos(1 / R))),
                    boxV=dict(n_minus=nV, m=mV, L=[LxV, LyV], rel_err=eV, qm_err=abs(2 * np.pi * nV / LxV - p0['q_minus']) / p0['q_minus']),
                    boxPerp=dict(n_minus=nP, n_plus=mP, L=[LxP, LyP], rel_err=eP))
        print(f'== c={c} R={R:.4f} V box {LxV}x{LyV} (n-={nV}, m={mV}, err {eV:.1e})  perp box {LxP}x{LyP} (n-={nP}, n+={mP}, err {eP:.1e})', flush=True)
        a = V_amps(c)
        p = landau_to_model(c, 2.0, 50, 0.0)
        specs = [((nV, 0), a[0]), ((nV, mV), a[1]), ((nV, -mV), a[2])]
        rV, mfV = run_state(p, (LxV, LyV), specs, nmax=nmax, tag='V')
        Phi = (2 * phase_of(mfV, (nV, 0)) - phase_of(mfV, (nV, mV)) - phase_of(mfV, (nV, -mV))) % (2 * np.pi)
        rV['Phi_quartet'] = float(Phi)
        print(f'  V: dE={rV["dE"]:.4f} steps={rV["steps"]} Phi={Phi/np.pi:.3f} pi', flush=True)
        p = landau_to_model(c, 2.0, 50, 0.0)
        rP, _ = run_state(p, (LxP, LyP), [((nP, 0), 1.0), ((0, mP), 1.0)], nmax=nmax, tag='perp')
        print(f'  perp: dE={rP["dE"]:.4f} steps={rP["steps"]}', flush=True)
        p = landau_to_model(c, 2.0, 50, 0.0)
        r1, _ = run_state(p, (LxP, LyP), [((nP, 0), 1.0)], nmax=nmax, tag='sp1')
        print(f'  sp1 : dE={r1["dE"]:.4f} steps={r1["steps"]}', flush=True)
        # V seed with the SAME (sign-flipped) k+ amplitudes but without the sign flip -> phase-wrong control
        p = landau_to_model(c, 2.0, 50, 0.0)
        specsw = [((nV, 0), a[0]), ((nV, mV), a[1]), ((nV, -mV), -a[2])]
        rW, _ = run_state(p, (LxV, LyV), specsw, nmax=nmax, tag='V_wrongsign')
        print(f'  V(wrong sign control): dE={rW["dE"]:.4f}', flush=True)
        out.append(dict(info=info, V=rV, perp=rP, sp1=r1, V_wrongsign=rW))
        save(f'A_followup_{cs[0]}.json', out)
    return out


# ------------------------------------------------------------------ Task B
def taskB(nmax):
    out = []
    # (1) analyse the report's L=512 noise winner at g=0.8
    f = os.path.join(RUNS, 'fields', 'P6_c0.95_rho2.0_K150_g0.8_L512_noise_m.npy')
    d = json.load(open(os.path.join(RUNS, 'P6_c0.95_rho2.0_K150_g0.8_L512.json')))
    p = {k: v for k, v in d['params'].items()}; p['landau'] = d['cfg']
    mf = np.load(f).astype(float)
    pk = an.find_peaks(mf, p, nmax=12, rel=1e-3)
    top = pk[:3]
    ks = [np.array(t['n']) for t in top]
    combos = []
    import itertools
    for s in itertools.product((1, -1), repeat=2):
        v = ks[0] + s[0] * ks[1] + s[1] * ks[2]
        combos.append((s, v.tolist()))
    kmin = [t for t in top if t['k'] < 1.1 * p['q_minus']]
    ang = None
    info = dict(top3=[dict(n=list(t['n']), k=t['k'], k_over_qm=t['k'] / p['q_minus'], angle=t['angle'], a1=t['a1'] / np.sqrt(p['r_abs']), a2=t['a2'] / np.sqrt(p['r_abs'])) for t in top], closure=combos)
    # angle between k+ and k- as lines and as vectors (using the closure signs)
    km = np.array(top[0]['n']) if top[0]['k'] < top[1]['k'] else None
    out.append(dict(kind='winner_analysis', **info))
    print('winner (L=512 noise):', json.dumps(info['top3']), '\nclosure sums', combos, flush=True)
    # (2) reseed with exact T+ / V in a box tuned to the geometry, g = 0.8 and 0.5
    for g in (0.8, 0.5):
        pg = landau_to_model(0.95, 2.0, 150, g)
        eT, nT, mT, LxT, LyT = box_T(pg)
        R = ratio_R(pg)
        print(f'== g={g} T box {LxT}x{LyT} n-={nT} m={mT} err {eT:.1e}  R={R:.4f}', flush=True)
        # amplitudes for T+: use the L=512 winner amplitudes ~ (0.68, 0.50, 0.50) negative (Phi = pi)
        specs = [((nT, 0), -0.68), ((-nT // 2, mT), -0.5), ((-nT // 2, -mT), -0.5)]
        res, mfT = run_state(pg, (LxT, LyT), specs, g=g, nmax=nmax, tag=f'T+ g={g}', calib=True)
        res['closure'] = [(nT) + 2 * (-nT // 2)]
        print(f'  T+: dE={res["dE"]:.4f} g_eff={np.round(res["g_eff"],3)} steps={res["steps"]}', flush=True)
        out.append(dict(kind='T+', g=g, box_info=[eT, nT, mT, LxT, LyT], res=res))
        # V seed in its own box
        pg = landau_to_model(0.95, 2.0, 150, g)
        eV, nV, mV, LxV, LyV = box_V(pg)
        a = np.array([0.745, 0.43, -0.43])
        specs = [((nV, 0), a[0]), ((nV, mV), a[1]), ((nV, -mV), a[2])]
        resV, _ = run_state(pg, (LxV, LyV), specs, g=g, nmax=nmax, tag=f'V g={g}', calib=True)
        print(f'  V : dE={resV["dE"]:.4f} (box {LxV}x{LyV}, err {eV:.1e})', flush=True)
        out.append(dict(kind='V', g=g, box_info=[eV, nV, mV, LxV, LyV], res=resV))
        save('B_followup.json', out)
    return out


def taskBg(gs, nmax):
    """V vs T+ energies at c=0.95, rho=2, K=150 for several g (V -> T+ transition predicted at g ~ 0.56)"""
    out = []
    for g in gs:
        row = dict(g=g)
        pg = landau_to_model(0.95, 2.0, 150, g); eT, nT, mT, LxT, LyT = box_T(pg)
        res, _ = run_state(pg, (LxT, LyT), [((nT, 0), -0.68), ((-nT // 2, mT), -0.5), ((-nT // 2, -mT), -0.5)], g=g, nmax=nmax, tag=f'T+ g={g}', calib=True)
        row['T+'] = res
        pg = landau_to_model(0.95, 2.0, 150, g); eV, nV, mV, LxV, LyV = box_V(pg)
        a = [0.745, 0.43, -0.43]
        resV, _ = run_state(pg, (LxV, LyV), [((nV, 0), a[0]), ((nV, mV), a[1]), ((nV, -mV), a[2])], g=g, nmax=nmax, tag=f'V g={g}', calib=True)
        row['V'] = resV
        print(f'g={g}: T+ {res["dE"]:.4f}  V {resV["dE"]:.4f}', flush=True)
        out.append(row); save(f'B_gscan_{gs[0]}.json', out)


# ------------------------------------------------------------------ Task C
def hex_modes(n, amps):
    """integer vectors of the (1, sqrt3, 2) hexagonal shells in a Lx x Ly=sqrt3 Lx box (n even): amps=(a1,a3,a2)"""
    h = n // 2
    q1 = [(n, 0), (-h, 3 * h), (-h, -3 * h)]                 # Q1 at 0, 120, 240 deg
    q3 = [(3 * h, 3 * h), (-3 * h, 3 * h), (0, -3 * n)]      # sqrt3 Q1 at 30, 150, 270 deg
    q2 = [(2 * n, 0), (-n, 3 * n), (-n, -3 * n)]             # 2 Q1 at 0, 120, 240 deg
    specs = [(v, amps[0]) for v in q1]
    if amps[1] != 0: specs += [(v, amps[1]) for v in q3]
    if amps[2] != 0: specs += [(v, amps[2]) for v in q2]
    return specs


def taskC(nmax, which, fsel=None):
    out = []
    P = {'P3': dict(c=0.6, rho=2.318, K=150, g=4.0, f=[0.965, 0.975, 0.985], amps=(-0.652, -0.094, -0.612), amps12=(-0.65, 0.0, -0.60)),
         'P4': dict(c=0.3, rho=1.781, K=1000, g=4.0, f=[0.9999], amps=(-0.7154, -0.7030, 0.0), amps12=None)}
    for name in which:
        cfg = dict(P[name])
        if fsel: cfg['f'] = fsel
        for f in cfg['f']:
            p = landau_to_model(cfg['c'], cfg['rho'], cfg['K'], cfg['g'])
            qm = p['q_minus']
            best = None
            for n in ([int(os.environ['HEXN'])] if os.environ.get('HEXN') else range(10, 40, 2)):
                Lx = 2 * np.pi * n / (f * qm)
                for Lxi in (int(np.floor(Lx)), int(np.ceil(Lx))):
                    fi = 2 * np.pi * n / Lxi / qm
                    Ly = int(round(np.sqrt(3) * Lxi))
                    if not (380 <= Lxi <= 560 or os.environ.get('HEXN')): continue
                    err = abs(fi - f)
                    if best is None or err < best[0]: best = (err, n, Lxi, Ly, fi)
            err, n, Lx, Ly, fi = best
            ratio_err = abs(Ly / Lx - np.sqrt(3)) / np.sqrt(3)
            print(f'== {name} f={f}: n={n} Lx={Lx} Ly={Ly} Q1/q-={fi:.4f} Ly/Lx={Ly/Lx:.5f} (err {ratio_err:.1e})', flush=True)
            seeds = {'landau': cfg['amps']}
            if cfg['amps12']: seeds['hex12'] = cfg['amps12']
            for sk, amps in seeds.items():
                p = landau_to_model(cfg['c'], cfg['rho'], cfg['K'], cfg['g'])
                res, mf = run_state(p, (Lx, Ly), hex_modes(n, amps), g=cfg['g'], nmax=nmax, tag=f'{name} {sk}', calib=True)
                res.update(point=name, seed=sk, f_target=f, f_box=fi, n=n, ratio_err=ratio_err)
                print(f'  {sk}: dF={res["dF"]:.4f} dE={res["dE"]:.4f} g_eff={np.round(res["g_eff"],3)} steps={res["steps"]} mono={res["monotone"]}', flush=True)
                np.save(os.path.join(RUNS, 'fields', f'C_{name}_{sk}_f{f}_n{n}_m.npy'), mf.astype(np.float32))
                out.append(res)
                save(f'C_followup_{"_".join(which)}_{cfg["f"][0]}{os.environ.get("HEXN","")}.json', out)
    return out




# ------------------------------------------------------------------ Task D
def taskD():
    import itertools
    f = os.path.join(RUNS, 'fields', 'P1_c0.3_rho2.0_K150_g4_L768_noise_m.npy')
    d = json.load(open(os.path.join(RUNS, 'P1_c0.3_rho2.0_K150_g4_L768.json')))
    p = dict(d['params']); p['landau'] = d['cfg']
    mf = np.load(f).astype(float)
    L = mf.shape[1]
    pk = an.find_peaks(mf, p, nmax=40, rel=1e-4)
    h = np.fft.fft2(mf[2] + mf[5])
    amp = {tuple(x['n']): x['a_landau'] for x in pk}
    keys = list(amp)
    rows = []
    for a, b in itertools.combinations(keys, 2):
        for sa in (1, -1):
            for sb in (1, -1):
                n1 = (sa * a[0], sa * a[1]); n2 = (sb * b[0], sb * b[1]); n3 = (-(n1[0] + n2[0]), -(n1[1] + n2[1]))
                k3 = n3 if (n3[0] > 0 or (n3[0] == 0 and n3[1] > 0)) else (-n3[0], -n3[1])
                if k3 in amp and k3 != a and k3 != b and keys.index(a) < keys.index(b) < keys.index(k3):
                    Phi = (np.angle(h[n1[0] % L, n1[1] % L]) + np.angle(h[n2[0] % L, n2[1] % L]) + np.angle(h[n3[0] % L, n3[1] % L])) % (2 * np.pi)
                    rows.append(dict(n1=n1, n2=n2, n3=n3, amps=[amp[a], amp[b], amp[k3]], Phi_over_pi=float(Phi / np.pi),
                                     weight=float(amp[a] * amp[b] * amp[k3])))
    rows.sort(key=lambda r: -r['weight'])
    save('D_followup.json', dict(n_triads=len(rows), rows=rows))
    for r in rows[:25]:
        print(r['n1'], r['n2'], r['n3'], np.round(r['amps'], 3), f"Phi/pi={r['Phi_over_pi']:.3f}")
    print('n triads', len(rows))



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('--cs', default='0.5,0.55,0.6,0.65,0.7,0.8'); ap.add_argument('--nmax', type=int, default=8000)
    ap.add_argument('--pts', default='P3,P4'); ap.add_argument('--fs', default=''); ap.add_argument('--gs', default='0.5,0.55,0.6,0.65,0.7')
    a = ap.parse_args()
    os.makedirs(os.path.join(RUNS, 'fields'), exist_ok=True)
    if a.task == 'A': taskA([float(x) for x in a.cs.split(',')], a.nmax)
    elif a.task == 'B': taskB(a.nmax)
    elif a.task == 'C': taskC(a.nmax, a.pts.split(','), [float(x) for x in a.fs.split(',')] if a.fs else None)
    elif a.task == 'D': taskD()
    elif a.task == 'Bg': taskBg([float(x) for x in a.gs.split(',')], a.nmax)

