"""Canonical (fixed-<m_z>) modulation energy, the quantity that corresponds to the Landau F at fixed M:
    dF = [E_state - (-H.<m_z> terms)] - E_uniform(<m_z>_1, <m_z>_2)    (no field terms in either)
The earlier 'dE' (grand-canonical, E - E_uniform(H) at the same H) exceeds dF by ~ (1/2) chi dH^2 > 0, because the
modulated state needs a slightly different H than the uniform one to reach the same <m_z>.
usage: python canonical.py   (recomputes dF for all g != 0 runs that have saved fields) -> runs/canonical.json"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from params import landau_to_model
import gl_gpu as G
ROOT = os.path.join(os.path.dirname(__file__), '..'); RUNS = os.path.join(ROOT, 'runs')


def dF(m, p):
    """m: (6,Lx,Ly) field; p: model params (r1,r2,c,u,...). returns dF/|r|^2 and the layer means"""
    q = dict(p); q['H1'] = q['H2'] = 0.0
    sim = G.GLBilayer(q, L=(m.shape[1], m.shape[2]), dt=1.0)
    sim.m = G.xp.asarray(m.astype(float))
    E0 = sim.energy()
    M1, M2 = sim.mz_mean()
    Eu = q['r1'] * M1 ** 2 + q['r2'] * M2 ** 2 + q['u'] * (M1 ** 4 + M2 ** 4) + q['c'] * (M1 - M2) ** 2
    return (E0 - Eu) / q['r_abs'] ** 2, (M1, M2)


if __name__ == '__main__':
    out = {}
    for f in sorted(glob.glob(os.path.join(RUNS, 'P[1-4]_*.json')) + glob.glob(os.path.join(RUNS, 'P6_*.json'))):
        d = json.load(open(f))
        if 'runs' not in d or d['cfg']['g'] == 0: continue
        cfg = d['cfg']; base = os.path.basename(f)[:-5]
        p = landau_to_model(cfg['c_rel'], cfg['rho'], cfg['K'], cfg['g'], q1=2 * np.pi / cfg.get('lambda1', 32))
        for sk, r in d['runs'].items():
            fm = os.path.join(RUNS, 'fields', f'{base}_{sk}_m.npy')
            if not os.path.exists(fm): continue
            m = np.load(fm)
            v, M = dF(m, p)
            out[f'{base}:{sk}'] = dict(dE=r['dE_over_r2'], dF=v, M=list(M), g_eff=[G.g_eff(p, x) for x in M])
            print(f'{base}:{sk}  dE={r["dE_over_r2"]:.4f}  dF={v:.4f}', flush=True)
    json.dump(out, open(os.path.join(RUNS, 'canonical.json'), 'w'), indent=1)
