"""Domain-wall analysis (task B): column-resolved energy density, local chirality chi(x) with windows of two sizes, 2D chirality map."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import gl_gpu as G, chiral as CHI
from params import landau_to_model
ROOT = os.path.join(os.path.dirname(__file__), '..'); RUNS = os.path.join(ROOT, 'runs')

def energy_density(m, p):
    """local energy density per site (real part), same functional as GLBilayer.energy: quadratic via L_q, quartic, field"""
    Lx, Ly = m.shape[1:]
    sim = G.GLBilayer(p, L=(Lx, Ly), dt=1.0); sim.m = G.xp.asarray(m)
    mh = np.fft.rfft2(m, axes=(1, 2)); mh = np.moveaxis(mh, 0, -1)
    Lm = np.einsum('abij,abj->abi', G.to_np(sim.Lq), mh)
    Lm = np.moveaxis(Lm, -1, 0)
    Lmr = np.fft.irfft2(Lm, s=(Lx, Ly), axes=(1, 2))
    e = (m * Lmr).sum(0)
    for o, H in ((0, p['H1']), (3, p['H2'])):
        m2 = (m[o:o + 3] ** 2).sum(0); e += p['u'] * m2 ** 2 - H * m[o + 2]
    return e

def local_map(m, bp, bm, w=128, step=48):
    """2D chi(x,y): windowed (Hann w x w) amplitude at the big-orbit wavevectors of the + and - domains"""
    Lx, Ly = m.shape[1:]; mz = m[2] + m[5]
    xs = np.arange(0, Lx, step); ys = np.arange(0, Ly, step)
    win1 = np.hanning(w); win = win1[:, None] * win1[None, :]
    def ks(vs): return [(2 * np.pi * n[0] / Lx, 2 * np.pi * n[1] / Ly) for n in vs]
    kp, km = ks(bp), ks(bm)
    X = np.arange(w)[:, None]; Y = np.arange(w)[None, :]
    chi = np.zeros((len(xs), len(ys)))
    for ia, x0 in enumerate(xs):
        ix = (np.arange(x0 - w // 2, x0 + w // 2)) % Lx
        for ib, y0 in enumerate(ys):
            iy = (np.arange(y0 - w // 2, y0 + w // 2)) % Ly
            sub = mz[np.ix_(ix, iy)] * win
            def amp(k):
                best = 0
                for dk in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:     # small neighbourhood (window resolution 2pi/w)
                    kk = (k[0] + dk[0] * np.pi / w, k[1] + dk[1] * np.pi / w)
                    ph = np.exp(-1j * (kk[0] * X + kk[1] * Y))
                    best = max(best, abs((sub * ph).sum()))
                return 2 * best / win.sum()
            wp = sum(amp(k) ** 2 for k in kp); wm = sum(amp(k) ** 2 for k in km)
            chi[ia, ib] = (wp - wm) / (wp + wm + 1e-30)
    return xs, ys, chi

if __name__ == '__main__':
    d = json.load(open(os.path.join(RUNS, 'chiral_B_P2.json')))
    p = CHI.model('P2'); H = json.load(open(CHI.Hpath('P2', 768))); p['H1'], p['H2'] = H['H1'], H['H2']
    m = np.load(os.path.join(RUNS, 'fields', 'chiral_wall_P2_L768_m.npy')).astype(float)
    bp = [tuple(n) for n in d['big_plus']]; bm = [tuple(n) for n in d['big_minus']]
    e = energy_density(m, p); prof = e.mean(1); ref = np.mean(prof[100:600] if False else prof)
    # reference bulk energy density: far from both walls (x in [300,500] and [1068,1268] are bulk-like)
    bulk = 0.5 * (prof[300:500].mean() + prof[1068:1268].mean())
    excess = prof - bulk
    out = dict(bulk=float(bulk / p['r_abs'] ** 2), prof=(prof / p['r_abs'] ** 2).tolist(),
               excess_integral=float(excess.sum() / 2 / p['r_abs'] ** 2))      # two walls
    xs, ys, chi2 = local_map(m, bp, bm)
    out.update(map_x=xs.tolist(), map_y=ys.tolist(), chi2=chi2.tolist())
    # finer 1D profile with a narrow window (64) full height
    xs1, chi64, _, _ = CHI.local_chirality(m, bp, bm, w=64, step=8)
    out.update(x64=xs1.tolist(), chi64=chi64.tolist())
    json.dump(out, open(os.path.join(RUNS, 'chiral_B_analysis.json'), 'w'))
    # wall centres and widths from the energy excess (FWHM)
    for name, c0 in (('wall1', 768), ('wall2', 0)):
        idx = (np.arange(c0 - 200, c0 + 200)) % 1536
        ex = excess[idx]; xc = (ex * np.arange(-200, 200)).sum() / ex.sum(); pk = ex.max()
        half = np.where(ex > pk / 2)[0]
        print(name, 'energy-excess centroid offset', round(xc, 1), 'FWHM', half.max() - half.min() + 1, 'peak', pk / p['r_abs'] ** 2)
    print('excess integral per wall (|r|^2 a):', out['excess_integral'], ' (sigma from dF:', d['sigma'], ')')
