import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'landau'))
ROOT = os.path.join(os.path.dirname(__file__), '..'); RUNS = os.path.join(ROOT, 'runs'); FIG = os.path.join(ROOT, 'figures')

def figA():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (pt, L) in zip(ax, (('P1', 512), ('P2', 768))):
        rows = []
        for f in sorted(glob.glob(os.path.join(RUNS, f'chiral_A_{pt}_L{L}_*.json'))): rows += json.load(open(f))
        for r in rows:
            col = {1: 'C3', -1: 'C0', 0: '0.6'}[r['chi'] if r['chi'] is not None else 0]
            a.scatter(r['seed'], r['dF'], c=col, marker='o' if r['is_qc'] else 'x')
        a.set_title(f'{pt} L={L}: red χ=+1, blue χ=−1, grey other (x = not strict QC)', fontsize=8); a.set_xlabel('seed'); a.set_ylabel('ΔF/|r|²')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'chiral_A.png'), dpi=130)

def figB():
    d = json.load(open(os.path.join(RUNS, 'chiral_B_analysis.json')))
    B = json.load(open(os.path.join(RUNS, 'chiral_B_P2.json')))
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
    ax[0, 0].plot(B['x'], B['chi'], 'o-', ms=3, label='window 160'); ax[0, 0].plot(d['x64'], d['chi64'], '.-', ms=3, label='window 64')
    ax[0, 0].set_xlabel('x'); ax[0, 0].set_ylabel('χ(x)'); ax[0, 0].legend(fontsize=7)
    prof = np.array(d['prof']); ex = prof - d['bulk']; k = np.ones(56) / 56; exs = np.convolve(np.r_[ex[-28:], ex, ex[:28]], k, mode='same')[28:-28]; ax[0, 1].plot(ex, c='0.8', lw=.5); ax[0, 1].plot(exs, c='C0'); ax[0, 1].set_xlabel('x'); ax[0, 1].set_ylabel('(e(x) − bulk)/|r|²  (per site; blue = 2 Q₁-period average)')
    cm = np.array(d['chi2']); xs = d['map_x']; ys = d['map_y']
    im = ax[1, 0].pcolormesh(xs, ys, cm.T, vmin=-np.abs(cm).max(), vmax=np.abs(cm).max(), cmap='coolwarm', shading='auto'); ax[1, 0].set_aspect('equal'); ax[1, 0].set_xlabel('x'); ax[1, 0].set_ylabel('y'); fig.colorbar(im, ax=ax[1, 0], shrink=.7)
    ax[1, 0].set_title('local χ(x,y), 128² windows', fontsize=8)
    ax[1, 1].axis('off'); ax[1, 1].text(0, 0.8, f"σ = {B['sigma']:.1f} |r|²·a\n  = {B['sigma_per_Q1']:.2f} |r|² λ₁\nbox {B['L'][0]}×{B['L'][1]}", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'chiral_B_wall.png'), dpi=130)

def figC():
    import chirality as CH
    cs = np.arange(0.60, 0.8601, 0.01); pr = CH.predict(4.0, cs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, sfx, K in zip(ax, ('', '_K150.0'), (1000, 150)):
        rows = []
        for f in sorted(glob.glob(os.path.join(RUNS, f'chiral_C{sfx}_c*.json'))): rows += json.load(open(f))
        a.plot([r['c'] for r in pr], [r['eta1'] for r in pr], 'k-', label='η₁ Landau (K=∞)'); a.plot([r['c'] for r in pr], [r['eta2'] for r in pr], 'k--', label='η₂ Landau (K=∞)')
        for sk, mk in (('a_landau', 'o'), ('b_sym_pert', 's')):
            rr = [r for r in rows if r['seed'] == sk and r['eta1'] is not None]
            a.plot([r['c'] for r in rr], [r['eta1'] for r in rr], mk, c='C0', mfc='none', label=f'η₁ sim ({sk})'); a.plot([r['c'] for r in rr], [r['eta2'] for r in rr], mk, c='C3', mfc='none', label=f'η₂ sim ({sk})')
        a.set_title(f'K={K}', fontsize=9); a.set_xlabel('c/c_L'); a.set_ylabel('η'); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'chiral_C.png'), dpi=130)

if __name__ == '__main__':
    figA(); figB(); figC()
