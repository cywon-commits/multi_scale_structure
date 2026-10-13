"""figures: hysteresis curves, S(q) maps of winners, box-size dependence."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), '..'); RUNS = os.path.join(ROOT, 'runs'); FIG = os.path.join(ROOT, 'figures')

def hyst():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for f, mk in sorted(zip(glob.glob(os.path.join(RUNS, 'hysteresis_*.json')), ['o', 's'])):
        h = json.load(open(f)); tag = os.path.basename(f)[11:-5]
        for d, ls in (('up', '-'), ('down', '--')):
            r = [x for x in h if x['dir'] == d]
            ax[0].plot([x['rho'] for x in r], [x['dE'] for x in r], ls, marker=mk, label=f'{tag} {d}')
            ax[1].plot([x['rho'] for x in r], [x['metrics']['Q2_2Q1']['n'] for x in r], ls, marker=mk, label=f'{tag} {d}')
    ax[0].set_xlabel('ρ = D₂/D₁'); ax[0].set_ylabel('ΔE/|r|²'); ax[1].set_xlabel('ρ'); ax[1].set_ylabel('# peaks in Q₂/2Q₁ shell (3: hex, 6: QC)')
    for a in ax: a.axvline(2.318, c='gray', lw=.5); a.axvline(2.429, c='gray', lw=.5)
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'hysteresis.png'), dpi=130)

def smaps():
    for f in sorted(glob.glob(os.path.join(RUNS, 'P[1-4]_*L512.json'))):
        d = json.load(open(f)); best = min(d['runs'], key=lambda k: d['runs'][k]['dE_over_r2'])
        S = np.load(f.replace('.json', f'_{best}.npz'))['S']; L = d['L']; sc = L / 2 / np.pi
        nmax = int(2.6 * d['params']['q_plus'] * sc)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.3))
        for i in range(2):
            s = np.fft.fftshift(S[i]); c = L // 2
            ax[i].imshow(np.log10(s[c-nmax:c+nmax, c-nmax:c+nmax].T + 1e-30), origin='lower', vmin=np.log10(s.max()) - 6, extent=[-nmax, nmax, -nmax, nmax], cmap='magma')
            ax[i].set_title(f'layer {i+1} log10 S(q), seed {best}'); ax[i].set_xlabel('n_x')
        fig.suptitle(os.path.basename(f)[:-5], fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'S_' + os.path.basename(f)[:-5] + '.png'), dpi=110); plt.close(fig)

def boxsize():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, P in zip(ax, ('P1', 'P2')):
        for L in (384, 512, 768):
            fs = glob.glob(os.path.join(RUNS, f'{P}_*_L{L}.json'))
            if not fs: continue
            d = json.load(open(fs[0]))
            for k, r in d['runs'].items():
                if k.startswith('noise') or k == 'qc': a.plot(L, r['dE_over_r2'], 'o' if k.startswith('noise') else '^', c='C0' if k.startswith('noise') else 'C3')
        a.set_title(P + ' (o noise seeds, ▲ qc seed)'); a.set_xlabel('L'); a.set_ylabel('ΔE/|r|²')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'boxsize.png'), dpi=130)

if __name__ == '__main__':
    hyst(); smaps(); boxsize()
