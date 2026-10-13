import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), '..'); RUNS = os.path.join(ROOT, 'runs'); FIG = os.path.join(ROOT, 'figures')

def load_A():
    rows = []
    for f in ['A_part1.json', 'A_followup_0.65.json']:
        rows += json.load(open(os.path.join(RUNS, f)))
    d = {r['info']['c']: r for r in rows}
    d[0.6] = dict(V=dict(dE=-0.4586), perp=dict(dE=-0.4589), info=dict(c=0.6))   # from the log of the interrupted run
    return d

LAND = {0.5: (-0.46724, -0.47062), 0.55: (-0.46239, -0.46489), 0.6: (-0.45833, -0.45878), 0.65: (-0.45611, -0.45232), 0.7: (-0.45774, -0.44554), 0.8: (-0.46389, -0.43120)}

def figA():
    A = load_A(); cs = sorted(A)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(cs, [A[c]['V']['dE'] for c in cs], 'o-', label='V (sim)')
    ax.plot(cs, [A[c]['perp']['dE'] for c in cs], 's-', label='orthogonal pair (sim)')
    ax.plot(cs, [LAND[c][0] for c in cs], 'o--', mfc='none', c='C0', label='V (Landau)')
    ax.plot(cs, [LAND[c][1] for c in cs], 's--', mfc='none', c='C1', label='pair (Landau)')
    ax.axvline(0.606, c='gray', lw=.6); ax.set_xlabel('c / c_L'); ax.set_ylabel('ΔE / |r|²  (g=0, ρ=2, K=50)'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'A_V_vs_pair.png'), dpi=130)

def figBg():
    rows = []
    for f in ['B_gscan_part1.json', 'B_gscan_0.6.json']:
        p = os.path.join(RUNS, f)
        if os.path.exists(p): rows += json.load(open(p))
    gs = [r['g'] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(gs, [r['T+']['dE'] for r in rows], 'o-', label='T+ (sim)'); ax.plot(gs, [r['V']['dE'] for r in rows], 's-', label='V (sim)')
    ax.set_xlabel('g'); ax.set_ylabel('ΔE/|r|²  (c=0.95, ρ=2, K=150)'); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'B_V_vs_Tplus.png'), dpi=130)

def figC():
    for f in sorted(glob.glob(os.path.join(RUNS, 'fields', 'C_P[34]_landau_*_n*_m.npy'))):
        m = np.load(f).astype(float); Lx, Ly = m.shape[1:]
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
        for i in range(2):
            S = sum(np.abs(np.fft.fft2(m[3 * i + a] - m[3 * i + a].mean())) ** 2 for a in range(3)); S = np.fft.fftshift(S)
            nx = np.fft.fftshift(np.fft.fftfreq(Lx, 1.0 / Lx)); ny = np.fft.fftshift(np.fft.fftfreq(Ly, 1.0 / Ly))
            kx = 2 * np.pi * nx / Lx; ky = 2 * np.pi * ny / Ly
            ax[i].pcolormesh(kx, ky, np.log10(S.T + 1e-30), vmin=np.log10(S.max()) - 6, cmap='magma', shading='auto')
            ax[i].set_xlim(-0.8, 0.8); ax[i].set_ylim(-0.8, 0.8); ax[i].set_aspect('equal'); ax[i].set_title(f'layer {i+1}'); ax[i].set_xlabel('kx')
        fig.suptitle(os.path.basename(f)[:-6], fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'S_' + os.path.basename(f)[:-6] + '.png'), dpi=110); plt.close(fig)

if __name__ == '__main__':
    figA(); figBg(); figC()
