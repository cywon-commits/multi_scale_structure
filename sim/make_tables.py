"""Collect runs/*.json into markdown tables (runs/tables.md) and figures (figures/). Landau predictions are read from
runs/landau_predictions.txt (output of landau/predict_point.py at the nominal and, where given, g_eff-corrected points)."""
import glob, json, os, re, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), '..')
RUNS = os.path.join(ROOT, 'runs')


def load_pred():
    pred = {}
    key = None
    for ln in open(os.path.join(RUNS, 'landau_predictions.txt')):
        ln = ln.rstrip()
        if ln.startswith('##'):
            key = tuple(float(x) for x in ln[2:].split()); pred[key] = {}
        elif ln.strip() and key is not None:
            parts = ln.split()
            try:
                pred[key][parts[0]] = float(parts[-1])
            except ValueError:
                pass
    return pred


def peaks_table(run, p, nshow=14, L=512):
    r0 = np.sqrt(p['r_abs']); q1 = p['q_minus']
    rows = ['| n=(nx,ny) | \\|n\\| | k/q₋ | angle° | a₁/√\\|r\\| | a₂/√\\|r\\| | a_mode/√\\|r\\| |', '|---|---|---|---|---|---|---|']
    for pk in run['peaks'][:nshow]:
        rows.append(f"| {tuple(pk['n'])} | {pk['kn']:.2f} | {pk['k']/q1:.3f} | {pk['angle']:.1f} | {pk['a1']/r0:.3f} | {pk['a2']/r0:.3f} | {pk['a_landau']:.3f} |")
    return '\n'.join(rows)


def main():
    pred = load_pred()
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, 'P*.json'))):
        d = json.load(open(f))
        if 'runs' not in d:
            continue
        cfg = d['cfg']; p = d['params']
        key = (cfg['c_rel'], cfg['rho'], cfg['K'], cfg['g'])
        out.append(f"### {os.path.basename(f)[:-5]}\n")
        out.append(f"c/c_L={cfg['c_rel']}, ρ={cfg['rho']}, K={cfg['K']}, g={cfg['g']}, L={d['L']}, |r|={p['r_abs']:.4e}, q₋={p['q_minus']:.4f} (n={p['q_minus']*d['L']/2/np.pi:.2f}), q₊={p['q_plus']:.4f} (n={p['q_plus']*d['L']/2/np.pi:.2f})\n")
        if key in pred:
            out.append('Landau (predict_point): ' + ', '.join(f'{k} {v:.4f}' for k, v in pred[key].items()) + '\n')
        out.append('| seed | ΔE/\\|r\\|² | g_eff (L1, L2) | steps | monotone |\n|---|---|---|---|---|')
        for sk, r in d['runs'].items():
            out.append(f"| {sk} | {r['dE_over_r2']:.4f} | {r['g_eff'][0]:.3f}, {r['g_eff'][1]:.3f} | {r['steps']} | {r['monotone']} |")
        best = min(d['runs'], key=lambda k: d['runs'][k]['dE_over_r2'])
        out.append(f"\nwinner: **{best}**\n")
        out.append(peaks_table(d['runs'][best], p) + '\n')
    open(os.path.join(RUNS, 'tables.md'), 'w').write('\n'.join(out))
    print('wrote tables.md')


if __name__ == '__main__':
    main()
