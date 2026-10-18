"""E_int(theta) = dE_both - dE_minus - dE_plus(theta) from runs/llg_ctrl.json (+ M1 for H = 0.3 H_sat)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llg_ctrl import OUT, DELTAS, HF, RUNS
from llg_m import HS, PHI2
d = json.load(open(OUT))
M1 = {k: json.load(open(os.path.join(RUNS, f'llgM_M1_d{k}.json'))) for k in DELTAS}
rows = []
for hf in HF:
    H = hf * HS
    mins = [d[f'minus|{s}|{hf}']['dE_sat'] for s in ('-5', '+5') if f'minus|{s}|{hf}' in d]
    if not mins: continue
    em = float(np.mean(mins))
    for k in DELTAS:
        if hf == 0.3:
            eb = M1[k]['final']['E'] + 2 * H; ebe = None
        else:
            b = d.get(f'both|{k}|{hf}')
            if b is None: continue
            eb, ebe = b['dE_sat'], b['dE_last_chunk']
        p = d.get(f'plus|{k}|{hf}')
        if p is None or 'dE_sat' not in p: continue
        rows.append(dict(hf=hf, delta=k, theta_seed=float(M1[k]['seed_info']['theta_box']), dE_both=eb, dE_both_lastchunk=ebe,
                         dE_minus=em, minus_spread=float(np.ptp(mins)) if len(mins) > 1 else None, dE_plus=p['dE_sat'],
                         E_int=eb - em - p['dE_sat'], u_angle=p.get('angle_u1_deg'), u_len=p.get('len_u_over_qp'), gen=p.get('gen')))
json.dump(rows, open(os.path.join(RUNS, 'llg_ctrl_table.json'), 'w'), indent=1, default=float)
for hf in HF:
    rr = [r for r in rows if r['hf'] == hf]
    if not rr: continue
    print(f'H/H_sat(1D) = {hf}')
    for r in rr:
        print(f"  d={r['delta']:>4} dE_both={r['dE_both']:.5e} dE_minus={r['dE_minus']:.5e} dE_plus={r['dE_plus']:.5e}  E_int={r['E_int']:.4e}  u1 angle={r['u_angle']} |u|/q+={np.round(r['u_len'],3) if r['u_len'] else None}")
