"""field continuation from a converged M1 state: python llg_cont.py M1_d+0 0.35,0.4,..."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llg_m import *
import llg, llg_m


def continuation(name_in, hfracs, nper_chunk=2500, nmax=20000):
    """raise H stepwise from a converged zero-... state, relaxing at each field; report E - E_sat and amplitude"""
    m = np.load(os.path.join(RUNS, 'fields', f'llgM_{name_in}_m.npy')).astype(float)
    out = []
    for hf in hfracs:
        Hh = hf * HS
        sim = llg.LLG(A, D1, RHO * D1, J / 2, Hh, 0.0, L=(LX, LY), B=(BX, BY)); sim.set_m(m)
        Eprev = None
        for c in range(nmax // nper_chunk):
            sim.step(nper_chunk); E = sim.energy()
            if Eprev is not None and abs(E - Eprev) < 1e-11 and sim.residual() < 1e-7: break
            Eprev = E
        r, m = measure(sim); mz = sim.mz_mean()
        dE = E - (-2 * Hh); out.append(dict(hfrac=hf, H=Hh, E=E, dE_sat=dE, mz=mz, A0=r['A_big'], steps=(c + 1) * nper_chunk))
        print(f'H/Hs={hf:.3f} (H={Hh:.4f}): E-Esat={dE:.3e} mz={np.round(mz,4)} A0={r["A_big"]} rel={r["rel"]}', flush=True)
        json.dump(out, open(os.path.join(RUNS, f'llgM_cont_{name_in}.json'), 'w'), indent=1, default=float)
        np.save(os.path.join(RUNS, 'fields', f'llgM_cont_{name_in}_h{hf}_m.npy'), m.astype(np.float32))
        if dE > -1e-12: break
    return out


if __name__ == '__main__':
    nm = sys.argv[1]; hs = [float(x) for x in sys.argv[2].split(',')]
    res = continuation(nm, hs); json.dump(res, open(os.path.join(RUNS, f'llgM_cont_{nm}.json'), 'w'), indent=1, default=float)
