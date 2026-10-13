"""Step 3: hysteresis in rho at K=400, g=4, c/c_L=0.6.  rho: 2.25 -> 2.50 -> 2.25 with warm starts.
usage: python run_hysteresis.py [--L 512] [--drho 0.025] [--out ../runs/hysteresis_L512.json]
At each rho: new model parameters (landau_to_model), keep the state, recalibrate (H1,H2) to M_target, relax.
Observables per rho: dE/|r|^2 (relative to uniform state), g_eff, ring metrics around Q2 ~ 1.93 q- (QC) vs 2 q- (hex 2Q1 shell),
number of peaks in the shell window, amplitude sums.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from params import landau_to_model
import gl_gpu as G
import analysis as an
import run_point as rp


def ring_metrics(m, p, rel=0.01):
    pk = an.find_peaks(m, p, nmax=80, rel=rel)
    q1 = p['q_minus']
    out = {}
    for name, (lo, hi) in dict(Q1=(0.85, 1.15), sqrt3=(1.6, 1.85), Q2_2Q1=(1.88, 2.12)).items():
        sel = [d for d in pk if lo * q1 <= d['k'] <= hi * q1]
        out[name] = dict(n=len(sel), a_sum=float(sum(d['a_landau'] for d in sel)),
                         a_max=float(max([d['a_landau'] for d in sel], default=0.0)),
                         a_big=[round(d['a_landau'], 3) for d in sel][:12])
    return out, pk


def settle(sim, p, tau_steps, nmax, tol, calib=True):
    if calib:
        rp.calibrate(sim, p, 6, 1500, 250, verbose=False)
        p['H1'], p['H2'] = sim.p['H1'], sim.p['H2']
    E, mono = rp.relax_checked(sim, nmax, 250, tol, p, verbose=False)
    return E, mono


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, default=512); ap.add_argument('--drho', type=float, default=0.025)
    ap.add_argument('--K', type=float, default=400); ap.add_argument('--c', type=float, default=0.6); ap.add_argument('--g', type=float, default=4.0)
    ap.add_argument('--rho0', type=float, default=2.25); ap.add_argument('--rho1', type=float, default=2.50)
    ap.add_argument('--tau', type=float, default=2.0); ap.add_argument('--kappa', type=float, default=12.0)
    ap.add_argument('--nmax', type=int, default=6000); ap.add_argument('--tol', type=float, default=1e-8)
    ap.add_argument('--start', default='qc'); ap.add_argument('--out')
    a = ap.parse_args()
    out = a.out or os.path.join(rp.ROOT, 'runs', f'hysteresis_L{a.L}_K{a.K:g}.json')
    n = int(round((a.rho1 - a.rho0) / a.drho))
    rhos = list(a.rho0 + a.drho * np.arange(n + 1))
    path = [('up', r) for r in rhos] + [('down', r) for r in rhos[-2::-1]]
    rec = []
    sim = None
    for k, (d, rho) in enumerate(path):
        t0 = time.time()
        p = landau_to_model(a.c, rho, a.K, a.g, q1=2 * np.pi / 32)
        if sim is None:
            sim = rp.make_sim(p, a.L, a.tau, a.start, kappa=a.kappa)
        else:
            m = sim.state()
            sim = G.GLBilayer(p, L=a.L, dt=a.tau / p['r_abs'], stab=a.kappa * p['r_abs'])
            sim.m = G.xp.asarray(m)
        E, mono = settle(sim, p, a.tau, a.nmax, a.tol)
        m = sim.state()
        met, pk = ring_metrics(m, p)
        Eu, _ = sim.uniform_energy()
        r = dict(dir=d, rho=float(rho), dE=(sim.energy() - Eu) / p['r_abs'] ** 2, mz=list(sim.mz_mean()),
                 g_eff=[G.g_eff(p, v) for v in sim.mz_mean()], monotone=mono, steps=int((len(E) - 1) * 250), metrics=met,
                 top=[(d_['n'], round(d_['a_landau'], 3)) for d_ in pk[:14]], wall=time.time() - t0)
        rec.append(r)
        print(f"{d:4s} rho={rho:.4f} dE={r['dE']:.4f} g_eff={r['g_eff'][0]:.3f} nQ2shell={met['Q2_2Q1']['n']} aQ2={met['Q2_2Q1']['a_sum']:.3f} sqrt3={met['sqrt3']['a_sum']:.3f} steps={r['steps']} {r['wall']:.0f}s", flush=True)
        json.dump(rec, open(out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
