"""Landau prediction at one parameter point: energies (units |r|^2/u) of all candidate states, lowest first.
   usage: python predict_point.py c_rel rho K g      e.g.  python predict_point.py 0.6 2.318 150 4
   rho = D2/D1 (bare ratio). Candidates are placed at the band minima for this rho and may detune (finite K)."""
import sys, numpy as np
from cands import build2, R12
from tongue import E_nonres_opt
def predict(c,rho,K,g,keys=('QC12s','HEX(1,√3,2)','HEX(1,2)','APXs','QC8')):
    S=build2(); s=np.sqrt(1-c*c); m=(rho+1)/(rho-1)        # m = qbar/delta
    E={k:S[k].Emin(c,m,g,K,(m-s)/S[k].mag[0]) for k in keys}
    k_inc,e_inc=E_nonres_opt(c,g,K); E['nonres:'+k_inc]=e_inc
    return dict(sorted(E.items(),key=lambda kv:kv[1]))
if __name__=='__main__':
    c,rho,K,g=map(float,sys.argv[1:5])
    for k,v in predict(c,rho,K,g).items(): print(f'{k:16s} {v: .5f}')
