"""Minimal resonant states missing from the original candidate set (theory notes sec. 18).
   V    : quartet 2k- = k+ + k+'  (exists if R=q+/q- < 2; tilt cos(theta)=1/R), dominant at g=0, large c
   T+   : triad k+ + k+' + k- = 0 (two q+ modes, one q- mode), favoured by field at large c
   T-   : triad k- + k-' + k+ = 0 (exists if R < 2)
   HEX3 : hex lattice with shells (1, sqrt3, 2), lattice constant optimised -> prints seed data
usage: python minimal_states.py c_rel rho K g"""
import sys, numpy as np
from scipy.optimize import minimize
from rings import State
from cands import build2
def geometry(c,rho):
    s=np.sqrt(1-c*c); m=(rho+1)/(rho-1); R=(m+s)/(m-s); return s,m,R
def states(c,rho):
    s,m,R=geometry(c,rho); out={}
    e=lambda a: np.array([np.cos(a),np.sin(a)])
    if R<2:
        th=np.arccos(1/R); out['V']=State('V',[[e(0)],[R*e(th)],[R*e(-th)]])
        ph=np.arccos(R/2); out['T-']=State('T-',[[e(ph)],[e(-ph)],[R*e(np.pi)]])
    ph=np.arccos(1/(2*R)); out['T+']=State('T+',[[e(0)],[R*e(np.pi-ph)],[R*e(np.pi+ph)]])
    return s,m,R,out
def hex3_seed(c,rho,K,g):
    s,m,R=geometry(c,rho); H=build2()['HEX(1,√3,2)']; lam0=m-s; best=None
    for A0 in [np.array([-.5,-.3,-.3]),np.array([-.6,-.1,-.4]),np.array([-.6,-.4,-.1])]:
        for dl in [-0.04,0,0.04]:
            r=minimize(lambda x:H.F(x[:3],x[3],c,m,g,K),np.concatenate([A0,[lam0*(1+dl)]]),method='Nelder-Mead',
                       options={'xatol':1e-8,'fatol':1e-11,'maxiter':20000})
            if best is None or r.fun<best.fun: best=r
    return best.fun,best.x[3]/lam0,best.x[:3]
if __name__=='__main__':
    c,rho,K,g=map(float,sys.argv[1:5]); s,m,R,S=states(c,rho)
    print(f'band-minimum ratio R = q+/q- = {R:.4f}')
    for k,st in S.items(): print(f'{k:4s} E = {st.Emin(c,m,g,K,m-s): .5f}')
    E,f,a=hex3_seed(c,rho,K,g)
    print(f'HEX3 E = {E: .5f}   Q1 = {f:.4f} x q-   ring amplitudes (1, sqrt3, 2) = {np.round(a,4)}  (signs matter)')
