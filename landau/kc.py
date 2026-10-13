import sys, json, time, numpy as np
from scipy.optimize import minimize_scalar
from tongue import *
def margin_c(gc,g,K):
    """best (most negative) E_QC - min(E_hex,E_inc) over rho near QC centre, at fixed c"""
    rs=rho_star('qc12',gc)
    if rs is None: return 1.0,None
    Ei=E_nonres_opt(gc,g,K)[1]
    D=lambda r: E_res('qc12',gc,r,g,K)-min(Ei,E_res('sqrt3',gc,r,g,K),E_res('two',gc,r,g,K))
    xs=rs*np.linspace(0.96,1.04,9); v=[D(x) for x in xs]; i=int(np.argmin(v))
    r=minimize_scalar(D,bounds=(xs[max(i-1,0)],xs[min(i+1,8)]),method='bounded',options={'xatol':1e-5})
    return (r.fun,r.x) if r.fun<v[i] else (v[i],xs[i])
def margin(g,K):
    cs=np.linspace(0.5,0.93,10); v=[margin_c(c,g,K)[0] for c in cs]; i=int(np.argmin(v))
    r=minimize_scalar(lambda c:margin_c(c,g,K)[0],bounds=(cs[max(i-1,0)],cs[min(i+1,9)]),method='bounded',options={'xatol':1e-4})
    return (r.fun,r.x) if r.fun<v[i] else (v[i],cs[i])
if __name__=='__main__':
    g=float(sys.argv[1]); t=time.time()
    lo,hi=float(sys.argv[2]),float(sys.argv[3])   # log-bisection: margin(lo)>0 (no island), margin(hi)<0
    for _ in range(int(sys.argv[4])):
        mid=np.sqrt(lo*hi); m,c=margin(g,mid)
        if m<0: hi=mid
        else: lo=mid
        print('K=%.1f margin=%.4f at c=%.3f'%(mid,m,c),flush=True)
    print('g',g,'Kc in',lo,hi,round(time.time()-t,1))
