import numpy as np
from scipy.optimize import minimize_scalar
import tongue as T
def E_res(k,gc,rho,g,K):
    m=(rho+1)/(rho-1); s=np.sqrt(1-gc**2)
    xs=np.linspace(-s-0.25,-s+0.25,11); v=[T.E_res_xi(k,gc,m,x,g,K) for x in xs]; i=int(np.argmin(v))
    r=minimize_scalar(lambda x:T.E_res_xi(k,gc,m,x,g,K),bounds=(xs[max(i-1,0)],xs[min(i+1,10)]),method='bounded',options={'xatol':1e-7})
    return min(r.fun,v[i])
def margin_c(gc,g,K,Ei=None):
    rs=T.rho_star('qc12',gc)
    if rs is None: return 1.0,None
    if Ei is None: Ei=T.E_nonres_opt(gc,g,K)[1]
    D=lambda r: E_res('qc12',gc,r,g,K)-min(Ei,E_res('sqrt3',gc,r,g,K),E_res('two',gc,r,g,K))
    xs=rs*np.linspace(0.97,1.03,5); v=[D(x) for x in xs]; i=int(np.argmin(v))
    r=minimize_scalar(D,bounds=(xs[max(i-1,0)],xs[min(i+1,4)]),method='bounded',options={'xatol':1e-5,'maxiter':12})
    return (r.fun,r.x) if r.fun<v[i] else (v[i],xs[i])
def margin(g,K,cs=None):
    cs=np.linspace(0.52,0.92,7) if cs is None else cs
    v=[margin_c(c,g,K)[0] for c in cs]; i=int(np.argmin(v))
    r=minimize_scalar(lambda c:margin_c(c,g,K)[0],bounds=(cs[max(i-1,0)],cs[min(i+1,len(cs)-1)]),method='bounded',options={'xatol':2e-3,'maxiter':10})
    return (r.fun,r.x) if r.fun<v[i] else (v[i],cs[i])
