import numpy as np
from collections import defaultdict
from scipy.optimize import minimize, minimize_scalar, brentq
import phason as P
from tongue import wts, fband, E_nonres_opt
from hess import hessian
R=P.R12
def F_det(amp,th,c,g,lam,m,K):
    s=np.sqrt(1-c*c); fs=fband(s,c)
    xi=[lam-m,R*lam-m]; W=[wts(x,c) for x in xi]; r=[-1+K*(fband(x,c)-fs) for x in xi]
    tot=sum(r[P.RING[j]]*amp[j]**2 for j in range(12))
    for l in range(2):
        sc=[np.sqrt(W[P.RING[j]][l]) for j in range(12)]
        c3,c4=P.averages(amp,th,sc); tot+=g*c3+c4
    return tot
def equilibrium(c,g,m,K,x0=None):
    s=np.sqrt(1-c*c)
    def Fl(lam,ret=False):
        r=minimize(lambda x:F_det(P.orbit_amps(x),np.zeros(12),c,g,lam,m,K),x0 if x0 is not None else np.array([-.6,-.1,-.6,-.1]),
                   method='Nelder-Mead',options={'xatol':1e-10,'fatol':1e-13,'maxiter':20000})
        return (r.fun,r.x) if ret else r.fun
    lam0=m-s; rr=minimize_scalar(Fl,bounds=(lam0-0.05,lam0+0.05),method='bounded',options={'xatol':1e-9})
    E,x=Fl(rr.x,True); return E,x,rr.x
def min_eig(c,g,m,K,x,lam,pmax=0.25):
    amp=P.orbit_amps(x); x0=np.concatenate([amp,np.zeros(12)])
    H=hessian(lambda z:F_det(z[:12],z[12:],c,g,lam,m,K),x0)
    T=np.diag(np.concatenate([np.ones(12),1/amp])); Huv=T@H@T
    kv=[lam*k if P.RING[j]==0 else lam*k for j,k in enumerate(P.KV)]
    def M(p):
        es=[];ea=[]
        for j,k in enumerate(kv):
            q0=np.linalg.norm(k); e=lambda P_: K*(fband(np.linalg.norm(k+P_)-m,c)-fband(q0-m,c))
            ep,em=e(p),e(-p); es.append(0.5*(ep+em)); ea.append(0.5*(ep-em))
        B=np.zeros((24,24),complex); B[:12,:12]=np.diag(es); B[12:,12:]=np.diag(es); B[:12,12:]=np.diag(1j*np.array(ea)); B[12:,:12]=np.diag(-1j*np.array(ea))
        return Huv+2*B
    best=(np.inf,None)
    for pm in np.linspace(0.005,pmax,25):
        for ang in np.radians(np.arange(0,180,7.5)):
            p=pm*np.array([np.cos(ang),np.sin(ang)]); w=np.linalg.eigvalsh(M(p))[0]
            if w<best[0]: best=(w,(pm,np.degrees(ang)))
    return best
