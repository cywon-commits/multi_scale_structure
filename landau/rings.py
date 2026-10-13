import numpy as np, itertools
from landau import averages
from scipy.optimize import minimize
from tongue import fband, wts
B1=np.array([1.0,0.0]); B2=np.array([0.5,np.sqrt(3)/2])     # hex reciprocal basis
def shell(N):
    v=[]
    for n in range(-15,16):
        for m in range(-15,16):
            if n*n+n*m+m*m==N: v.append(n*B1+m*B2)
    # one per +/- pair
    out=[]
    for x in v:
        if not any(np.allclose(x,-y) for y in out): out.append(x)
    return out
def orbits(vecs):
    """split a shell into orbits under C6 (rotation by 60deg); chiral shells give two orbits"""
    Rm=np.array([[0.5,-np.sqrt(3)/2],[np.sqrt(3)/2,0.5]]); left=list(vecs); orbs=[]
    while left:
        x=left.pop(0); orb=[x]; y=x.copy()
        for _ in range(5):
            y=Rm@y
            for j,z in enumerate(left):
                if np.allclose(z,y) or np.allclose(z,-y): orb.append(left.pop(j)); break
        orbs.append(orb)
    return orbs
def ring_dirs(Q,angles_deg): return [Q*np.array([np.cos(np.radians(a)),np.sin(np.radians(a))]) for a in angles_deg]
class State:
    """rings: list of lists of 2D vectors (modes); magnitudes in lattice units"""
    def __init__(s,name,rings):
        s.name=name; s.rings=rings; s.n=[len(r) for r in rings]; s.mag=[np.linalg.norm(r[0]) for r in rings]
        k=len(rings); s.k=k
        s.m3=[p for p in itertools.product(range(4),repeat=k) if sum(p)==3]
        s.m4=[p for p in itertools.product(range(5),repeat=k) if sum(p)==4]
        rng=np.random.default_rng(1); X=[];Y3=[];Y4=[]
        for _ in range(len(s.m4)+len(s.m3)+10):
            A=rng.normal(size=k)
            modes=[(v,A[i]) for i,r in enumerate(rings) for v in r]
            _,c3,c4=averages(modes); X.append(A); Y3.append(c3); Y4.append(c4)
        X=np.array(X)
        M3=np.array([[np.prod(x**np.array(p)) for p in s.m3] for x in X]); M4=np.array([[np.prod(x**np.array(p)) for p in s.m4] for x in X])
        s.c3=np.linalg.lstsq(M3,Y3,rcond=None)[0]; s.c4=np.linalg.lstsq(M4,Y4,rcond=None)[0]
        s.c3[np.abs(s.c3)<1e-8]=0; s.c4[np.abs(s.c4)<1e-8]=0
    def F(s,A,lam,c,m,g,K):
        sv=np.sqrt(1-c*c); fs=fband(sv,c)
        xi=[lam*mg-m for mg in s.mag]; W=[wts(x,c) for x in xi]
        r=[-1+K*(fband(x,c)-fs) for x in xi]
        F=sum(s.n[i]*r[i]*A[i]**2 for i in range(s.k))
        for p,co in zip(s.m3,s.c3):
            if co: F+=g*co*np.prod([A[i]**p[i] for i in range(s.k)])*sum(np.prod([W[i][l]**(p[i]/2) for i in range(s.k)]) for l in range(2))
        for p,co in zip(s.m4,s.c4):
            if co: F+=co*np.prod([A[i]**p[i] for i in range(s.k)])*sum(np.prod([W[i][l]**(p[i]/2) for i in range(s.k)]) for l in range(2))
        return F
    def Emin(s,c,m,g,K,lam0):
        best=0.0
        starts=[np.full(s.k,-0.5),np.full(s.k,-0.25)]+[np.eye(s.k)[i]*-0.7 for i in range(s.k)]
        for A0 in starts:
            for dl in [0.0,-0.01,0.01]:
                x0=np.concatenate([A0,[lam0*(1+dl)]])
                r=minimize(lambda x:s.F(x[:-1],x[-1],c,m,g,K),x0,method='Nelder-Mead',options={'xatol':1e-7,'fatol':1e-10,'maxiter':4000*s.k})
                best=min(best,r.fun)
        return best
