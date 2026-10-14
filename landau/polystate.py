import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from tongue import wts
class Poly:
    """K=infinity energy of a ring State with all rings at band minima; band[i]=0 (q-) or 1 (q+)"""
    def __init__(s,state,band):
        s.st=state; s.band=np.array(band); s.k=state.k; s.n=np.array(state.n,float)
        s.P3=np.array(state.m3,float); s.C3=np.array(state.c3); s.P4=np.array(state.m4,float); s.C4=np.array(state.c4)
        k3=s.C3!=0; k4=s.C4!=0; s.P3,s.C3,s.P4,s.C4=s.P3[k3],s.C3[k3],s.P4[k4],s.C4[k4]
    def setc(s,c):
        sv=np.sqrt(1-c*c); W=[wts(-sv,c),wts(sv,c)]; Wr=np.array([W[b] for b in s.band])   # (k,2)
        s.L3=np.array([sum(np.prod(Wr[:,l]**(p/2)) for l in range(2)) for p in s.P3])
        s.L4=np.array([sum(np.prod(Wr[:,l]**(p/2)) for l in range(2)) for p in s.P4])
    def F(s,A,g):
        A=np.asarray(A); 
        t3=np.prod(A[None,:]**s.P3,axis=1); t4=np.prod(A[None,:]**s.P4,axis=1)
        return -np.sum(s.n*A*A)+g*np.sum(s.C3*s.L3*t3)+np.sum(s.C4*s.L4*t4)
    def grad(s,A,g):
        A=np.asarray(A); G=-2*s.n*A
        for P,C,L,fac in ((s.P3,s.C3,s.L3,g),(s.P4,s.C4,s.L4,1.0)):
            for i in range(s.k):
                Pi=P.copy(); m=Pi[:,i]>0; Pi[m,i]-=1
                G[i]+=fac*np.sum((C*L*P[:,i]*np.prod(A[None,:]**Pi,axis=1))[m])
        return G
    def emin(s,g,starts=None,rng=np.random.default_rng(0)):
        best=0.0; xb=None
        starts=starts if starts is not None else [rng.choice([-1,1],s.k)*rng.uniform(0.2,0.8,s.k) for _ in range(10)]+[np.full(s.k,-0.5),np.full(s.k,-0.3)]
        for x0 in starts:
            r=minimize(s.F,x0,args=(g,),jac=s.grad,method='BFGS',options={'gtol':1e-10})
            if r.fun<best: best,xb=r.fun,r.x
        return best,xb
