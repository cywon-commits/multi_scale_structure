import numpy as np, itertools
from landau import averages, ring
from scipy.optimize import minimize
d=np.radians
def geoms(R):
    G={}
    G['1q']        =([np.array([1,0.])],[])
    G['SkX']       =(ring(1,d(np.array([0,120,240]))),[])
    G['2x1q-par']  =([np.array([1,0.])],[np.array([R,0.])])
    G['2x1q-perp'] =([np.array([1,0.])],[np.array([0,R])])
    G['SkX+SkX']   =(ring(1,d(np.array([0,120,240]))),ring(R,d(np.array([7,127,247]))))
    if abs(R-np.sqrt(3))<1e-9: G['RES']=(ring(1,d(np.array([0,120,240]))),ring(R,d(np.array([30,150,270]))))
    if abs(R-2)<1e-9:          G['RES']=(ring(1,d(np.array([0,120,240]))),ring(R,d(np.array([0,120,240]))))
    if abs(R-2*np.cos(np.pi/12))<1e-9:
        G['RES']=(ring(1,d(np.arange(0,180,30))),ring(R,d(np.arange(15,180,30))))
    return G
def polycoef(k1,k2):
    # c3,c4 homogeneous polys in (A1,A2); fit coefficients
    pts=[(1,0),(0,1),(1,1),(1,-1),(2,1),(1,2),(1,3)]
    rows3=[];rows4=[];y3=[];y4=[]
    for A1,A2 in pts:
        modes=[(k,A1) for k in k1]+[(k,A2) for k in k2]
        _,c3,c4=averages(modes)
        rows3.append([A1**(3-j)*A2**j for j in range(4)]); y3.append(c3)
        rows4.append([A1**(4-j)*A2**j for j in range(5)]); y4.append(c4)
    a=np.linalg.lstsq(rows3,y3,rcond=None)[0]; b=np.linalg.lstsq(rows4,y4,rcond=None)[0]
    return a,b
def Fmin(k1,k2,s,g,coef,r=-1.0):
    a,b=coef; n1,n2=len(k1),len(k2)
    w1=((1+s)/2,(1-s)/2); w2=((1-s)/2,(1+s)/2)
    def F(x):
        a1,a2=x[0],(x[1] if n2 else 0)
        tot=r*(n1*a1**2+n2*a2**2)
        for i in range(2):
            A1=a1*np.sqrt(w1[i]); A2=a2*np.sqrt(w2[i])
            tot+=g*sum(a[j]*A1**(3-j)*A2**j for j in range(4))+sum(b[j]*A1**(4-j)*A2**j for j in range(5))
        return tot
    best=0.0
    for x0 in itertools.product([-1.2,-0.4,0.4,1.2],repeat=2):
        res=minimize(F,np.array(x0,float),method='Nelder-Mead',options={'xatol':1e-9,'fatol':1e-12,'maxiter':4000})
        best=min(best,res.fun)
    return best
