import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scan import geoms, polycoef
from exact import cubquart
RS={'sqrt3':np.sqrt(3),'qc12':2*np.cos(np.pi/12),'two':2.0}
GEO={k:geoms(R)['RES'] for k,R in RS.items()}
COE={k:polycoef(*GEO[k]) for k in RS}
TH=np.linspace(0,2*np.pi,721)[:-1]
def fband(xi,gc): return 0.5*(xi**2+1)+gc-np.sqrt(xi**2+gc**2)
def wts(xi,gc):
    t=xi/np.sqrt(xi**2+gc**2); return np.array([(1-t)/2,(1+t)/2])
def Epolar(th,r1,r2,n1,n2,a,b,w1,w2,g):
    c,s_=np.cos(th),np.sin(th); n=-(n1*r1*c*c+n2*r2*s_*s_)
    c3=0;c4=0
    for i in range(2):
        A1=c*np.sqrt(w1[i]);A2=s_*np.sqrt(w2[i])
        c3=c3+sum(a[j]*A1**(3-j)*A2**j for j in range(4)); c4=c4+sum(b[j]*A1**(4-j)*A2**j for j in range(5))
    c3=g*c3; disc=9*c3**2+32*n*c4
    rho=np.where(disc>0,(-3*c3+np.sqrt(np.maximum(disc,0)))/(8*c4),0); rho=np.maximum(rho,0)
    return np.minimum(-n*rho**2+c3*rho**3+c4*rho**4,0)
def E_res_xi(k,gc,m,xi1,g,K,ret=False):
    R=RS[k]; s=np.sqrt(1-gc**2); xi2=R*(m+xi1)-m
    r1=-1+K*(fband(xi1,gc)-fband(s,gc)); r2=-1+K*(fband(xi2,gc)-fband(s,gc))
    k1,k2=GEO[k]; a,b=COE[k]; w1=wts(xi1,gc); w2=wts(xi2,gc)
    f=lambda th: Epolar(th,r1,r2,len(k1),len(k2),a,b,w1,w2,g)
    v=f(TH); i=np.argmin(v); h=TH[1]-TH[0]
    r=minimize_scalar(f,bounds=(TH[i]-h,TH[i]+h),method='bounded',options={'xatol':1e-10})
    return min(float(r.fun),float(v[i]))
def E_res(k,gc,rho,g,K):
    m=(rho+1)/(rho-1); s=np.sqrt(1-gc**2)
    xs=np.linspace(-s-0.6,-s+0.6,25); v=[E_res_xi(k,gc,m,x,g,K) for x in xs]; i=int(np.argmin(v))
    r=minimize_scalar(lambda x:E_res_xi(k,gc,m,x,g,K),bounds=(xs[max(i-1,0)],xs[min(i+1,24)]),method='bounded',options={'xatol':1e-8})
    return min(r.fun,v[i])
def E_nonres(gc,g):
    s=np.sqrt(1-gc**2); w=np.array([(1+s)/2,(1-s)/2]); P2=(w**2).sum(); P32=(w**1.5).sum(); X=2*w[0]*w[1]
    E={'1q':-1/(4*P2),'2x1q-perp':-2/(5-s*s),'SkX':cubquart(3,2.25*P32*g,12.75*P2),
       'SkX+SkX':cubquart(6,4.5*P32*g,25.5*P2+31.5*X)}
    k=min(E,key=E.get); return k,E[k]
def rho_star(k,gc):
    x=(RS[k]-1)/(RS[k]+1); s=np.sqrt(1-gc**2); return (s+x)/(s-x) if s>x else None

def E_nonres_opt(gc,g,K):
    """non-resonant states with wavevector positions also optimized (consistent with E_res)"""
    s=np.sqrt(1-gc**2); fs=fband(s,gc)
    def parts(xi):
        w=wts(xi,gc); r=-1+K*(fband(xi,gc)-fs)
        return w,r,(w**2).sum(),(w**1.5).sum()
    def one(fun):
        xs=np.linspace(-s-0.5,min(-s+0.8,0),40); v=[fun(x) for x in xs]; i=int(np.argmin(v))
        r=minimize_scalar(fun,bounds=(xs[max(i-1,0)],xs[min(i+1,39)]),method='bounded',options={'xatol':1e-9})
        return min(r.fun,v[i])
    def q1(xi):
        w,r,P2,_=parts(xi); return -(r*r)/(4*P2) if r<0 else 0.0
    def skx(xi):
        w,r,P2,P32=parts(xi); return cubquart(-3*r,2.25*P32*g,12.75*P2) if r<0 else 0.0  # r<0 needed only approx
    def perp(xi):      # symmetric pair at xi and -xi
        w,r,P2,_=parts(xi); X=2*w[0]*w[1]
        K4=2*P2+3*X; return -(2*r)**2/(4*K4) if r<0 else 0.0
    def skx2(xi):
        w,r,P2,P32=parts(xi); X=2*w[0]*w[1]; return cubquart(-6*r,4.5*P32*g,25.5*P2+31.5*X) if r<0 else 0.0
    E={'1q':one(q1),'SkX':one(skx),'2x1q-perp':one(perp),'SkX+SkX':one(skx2)}
    k=min(E,key=E.get); return k,E[k]
