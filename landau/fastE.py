import numpy as np
from scipy.optimize import minimize_scalar
from exact import Panel, lf
TH=np.linspace(0,2*np.pi,1441)[:-1]
def Fpolar(th,n1,n2,a,b,w1,w2,g):
    c,s_=np.cos(th),np.sin(th)
    n=n1*c*c+n2*s_*s_
    c3=0; c4=0
    for i in range(2):
        A1=c*np.sqrt(w1[i]); A2=s_*np.sqrt(w2[i])
        c3=c3+sum(a[j]*A1**(3-j)*A2**j for j in range(4))
        c4=c4+sum(b[j]*A1**(4-j)*A2**j for j in range(5))
    c3=g*c3
    rho=(-3*c3+np.sqrt(9*c3**2+32*n*c4))/(8*c4)
    return -n*rho**2+c3*rho**3+c4*rho**4
class FastPanel(Panel):
    def E(s_,k,s,g):
        if k in ('RES',) or (k=='2x1q-par' and s_.res_par):
            w1,w2=lf(s); a,b=s_.C[k]; n1,n2=len(s_.G[k][0]),len(s_.G[k][1])
            f=lambda th: Fpolar(th,n1,n2,a,b,w1,w2,g)
            v=f(TH); i=np.argmin(v); h=TH[1]-TH[0]
            r=minimize_scalar(f,bounds=(TH[i]-h,TH[i]+h),method='bounded',options={'xatol':1e-12})
            return min(0.0,float(r.fun),float(v[i]))
        return Panel.E(s_,k,s,g)
