import numpy as np
from scipy.optimize import minimize, brentq
from scan import geoms, polycoef
def lf(s):
    w1=np.array([(1+s)/2,(1-s)/2]); return w1, w1[::-1]
def cubquart(n,al,be):          # min of -n a^2 - al a^3 + be a^4 (al>=0)
    a=(3*al+np.sqrt(9*al**2+32*n*be))/(8*be); return -n*a*a-al*a**3+be*a**4
class Panel:
    def __init__(s_,R):
        s_.G=geoms(R); s_.C={k:polycoef(*v) for k,v in s_.G.items()}
        s_.R=R; s_.res_par = abs(R-2)<1e-9
    def E(s_,k,s,g):
        w1,w2=lf(s); P2=(w1**2).sum(); P32=(w1**1.5).sum(); X=(w1*w2).sum()
        if k=='PM': return 0.0
        if k=='1q': return -1/(4*P2)
        if k=='2x1q-perp': return -2/(5-s*s)
        if k=='2x1q-par' and not s_.res_par: return -1/(3-s*s)
        if k=='SkX': return cubquart(3,2.25*P32*g,12.75*P2)
        if k=='SkX+SkX':      # symmetric a1=a2 (checked against free 2-amplitude min)
            return cubquart(6,4.5*P32*g,25.5*P2+31.5*X)
        a,b=s_.C[k]; n1,n2=len(s_.G[k][0]),len(s_.G[k][1])
        def F(x):
            A1=x[0]*np.sqrt(w1); A2=x[1]*np.sqrt(w2)
            return -(n1*x[0]**2+n2*x[1]**2)+g*sum(a[j]*A1**(3-j)*A2**j for j in range(4)).sum()+sum(b[j]*A1**(4-j)*A2**j for j in range(5)).sum()
        best=0.0
        for x0 in [(-.6,-.6),(-.6,.6),(.6,.6),(.6,-.6),(-.3,-.1),(-.1,-.3),(-.8,-.2),(-.2,-.8)]:
            r=minimize(F,np.array(x0),method='BFGS',options={'gtol':1e-11}); best=min(best,r.fun)
        return best
    def win(s_,s,g):
        E={k:s_.E(k,s,g) for k in ['PM']+list(s_.G)}
        m=min(E.values()); order=['PM','1q','SkX','2x1q-par','2x1q-perp','SkX+SkX','RES']
        return [k for k in order if k in E and E[k]<m+1e-9][0], E
