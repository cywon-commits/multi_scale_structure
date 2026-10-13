import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
import phason as P
from tongue import wts
def F_extra(amp,c,g,kx,bx,ring):
    s=np.sqrt(1-c*c); w1=wts(-s,c); w2=wts(s,c); tot=-np.sum(amp**2)-bx**2
    KV=P.KV+[kx]
    for l in range(2):
        sc=[np.sqrt(w1[l]) if P.RING[j]==0 else np.sqrt(w2[l]) for j in range(12)]+[np.sqrt((w1 if ring==0 else w2)[l])]
        pw=[]
        for k,a,s_ in zip(KV,list(amp)+[bx],sc):
            kh=np.array([k[0],k[1],0])/np.linalg.norm(k); Ev=s_*a*(np.array([0,0,1])-1j*kh)/2
            pw.append((k,Ev)); pw.append((-k,np.conj(Ev)))
        Pd=defaultdict(complex)
        for kb,Eb in pw:
            for kc,Ec in pw: Pd[P.key(kb+kc)]+=Eb@Ec
        c4=sum(Pd[k]*Pd.get(P.key(-np.array(k)),0) for k in Pd).real
        c3=sum(Ea[2]*Pd.get(P.key(-ka),0) for ka,Ea in pw).real
        tot+=g*c3+c4
    return tot
def scan(c,g):
    f=lambda x: P.F(P.orbit_amps(x),np.zeros(12),c,g)
    r=minimize(f,np.array([-.55,-.2,-.6,-.2]),method='Nelder-Mead',options={'xatol':1e-10,'fatol':1e-13,'maxiter':20000})
    amp=P.orbit_amps(r.x); out=[]
    for ring,Q in [(0,1.0),(1,P.R12)]:
        for phi in np.arange(1,180,2.0):
            if min(abs(((phi-a)+90)%180-90) for a in (np.arange(0,180,30) if ring==0 else np.arange(15,180,30)))<0.5: continue
            kx=Q*np.array([np.cos(np.radians(phi)),np.sin(np.radians(phi))]); b=1e-3
            h=(F_extra(amp,c,g,kx,b,ring)+F_extra(amp,c,g,kx,-b,ring)-2*F_extra(amp,c,g,kx,0.0,ring))/b**2
            out.append((ring,phi,h))
    return out
if __name__=='__main__':
    for c in [0.3,0.6]:
        o=scan(c,4.0); h=np.array([t[2] for t in o]); i=int(np.argmin(h))
        print('c=%.1f: min stiffness for a new mode = %.4f at ring %d, angle %.0f deg ; max %.3f'%(c,h[i],o[i][0],o[i][1],h.max()),flush=True)
