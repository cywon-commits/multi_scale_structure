import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

def key(k): return (round(k[0],7), round(k[1],7))

def plane_waves(modes):
    """modes: list of (kvec(2), amp). Neel cycloid a(z cos kr + khat sin kr).
       returns list of (k, Evec(3) complex)"""
    pw=[]
    for k,a in modes:
        k=np.asarray(k,float); kh=np.array([k[0],k[1],0])/np.linalg.norm(k)
        E=a*(np.array([0,0,1])-1j*kh)/2
        pw.append((k,E)); pw.append((-k,np.conj(E)))
    return pw

def averages(modes):
    pw=plane_waves(modes)
    P=defaultdict(complex); # Fourier comps of dm^2
    for kb,Eb in pw:
        for kc,Ec in pw:
            P[key(kb+kc)]+=Eb@Ec
    m2=P.get((0.0,0.0),0).real
    c4=sum((P[k]*P.get(key(-np.array(k)),0)) for k in P).real
    c3=sum((Ea[2]*P.get(key(-ka),0)) for ka,Ea in pw).real
    return m2,c3,c4

def ring(Q,angles): return [Q*np.array([np.cos(t),np.sin(t)]) for t in angles]

def F(a1,a2,geom,s,g,r=-1.0):
    """two layers; scale-1 modes weight w1=((1+s)/2,(1-s)/2), scale-2 mirrored"""
    k1,k2=geom
    w1=((1+s)/2,(1-s)/2); w2=((1-s)/2,(1+s)/2)
    tot=r*(len(k1)*a1**2+len(k2)*a2**2)
    for i in range(2):
        modes=[(k,a1*np.sqrt(w1[i])) for k in k1]+[(k,a2*np.sqrt(w2[i])) for k in k2]
        if not modes: continue
        m2,c3,c4=averages(modes)
        tot+=g*c3+c4
    return tot
