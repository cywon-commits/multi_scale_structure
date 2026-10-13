import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from tongue import wts
R12=2*np.cos(np.pi/12)
def key(k): return (round(k[0],7),round(k[1],7))
# 12 modes: Q1 e_n (n=0..5, angle 30n), Q2 = e_n+e_{n+1} (angle 15+30n)
E=[np.array([np.cos(np.radians(30*n)),np.sin(np.radians(30*n))]) for n in range(12)]
Ep=[np.array([np.cos(np.radians(150*n)),np.sin(np.radians(150*n))]) for n in range(12)]   # perp-space images
K1=[E[n] for n in range(6)]; K1p=[Ep[n] for n in range(6)]
K2=[E[n]+E[n+1] for n in range(6)]; K2p=[Ep[n]+Ep[n+1] for n in range(6)]
KV=K1+K2; KP=K1p+K2p
RING=[0]*6+[1]*6
ORB=[n%2 for n in range(6)]+[n%2 for n in range(6)]    # C6 orbit label inside each ring
def averages(amp,th,scale):
    pw=[]
    for k,a,t,s_ in zip(KV,amp,th,scale):
        kh=np.array([k[0],k[1],0])/np.linalg.norm(k); Ev=s_*a*np.exp(1j*t)*(np.array([0,0,1])-1j*kh)/2
        pw.append((k,Ev)); pw.append((-k,np.conj(Ev)))
    P=defaultdict(complex)
    for kb,Eb in pw:
        for kc,Ec in pw: P[key(kb+kc)]+=Eb@Ec
    c4=sum(P[k]*P.get(key(-np.array(k)),0) for k in P).real
    c3=sum(Ea[2]*P.get(key(-ka),0) for ka,Ea in pw).real
    return c3,c4
def F(amp,th,c,g):
    s=np.sqrt(1-c*c); w1=wts(-s,c); w2=wts(s,c)
    tot=-np.sum(amp**2)
    for l in range(2):
        sc=[np.sqrt(w1[l]) if RING[j]==0 else np.sqrt(w2[l]) for j in range(12)]
        c3,c4=averages(amp,th,sc); tot+=g*c3+c4
    return tot
def orbit_amps(x):   # x = 4 orbit amplitudes -> 12 mode amplitudes
    return np.array([x[2*RING[j]+ORB[j]] for j in range(12)])
