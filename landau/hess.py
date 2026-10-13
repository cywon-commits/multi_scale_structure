import numpy as np
from phason import *
def hessian(f,x0,h=2e-4):
    n=len(x0); H=np.zeros((n,n)); f0=f(x0)
    for i in range(n):
        for j in range(i,n):
            if i==j:
                e=np.zeros(n); e[i]=h; H[i,i]=(f(x0+e)-2*f0+f(x0-e))/h**2
            else:
                ei=np.zeros(n); ej=np.zeros(n); ei[i]=h; ej[j]=h
                H[i,j]=H[j,i]=(f(x0+ei+ej)-f(x0+ei-ej)-f(x0-ei+ej)+f(x0-ei-ej))/(4*h*h)
    return H
def analyse(xorb,c,g,label):
    a0=orbit_amps(xorb); x0=np.concatenate([a0,np.zeros(12)])
    f=lambda x: F(x[:12],x[12:],c,g)
    H=hessian(f,x0)
    # Goldstone vectors (phase part only)
    G=[]
    for d in range(2):
        v=np.zeros(24); v[12:]=[k[d] for k in KV]; G.append(v)
        v=np.zeros(24); v[12:]=[k[d] for k in KP]; G.append(v)
    G=np.array(G); res=[np.linalg.norm(H@v)/np.linalg.norm(v) for v in G]
    w,V=np.linalg.eigh(H)
    print(f'--- {label}: eigenvalues of full 24x24 Hessian (sorted)')
    print('  ',np.round(w,4))
    print('   |H v| for phonon_x, phason_x, phonon_y, phason_y:',np.round(res,6))
    neg=w<-1e-4
    if neg.any():
        for i in np.where(neg)[0]:
            v=V[:,i]; amp=v[:12]; ph=v[12:]
            print('   negative mode %.4f: amp part %.3f phase part %.3f'%(w[i],np.linalg.norm(amp),np.linalg.norm(ph)))
            print('     amp pattern by (ring,orbit):',[round(float(np.mean(amp[[j for j in range(12) if RING[j]==r_ and ORB[j]==o]])),3) for r_ in (0,1) for o in (0,1)])
    return H,w,V
if __name__=='__main__':
    c,g=0.7,4.0
    analyse(np.load('qc12s_x.npy'),c,g,'6-fold QC (split orbits)')
    y=np.load('qc12sym_x.npy'); analyse(np.array([y[0],y[0],y[1],y[1]]),c,g,'12-fold symmetric')
