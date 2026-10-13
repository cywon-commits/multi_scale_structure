import sys, json, time, numpy as np
from scipy.optimize import brentq
from tongue import *
K=float(sys.argv[1]); g=float(sys.argv[2]); mode=sys.argv[3]; i0,i1=int(sys.argv[4]),int(sys.argv[5])
RHO=np.exp(np.linspace(np.log(1.15),np.log(8),48)); CC=np.linspace(0.04,0.98,48)
NR={}
def nr(gc):
    k=round(gc,10)
    if k not in NR: NR[k]=E_nonres_opt(gc,g,K)
    return NR[k]
def allE(gc,rho):
    E={'nr:'+nr(gc)[0]:nr(gc)[1]}
    for k in RS: E[k]=E_res(k,gc,rho,g,K)
    return E
def win(gc,rho):
    E=allE(gc,rho); k=min(E,key=E.get); return k,E
pts=[]; rows={}
t=time.time()
if mode=='rho':
    for gc in CC[i0:i1]:
        W=[win(gc,r) for r in RHO]; rows[str(round(gc,4))]=[w[0] for w in W]
        for j in range(len(RHO)-1):
            a,b=W[j][0],W[j+1][0]
            if a!=b:
                f=lambda r: allE(gc,r)[a]-allE(gc,r)[b]
                try: pts.append((brentq(f,RHO[j],RHO[j+1],xtol=1e-6),gc,a,b))
                except ValueError: pass
else:
    for rho in RHO[i0:i1]:
        W=[win(c,rho) for c in CC]
        for j in range(len(CC)-1):
            a,b=W[j][0],W[j+1][0]
            if a!=b:
                f=lambda c: allE(c,rho)[a]-allE(c,rho)[b]
                try: pts.append((rho,brentq(f,CC[j],CC[j+1],xtol=1e-6),a,b))
                except ValueError: pass
json.dump({'pts':pts,'rows':rows},open(f'tg_{int(K)}_{g}_{mode}_{i0}.json','w'))
print(K,mode,i0,i1,len(pts),round(time.time()-t,1))
