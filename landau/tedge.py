import sys, json, time, numpy as np
from scipy.optimize import brentq, minimize_scalar
from tongue import *
K=float(sys.argv[1]); g=float(sys.argv[2]); job=sys.argv[3]; i0,i1=int(sys.argv[4]),int(sys.argv[5])
RLO,RHI=1.05,8.0
CACHE={}
def Einc(gc):
    k=round(gc,12)
    if k not in CACHE: CACHE[k]=E_nonres_opt(gc,g,K)[1]
    return CACHE[k]
def Ehex(gc,r): return min(E_res('sqrt3',gc,r,g,K),E_res('two',gc,r,g,K))
def EQ(gc,r): return E_res('qc12',gc,r,g,K)
def edges(D,center,lo,hi):
    """D(rho)<0 inside tongue; find peak near center and both edges"""
    a,b=max(lo,center*0.75),min(hi,center*1.3)
    xs=np.linspace(a,b,13); v=[D(x) for x in xs]; i=int(np.argmin(v))
    r=minimize_scalar(D,bounds=(xs[max(i-1,0)],xs[min(i+1,12)]),method='bounded',options={'xatol':1e-7})
    pk,dv=(r.x,r.fun) if r.fun<v[i] else (xs[i],v[i])
    if dv>=0: return None
    out=[]
    for sgn in (-1,1):
        x0=pk; step=0.01*pk; x1=x0
        while True:
            x1=x0+sgn*step
            if x1<lo or x1>hi: out.append(None); break
            if D(x1)>0: out.append(brentq(D,min(x0,x1),max(x0,x1),xtol=1e-7)); break
            x0=x1; step*=1.6
    return pk,out
res=[]
t=time.time()
if job=='rows':
    CC=np.linspace(0.04,0.98,48)
    for gc in CC[i0:i1]:
        row={'c':gc}
        for k in ['sqrt3','two','qc12']:
            rs=rho_star(k,gc); x=(RS[k]-1)/(RS[k]+1); s=np.sqrt(1-gc**2)
            center=rs if rs else (s+x)/max(s-x,1e-3)
            center=min(center,RHI*0.9)
            Ek=(lambda r,k=k: E_res(k,gc,r,g,K))
            row[k]=edges(lambda r:Ek(r)-Einc(gc),center,RLO,RHI)
        # QC vs hex inside QC tongue
        if row['qc12'] and row['qc12'][1][0] and row['qc12'][1][1]:
            L,R=row['qc12'][1]; xs=np.linspace(L,R,15)
            d=[EQ(gc,x)-Ehex(gc,x) for x in xs]
            cuts=[brentq(lambda x:EQ(gc,x)-Ehex(gc,x),xs[j],xs[j+1],xtol=1e-7) for j in range(14) if d[j]*d[j+1]<0]
            row['qc_hex']={'cuts':cuts,'signs':[bool(v<0) for v in d]}
        res.append(row)
else:   # cols: c where hex region lower edge crosses inc, at fixed rho
    RH=np.exp(np.linspace(np.log(RLO),np.log(RHI),30))
    for r in RH[i0:i1]:
        cs=np.linspace(0.55,0.95,17); d=[Ehex(c,r)-Einc(c) for c in cs]
        cuts=[brentq(lambda c:Ehex(c,r)-Einc(c),cs[j],cs[j+1],xtol=1e-7) for j in range(16) if d[j]*d[j+1]<0]
        res.append({'rho':r,'cuts':cuts})
json.dump(res,open(f'te_{int(K)}_{g}_{job}_{i0}.json','w'),default=float)
print(job,i0,i1,round(time.time()-t,1))
