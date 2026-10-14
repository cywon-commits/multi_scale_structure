"""Chirality of the two-scale quasicrystal (theory notes sec. 21).
Order parameters: eta1 = a_A - a_B (Q1 ring), eta2 = a_C - a_D (Q2 ring), chi = sign(eta1*eta2).
Equivalent S(q) reading: the major Q2 orbit is rotated by +15 deg (chi>0) or -15 deg (chi<0) from the major Q1 orbit.

  python chirality.py predict g [c_min c_max dc]     # K=inf Landau orbit amplitudes, eta1, eta2 vs c
  python chirality.py measure peaks.json             # peaks: [{"n":[nx,ny],"a":amp}, ...] -> rings, orbits, eta, chi
"""
import sys, json, numpy as np
from polystate import Poly
from cands import build2
def predict(g,cs):
    qc=Poly(build2()['QC12s'],[0,0,1,1]); out=[]; x=None
    for c in cs:
        qc.setc(c); st=[np.array([-.6,-.15,-.6,-.15]),np.array([-.45,-.4,-.45,-.4])]+([x] if x is not None else [])
        e,x=qc.emin(g,st); a=np.abs(x); out.append(dict(c=float(c),F=float(e),A=a[0],B=a[1],C=a[2],D=a[3],eta1=a[0]-a[1],eta2=a[2]-a[3]))
    return out
def measure(peaks,ratio=2*np.cos(np.pi/12),tol=0.06):
    """peaks: list of dicts with integer box vector 'n' and mode amplitude 'a' (one entry per +/- pair or both; duplicates merged)"""
    P={}
    for p in peaks:
        n=np.array(p['n'],float); key=tuple(n) if (n[0]>0 or (n[0]==0 and n[1]>0)) else tuple(-n)
        P[key]=max(P.get(key,0),p['a'])
    k=np.array([np.hypot(*v) for v in P]); amp=np.array(list(P.values())); vec=np.array(list(P.keys()))
    k1=np.median(k[amp>0.5*amp.max()][np.argsort(k[amp>0.5*amp.max()])[:3]])   # Q1 radius from the strongest small-|k| peaks
    res={}
    for name,kr in [('Q1',k1),('Q2',ratio*k1)]:
        m=np.abs(k/kr-1)<tol; ang=np.degrees(np.arctan2(vec[m,1],vec[m,0]))%180; a=amp[m]
        if m.sum()<4: res[name]=None; continue
        big=ang[np.argsort(a)[::-1][:3]]; ph=np.degrees(np.angle(np.mean(np.exp(1j*np.radians(big)*6))))/6   # orbit angle mod 60
        mod=lambda x:((x-ph+30)%60)-30
        maj=np.abs([mod(t) for t in ang])<15
        res[name]=dict(orbit_angle_mod60=float(ph%60),major_mean=float(a[maj].mean()),minor_mean=float(a[~maj].mean()) if (~maj).any() else 0.0,
                       eta=float(a[maj].mean()-(a[~maj].mean() if (~maj).any() else 0.0)),n_peaks=int(m.sum()))
    if res.get('Q1') and res.get('Q2'):
        off=((res['Q2']['orbit_angle_mod60']-res['Q1']['orbit_angle_mod60']+30)%60)-30
        res['offset_deg']=float(off); res['chi']=int(np.sign(off)) if abs(abs(off)-15)<7 else 0
    return res
if __name__=='__main__':
    if sys.argv[1]=='predict':
        g=float(sys.argv[2]); lo,hi,dc=(map(float,sys.argv[3:6]) if len(sys.argv)>3 else (0.60,0.86,0.02))
        for r in predict(g,np.arange(lo,hi+1e-9,dc)): print('c=%.3f  A=%.3f B=%.3f C=%.3f D=%.3f  eta1=%.3f eta2=%.3f  F=%.4f'%(r['c'],r['A'],r['B'],r['C'],r['D'],r['eta1'],r['eta2'],r['F']))
    else:
        print(json.dumps(measure(json.load(open(sys.argv[2]))),indent=1))
