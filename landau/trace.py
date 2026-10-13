import sys, json, time, numpy as np
from scipy.optimize import brentq
from fastE import FastPanel as Panel
nm=sys.argv[1]; R={'sqrt3':np.sqrt(3),'two':2.0,'qc12':2*np.cos(np.pi/12)}[nm]
p=Panel(R); S=lambda c: np.sqrt(1-c*c)
cmax=0.995
pts=[]
def bracket_roots(f_win, f_E, xs, other, axis):
    W=[f_win(x)[0] for x in xs]
    for x0,x1,a,b in zip(xs[:-1],xs[1:],W[:-1],W[1:]):
        if a==b: continue
        d=lambda x: f_E(a,x)-f_E(b,x)
        try:
            r=brentq(d,x0,x1,xtol=1e-9)
            pts.append((r,other,a,b) if axis=='g' else (other,r,a,b))
        except ValueError: pass
t=time.time()
mode=sys.argv[2]
if mode=='g':   # fixed c, roots in g
    for c in np.linspace(0.02,cmax,int(sys.argv[3]))[int(sys.argv[4])::int(sys.argv[5])]:
        s=S(c)
        bracket_roots(lambda g:p.win(s,g), lambda k,g:p.E(k,s,g), np.linspace(0,6,25), c,'g')
else:           # fixed g, roots in c
    for g in np.linspace(0,6,int(sys.argv[3]))[int(sys.argv[4])::int(sys.argv[5])]:
        bracket_roots(lambda c:p.win(S(c),g), lambda k,c:p.E(k,S(c),g), np.linspace(0.02,cmax,25), g,'c')
fn=f'bd_{nm}_{mode}_{sys.argv[4]}.json'; json.dump(pts,open(fn,'w')); print(nm,mode,len(pts),round(time.time()-t,1))
