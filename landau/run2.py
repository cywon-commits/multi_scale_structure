from scan import *
import json
ORDER=['PM','1q','SkX','2x1q-par','2x1q-perp','SkX+SkX','RES']
Rs={'sqrt3':np.sqrt(3),'two':2.0,'qc12':2*np.cos(np.pi/12)}
S=np.round(np.linspace(0.05,0.95,10),3); Gs=np.round(np.linspace(0,3,13),2)
out={}
for nm,R in Rs.items():
    G=geoms(R); C={n:polycoef(*G[n]) for n in G}
    tab={}; print('==',nm)
    for s in S:
        row=[]
        for g in Gs:
            E={n:Fmin(*G[n],s,g,C[n]) for n in G}; E['PM']=0.0
            emin=min(E.values())
            win=[n for n in ORDER if n in E and E[n]<emin+1e-6][0]   # simplest within tol
            row.append(win)
        tab[float(s)]=row
        print('s=%.2f'%s,' '.join('%-9s'%w for w in row))
    out[nm]=tab
json.dump({'S':S.tolist(),'G':Gs.tolist(),'out':out},open('phase.json','w'))
