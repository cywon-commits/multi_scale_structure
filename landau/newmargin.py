import sys, json, time, numpy as np
from cands import *
from tongue import E_nonres_opt
S=build2()
g=float(sys.argv[1]); Ks=[float(k) for k in sys.argv[2].split(',')]; cs=[float(c) for c in sys.argv[3].split(',')]
comp=sys.argv[4].split(';')
out=[]
for K in Ks:
    for c in cs:
        sv=np.sqrt(1-c*c); x=(R12-1)/(R12+1); m=sv/x
        E={k:S[k].Emin(c,m,g,K,(m-sv)/S[k].mag[0]) for k in ['QC12s']+comp}
        E['inc']=E_nonres_opt(c,g,K)[1]
        q=E.pop('QC12s'); w=min(E,key=E.get)
        out.append((K,c,q,w,E[w],q-E[w])); print('K=%g c=%.2f QC12s=%.4f best other %s=%.4f margin=%.4f'%(K,c,q,w,E[w],q-E[w]),flush=True)
json.dump(out,open(f'nm_{g}_{sys.argv[2]}_{sys.argv[3]}.json','w'))
