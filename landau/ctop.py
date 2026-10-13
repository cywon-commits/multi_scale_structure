import sys, json, numpy as np
from cands import *
from tongue import E_nonres_opt
S=build2(); comp=['HEX(1,√3,2)','APXs','HEX(1,2)']
def margin(c,g,K):
    sv=np.sqrt(1-c*c); x=(R12-1)/(R12+1); m=sv/x
    q=S['QC12s'].Emin(c,m,g,K,(m-sv)/S['QC12s'].mag[0])
    E={k:S[k].Emin(c,m,g,K,(m-sv)/S[k].mag[0]) for k in comp}; E['inc']=E_nonres_opt(c,g,K)[1]
    w=min(E,key=E.get); return q-E[w],w
g=float(sys.argv[1]); K=float(sys.argv[2]); lo,hi=float(sys.argv[3]),float(sys.argv[4]); n=int(sys.argv[5])
for _ in range(n):
    mid=0.5*(lo+hi); mg,w=margin(mid,g,K)
    print('K=%g c=%.4f margin=%.4f (%s)'%(K,mid,mg,w),flush=True)
    if mg<0: lo=mid
    else: hi=mid
print('RESULT K=%g c_top in [%.4f, %.4f]'%(K,lo,hi))
