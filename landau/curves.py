import json, numpy as np, collections
from fastE import FastPanel
out={}
for n,R in [('sqrt3',np.sqrt(3)),('two',2.0),('qc12',2*np.cos(np.pi/12))]:
    P=json.load(open(f'bd_{n}_g_0.json'))+json.load(open(f'bd_{n}_c_0.json'))
    groups=collections.defaultdict(list)
    for g,c,a,b in P: groups[tuple(sorted((a,b)))].append((g,c))
    curves=[]
    for pair,pts in groups.items():
        pts=np.array(sorted(set((round(g,5),round(c,5)) for g,c in pts)))
        if len(pts)<3: continue
        Z=pts/np.array([6,1.0]); left=list(range(len(Z)))
        # chain: start at point with extreme coordinate
        while left:
            i=min(left,key=lambda j:(Z[j,0]+Z[j,1])); chain=[i]; left.remove(i)
            while left:
                d=[np.hypot(*(Z[j]-Z[chain[-1]])) for j in left]; k=int(np.argmin(d))
                if d[k]>0.06: break
                chain.append(left.pop(k))
            # try extend from the other end too
            while left:
                d=[np.hypot(*(Z[j]-Z[chain[0]])) for j in left]; k=int(np.argmin(d))
                if d[k]>0.06: break
                chain.insert(0,left.pop(k))
            if len(chain)>=3:
                c_=pts[chain]; step=max(1,len(c_)//120)
                c_=np.vstack([c_[::step],c_[-1:]])
                curves.append({'pair':pair,'xy':np.round(c_,3).tolist()})
    # label positions: coarse winner sampling (placement only)
    p=FastPanel(R); lab=collections.defaultdict(list)
    for c in np.linspace(0.03,0.99,33):
        s=np.sqrt(1-c*c)
        for g in np.linspace(0.05,5.95,41):
            lab[p.win(s,g)[0]].append((g,c))
    labels={k:np.round(np.median(np.array(v),axis=0),3).tolist()+[len(v)] for k,v in lab.items()}
    out[n]={'curves':curves,'labels':labels}
    print(n,[(c['pair'],len(c['xy'])) for c in curves],labels)
json.dump(out,open('curves.json','w'))
