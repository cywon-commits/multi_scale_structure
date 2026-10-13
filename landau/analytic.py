import numpy as np
from scipy.optimize import brentq, minimize_scalar
def phi(t):                       # min_x (-x^2 - t x^3 + x^4)
    x=(3*t+np.sqrt(9*t*t+32))/8; return -x*x-t*x**3+x**4
def Fstar(n,A,B): return n*n/B*phi(A/np.sqrt(n*B))
def LF(c):
    s=np.sqrt(1-c*c); w1=np.array([(1+s)/2,(1-s)/2]); w2=w1[::-1]
    return dict(s=s,P2=(w1**2).sum(),P32=(w1**1.5).sum(),X=(w1*w2).sum(),
                M=(w1*np.sqrt(w2)).sum(),Y=(w1**1.5*w2**0.5).sum())
# exact geometric coefficients (single layer, from Fourier enumeration)
T=lambda phi_: 3-np.cos(phi_)+2*np.cos(phi_/2)
a12=6*T(np.radians(30))/2; a21=6*T(np.radians(150))/2      # 12.1975, 13.151
QC4=dict(A4=57.0,A3B=57.8957,A2B2=194.6969,AB3=70.1951)
def F_SkX(c,g):  L=LF(c); return Fstar(3,2.25*L['P32']*g,12.75*L['P2'])
def F_SS(c,g):   L=LF(c); return Fstar(6,4.5*L['P32']*g,25.5*L['P2']+31.5*L['X'])
def F_QC(c,g):   # symmetric ring amplitudes a1=a2
    L=LF(c)
    A=g*(9.0*L['P32']+(a12+a21)*L['M'])
    B=2*QC4['A4']*L['P2']+(QC4['A3B']+QC4['AB3'])*L['Y']+QC4['A2B2']*L['X']
    return Fstar(12,A,B)
def m_c(c,g): return F_QC(c,g)-min(F_SkX(c,g),F_SS(c,g))
def m_inf(g):
    cs=np.linspace(0.45,0.97,53); v=[m_c(c,g) for c in cs]; i=int(np.argmin(v))
    r=minimize_scalar(lambda c:m_c(c,g),bounds=(cs[max(i-1,0)],cs[min(i+1,52)]),method='bounded',options={'xatol':1e-8})
    return r.fun,r.x
def c_cross(g):  # SkX = SkX+SkX line
    return brentq(lambda c:F_SkX(c,g)-F_SS(c,g),0.3,0.99)

# ---------- finite-K corrections (second-order perturbation, closed forms) ----------
def wv(xi,c):
    t=xi/np.sqrt(xi*xi+c*c); return np.array([(1-t)/2,(1+t)/2])
def fb(xi,c): return 0.5*(xi*xi+1)+c-np.sqrt(xi*xi+c*c)
def lf(p,q,w1,w2): return (w1**(p/2)*w2**(q/2)).sum()
def xstar(n,A,B):
    t=A/np.sqrt(n*B); return np.sqrt(n/B)*(3*t+np.sqrt(9*t*t+32))/8
def skx(w,g):     return 3,2.25*lf(3,0,w,w)*g,12.75*lf(4,0,w,w)
def ss(w1,w2,g):  return 6,2.25*(lf(3,0,w1,w1)+lf(3,0,w2,w2))*g,12.75*(lf(4,0,w1,w1)+lf(4,0,w2,w2))+31.5*lf(2,2,w1,w2)
def qc(w1,w2,g):
    A=g*(4.5*(lf(3,0,w1,w1)+lf(3,0,w2,w2))+a12*lf(2,1,w1,w2)+a21*lf(1,2,w1,w2))
    B=QC4['A4']*(lf(4,0,w1,w1)+lf(4,0,w2,w2))+QC4['A3B']*lf(3,1,w1,w2)+QC4['AB3']*lf(1,3,w1,w2)+QC4['A2B2']*lf(2,2,w1,w2)
    return 12,A,B
SHELL={'sqrt3':dict(R=np.sqrt(3),n=3,g3=6.3481,q31=16.7942,q22=47.1962),
       'two':  dict(R=2.0,       n=3,g3=3.0,   q31=13.5,   q22=38.25)}
def corrections(c,g):
    """returns K=inf energies and the pieces needed for finite-K energies at coupling c"""
    s=np.sqrt(1-c*c); xR=(2*np.cos(np.pi/12)-1)/(2*np.cos(np.pi/12)+1); m=s/xR; R=2*np.cos(np.pi/12)
    W=lambda x: wv(x,c); h=1e-6
    out={}
    # SkX (ring at xi=-s), position-shift gradient
    Fs=lambda x: Fstar(*skx(W(x),g)); GS=(Fs(-s+h)-Fs(-s-h))/(2*h)
    n,A,B=skx(W(-s),g); aS=-xstar(n,A,B)            # a<0 so that cubic term is a gain
    out['S']=(Fs(-s),GS**2/(2*s*s*3*aS*aS),aS)
    # shells of the SkX ring
    sh={}
    for k,d in SHELL.items():
        xsh=d['R']*(m-s)-m; wsh=W(xsh); w1=W(-s)
        hdrive=d['g3']*g*lf(2,1,w1,wsh)*aS**2+d['q31']*lf(3,1,w1,wsh)*aS**3
        sh[k]=(fb(xsh,c)-fb(s,c),hdrive,d['q22']*lf(2,2,w1,wsh)*aS**2,d['n'])
    out['shell']=sh
    # SkX+SkX (rings at -s, +s, independent shifts)
    F2=lambda x1,x2: Fstar(*ss(W(x1),W(x2),g))
    G1=(F2(-s+h,s)-F2(-s-h,s))/(2*h); G2=(F2(-s,s+h)-F2(-s,s-h))/(2*h)
    n,A,B=ss(W(-s),W(s),g); a=xstar(n,A,B)
    out['SS']=(F2(-s,s),(G1**2+G2**2)/(2*s*s*3*a*a))
    # QC (rings tied by ratio: dxi2 = R dxi1)
    Fq=lambda x1,x2: Fstar(*qc(W(x1),W(x2),g))
    Q1=(Fq(-s+h,s)-Fq(-s-h,s))/(2*h); Q2=(Fq(-s,s+h)-Fq(-s,s-h))/(2*h)
    n,A,B=qc(W(-s),W(s),g); a=xstar(n,A,B)
    out['QC']=(Fq(-s,s),(Q1+R*Q2)**2/(2*s*s*6*a*a*(1+R*R)))
    return out
def energies(c,g,K,co=None):
    co=co or corrections(c,g)
    FS,cS,aS=co['S']; ES=FS-cS/K
    best=0.0
    for k,(df,hd,kap,n) in co['shell'].items():
        den=n*(-1+K*df)+kap
        if den>0: best=max(best,hd*hd/(4*den))
        else: best=np.inf
    Ehex=ES-best
    Einc=co['SS'][0]-co['SS'][1]/K
    EQ=co['QC'][0]-co['QC'][1]/K
    return EQ,Ehex,Einc
def Kc_at_c(c,g):
    co=corrections(c,g)
    f=lambda lK: (lambda e: e[0]-min(e[1],e[2]))(energies(c,g,np.exp(lK),co))
    if f(np.log(1e6))>=0: return np.inf
    lo=np.log(5.0)
    if f(lo)<0: return np.exp(lo)
    return np.exp(brentq(f,lo,np.log(1e6),xtol=1e-6))
def Kc(g):
    cs=np.linspace(0.45,0.95,101); v=[Kc_at_c(c,g) for c in cs]; i=int(np.argmin(v))
    r=minimize_scalar(lambda c:Kc_at_c(c,g),bounds=(cs[max(i-1,0)],cs[min(i+1,100)]),method='bounded',options={'xatol':1e-5})
    return (r.fun,r.x) if r.fun<v[i] else (v[i],cs[i])

def hex_energy(c,g,K,shells=('sqrt3','two'),pert=True,m=None):
    """SkX ring at xi1=-s+u plus one induced shell (ratio-tied), optimized over u.
       pert=True: shell treated by 2nd-order perturbation (closed form);
       pert=False: exact 2-amplitude minimum (polar formula)"""
    s=np.sqrt(1-c*c); R=2*np.cos(np.pi/12); xR=(R-1)/(R+1); m=s/xR if m is None else m
    W=lambda x: wv(x,c); fs=fb(s,c)
    best=np.inf
    for k in shells:
        d=SHELL[k]
        def E(u):
            x1=-s+u; xsh=d['R']*(m+x1)-m; w1=W(x1); wsh=W(xsh)
            r1=-1+K*(fb(x1,c)-fs); rsh=-1+K*(fb(xsh,c)-fs)
            if r1>=0: return 0.0
            n,A,B=skx(w1,g); n=-3*r1
            F0=Fstar(n,A,B); a=-xstar(n,A,B)
            if pert:
                hd=d['g3']*g*lf(2,1,w1,wsh)*a**2+d['q31']*lf(3,1,w1,wsh)*a**3
                den=3*rsh+d['q22']*lf(2,2,w1,wsh)*a**2
                return F0-hd*hd/(4*den) if den>0 else -np.inf
            # exact: polar minimisation over (a,b)
            th=np.linspace(0,2*np.pi,2001)
            cth,sth=np.cos(th),np.sin(th)
            nn=-(3*r1*cth**2+3*rsh*sth**2)
            c3=g*(2.25*lf(3,0,w1,w1)*cth**3+d['g3']*lf(2,1,w1,wsh)*cth**2*sth+2.25*lf(3,0,wsh,wsh)*sth**3)
            c4=12.75*lf(4,0,w1,w1)*cth**4+d['q31']*lf(3,1,w1,wsh)*cth**3*sth+d['q22']*lf(2,2,w1,wsh)*cth**2*sth**2+12.75*lf(4,0,wsh,wsh)*sth**4
            disc=9*c3**2+32*nn*c4; rho=np.where(disc>0,(-3*c3+np.sqrt(np.maximum(disc,0)))/(8*c4),0); rho=np.maximum(rho,0)
            return float(np.min(np.minimum(-nn*rho**2+c3*rho**3+c4*rho**4,0)))
        us=np.linspace(-0.3,0.3,61); v=np.array([E(u) for u in us])
        if np.isneginf(v).any(): best=min(best,-np.inf); continue
        i=int(np.argmin(v)); r=minimize_scalar(E,bounds=(us[max(i-1,0)],us[min(i+1,60)]),method='bounded',options={'xatol':1e-9})
        best=min(best,r.fun,v[i])
    return best

def dfs(c):
    s=np.sqrt(1-c*c); R=2*np.cos(np.pi/12); xR=(R-1)/(R+1); m=s/xR
    return {k:fb(d['R']*(m-s)-m,c)-fb(s,c) for k,d in SHELL.items()}
def Kc_exact_hex(g,cs):
    res=[]
    for c in cs:
        co=corrections(c,g)
        def f(lK):
            K=np.exp(lK); EQ=co['QC'][0]-co['QC'][1]/K; Ei=co['SS'][0]-co['SS'][1]/K
            return EQ-min(Ei,hex_energy(c,g,K,pert=False))
        try:
            k=np.exp(brentq(f,np.log(20),np.log(5e4),xtol=1e-4)) if f(np.log(5e4))<0 and f(np.log(20))>0 else np.inf
        except Exception: k=np.inf
        res.append(k)
    i=int(np.argmin(res)); return res[i],cs[i]

def Kc_c_exact(c,g):
    co=corrections(c,g)
    def f(lK):
        K=np.exp(lK); EQ=co['QC'][0]-co['QC'][1]/K; Ei=co['SS'][0]-co['SS'][1]/K
        return EQ-min(Ei,hex_energy(c,g,K,pert=False))
    if f(np.log(5e4))>=0: return np.inf
    if f(np.log(20))<0: return 20.0
    return np.exp(brentq(f,np.log(20),np.log(5e4),xtol=1e-5))
def Kc_semi(g):
    cs=np.linspace(0.5,0.85,15); v=[Kc_c_exact(c,g) for c in cs]; i=int(np.argmin(v))
    r=minimize_scalar(lambda c:Kc_c_exact(c,g),bounds=(cs[max(i-1,0)],cs[min(i+1,14)]),method='bounded',options={'xatol':1e-4})
    return (r.fun,r.x) if r.fun<v[i] else (v[i],cs[i])

def margin_cm(c,g,K,co=None):
    """margin optimised over the D-ratio (m) around the QC centre; QC detuning harmonic (tongue formula)"""
    co=co or corrections(c,g); s=np.sqrt(1-c*c); R=2*np.cos(np.pi/12); xR=(R-1)/(R+1); m0=s/xR
    n,A,B=qc(wv(-s,c),wv(s,c),g); a=xstar(n,A,B); N1=N2=6*a*a          # 6 modes per ring, amplitude a each
    EQ0=co['QC'][0]-co['QC'][1]/K; Ei=co['SS'][0]-co['SS'][1]/K
    stiff=0.5*K*s*s*(R-1)**2/(1/N2+R*R/N1)
    D=lambda dm: (EQ0+stiff*dm*dm)-min(Ei,hex_energy(c,g,K,pert=False,m=m0+dm))
    ds=np.linspace(-0.08,0.08,17)*m0; v=[D(x) for x in ds]; i=int(np.argmin(v))
    r=minimize_scalar(D,bounds=(ds[max(i-1,0)],ds[min(i+1,16)]),method='bounded',options={'xatol':1e-6*m0})
    return min(r.fun,v[i]), (r.x if r.fun<v[i] else ds[i])/m0

def Kc_c_m(c,g,shells=('two',)):
    co=corrections(c,g)
    import analytic as A_
    def f(lK):
        K=np.exp(lK); s=np.sqrt(1-c*c); R=2*np.cos(np.pi/12); xR=(R-1)/(R+1); m0=s/xR
        n,A,B=qc(wv(-s,c),wv(s,c),g); a=xstar(n,A,B); N=6*a*a
        EQ0=co['QC'][0]-co['QC'][1]/K; Ei=co['SS'][0]-co['SS'][1]/K
        stiff=0.5*K*s*s*(R-1)**2/((1+R*R)/N)
        D=lambda dm:(EQ0+stiff*dm*dm)-min(Ei,hex_energy(c,g,K,shells=shells,pert=False,m=m0+dm))
        r=minimize_scalar(D,bounds=(-0.06*m0,0.06*m0),method='bounded',options={'xatol':1e-5*m0})
        return min(r.fun,D(0.0))
    if f(np.log(2e4))>=0: return np.inf
    if f(np.log(20))<0: return 20.0
    return np.exp(brentq(f,np.log(20),np.log(2e4),xtol=2e-3))
