"""Reference soft-spin GL relaxation for the DMI bilayer (pseudo-spectral, semi-implicit).
   dm/dt = -dF/dm ; linear part (exchange, DMI, interlayer, r) implicit per q via a 6x6 solve, nonlinear explicit.
   Intended as a small, verified starting point; production runs (L>=512, long times, GPU) belong in Claude Code."""
import numpy as np
from params import landau_to_model, kernel
class Bilayer:
    def __init__(s,p,L=128,dt=0.5,seed=0):
        s.p=p; s.L=L; s.dt=dt
        k=2*np.pi*np.fft.fftfreq(L); s.qx,s.qy=np.meshgrid(k,k,indexing='ij')
        Lq=np.zeros((L,L,6,6),complex)
        for a in range(L):
            for b in range(L): Lq[a,b]=kernel(p,s.qx[a,b],s.qy[a,b])
        s.Lq=Lq; s.Ainv=np.linalg.inv(np.eye(6)[None,None]+2*dt*Lq)
        rng=np.random.default_rng(seed); s.m=1e-3*rng.normal(size=(6,L,L))
        s.m[2]+=p['M_target']; s.m[5]+=p['M_target']
    def nonlin(s,m):
        g=np.zeros_like(m)
        for o,H in ((0,s.p['H1']),(3,s.p['H2'])):
            m2=(m[o:o+3]**2).sum(0); g[o:o+3]=4*s.p['u']*m2*m[o:o+3]
            g[o+2]-=H
        return g
    def step(s,n=1):
        for _ in range(n):
            mh=np.fft.fft2(s.m,axes=(1,2)); nh=np.fft.fft2(s.nonlin(s.m),axes=(1,2))
            rhs=np.moveaxis(mh-s.dt*nh,0,-1)
            s.m=np.real(np.fft.ifft2(np.moveaxis(np.einsum('abij,abj->abi',s.Ainv,rhs),-1,0),axes=(1,2)))
    def energy(s):
        mh=np.moveaxis(np.fft.fft2(s.m,axes=(1,2)),0,-1); N=s.L*s.L
        e2=np.real(np.einsum('abi,abij,abj->',mh.conj(),s.Lq,mh))/N
        e4=sum(s.p['u']*((s.m[o:o+3]**2).sum(0)**2).sum()-H*s.m[o+2].sum() for o,H in ((0,s.p['H1']),(3,s.p['H2'])))
        return (e2+e4)/N
    def structure(s,layer):
        o=3*layer; return sum(np.abs(np.fft.fft2(s.m[o+i]-s.m[o+i].mean()))**2 for i in range(3))
if __name__=='__main__':
    # smoke test: small box, soft band (K=5) so it relaxes quickly; q1 = 2pi/16 gives 4 periods
    p=landau_to_model(0.3,2.0,5.0,4.0,q1=2*np.pi/16)
    sim=Bilayer(p,L=64,dt=2.0)
    E=[sim.energy()]
    for _ in range(6): sim.step(400); E.append(sim.energy())
    print('energy trace:',np.round(np.array(E)/p['r_abs']**2,3))
    print('monotone decreasing:',all(np.diff(E)<=1e-12))
    print('<m_z> layer1,2 =',round(sim.m[2].mean(),5),round(sim.m[5].mean(),5),' target M =',round(p['M_target'],5))
    S1=sim.structure(0); k=np.hypot(sim.qx,sim.qy); i=np.unravel_index(np.argmax(S1*(k>0)),S1.shape)
    print('layer-1 S(q) peak |q| =',round(k[i],4),' (band minima q-,q+ =',round(p['q_minus'],4),round(p['q_plus'],4),')')
