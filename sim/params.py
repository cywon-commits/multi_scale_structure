"""Map Landau-theory variables (c/c_L, rho=D2/D1, K, g) to real-space soft-spin GL bilayer parameters.

Real-space functional (per layer i=1,2, summed over grid sites, lattice spacing h=1):
    f = sum_i [ r_i m_i^2 + A |grad m_i|^2 + 2 D_i m_iz (div m_i,parallel) + u (m_i^2)^2 - H_i m_iz ]
        + c |m_1 - m_2|^2
(2 D m_z div m is the integrated-by-parts form of interfacial (Neel) DMI  D[m_z div m - (m.grad) m_z].)
With these conventions a single cycloid of amplitude a costs (r_i + A q^2 - D_i q) a^2, so q_i = D_i/(2A),
and the 2x2 interlayer kernel is [[eps_1+c, -c], [-c, eps_2+c]] exactly as in the theory notes (sec. 7).
"""
import numpy as np
def landau_to_model(c_rel, rho, K, g, q1=2*np.pi/32, A=1.0, u=1.0):
    q2=rho*q1; D1,D2=2*A*q1,2*A*q2
    delta=(q2-q1)/2; cL=2*A*delta**2; c=c_rel*cL; r_abs=cL/K
    M=g*np.sqrt(u*r_abs)/(4*u)                    # uniform m_z that realises the cubic coupling g = 4uM/sqrt(u|r|)
    # band minimum of the lower interlayer band = e + c - c^2/(2 c_L) (+4uM^2 cycloid shift from the uniform moment)
    e=-r_abs-c+c*c/(2*cL)-4*u*M*M
    r1,r2=e+A*q1**2,e+A*q2**2                       # Delta=0: equal minima energies
    # layer-resolved fields so that both layers carry the same uniform moment M (r1 != r2 because Delta=0)
    H1,H2=2*r1*M+4*u*M**3,2*r2*M+4*u*M**3          # first guess; calibrate on measured <m_z> (modulations shift it)
    s=np.sqrt(max(1-c_rel**2,0)); qbar=0.5*(q1+q2)
    return dict(A=A,u=u,D1=D1,D2=D2,r1=r1,r2=r2,c=c,H1=H1,H2=H2,M_target=M,
                q1=q1,q2=q2,cL=cL,r_abs=r_abs,q_minus=qbar-delta*s,q_plus=qbar+delta*s,
                landau=dict(c_rel=c_rel,rho=rho,K=K,g=g))
def kernel(p,qx,qy):
    """6x6 Hermitian quadratic kernel L_q (layer1 xyz, layer2 xyz) of the real-space functional"""
    L=np.zeros((6,6),complex)
    q2=qx*qx+qy*qy
    for i,(r,D) in enumerate([(p['r1'],p['D1']),(p['r2'],p['D2'])]):
        o=3*i; L[o:o+3,o:o+3]+=np.eye(3)*(r+p['A']*q2+p['c'])
        L[o+2,o+0]+=1j*D*qx; L[o+0,o+2]+=-1j*D*qx
        L[o+2,o+1]+=1j*D*qy; L[o+1,o+2]+=-1j*D*qy
    L[0:3,3:6]+=-p['c']*np.eye(3); L[3:6,0:3]+=-p['c']*np.eye(3)
    return L
if __name__=='__main__':
    p=landau_to_model(0.3,2.0,150,0.0)
    qs=np.linspace(0.01,0.6,6000); lam=np.array([np.linalg.eigvalsh(kernel(p,q,0))[0] for q in qs])
    i=[k for k in range(1,len(qs)-1) if lam[k]<lam[k-1] and lam[k]<lam[k+1]]
    print('band minima found at q =',np.round(qs[i],4),' predicted q-,q+ =',round(p['q_minus'],4),round(p['q_plus'],4))
    print('band minimum values    =',np.round(lam[i],7),' target -|r| =',-round(p['r_abs'],7))
