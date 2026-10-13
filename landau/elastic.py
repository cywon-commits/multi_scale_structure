import numpy as np
from phason import *
def elastic(a):
    """E_el = (K s^2/2) sum_j a_j^2 (khat_j . grad theta_j)^2, theta_j = k_j.u + kperp_j.w
       basis: [d_a u_b (ab=xx,xy,yx,yy), d_a w_b (xx,xy,yx,yy)]"""
    M=np.zeros((8,8))
    for k,kp,aj in zip(KV,KP,a):
        kh=k/np.linalg.norm(k)
        v=np.array([kh[al]*k[be] for al in range(2) for be in range(2)]+[kh[al]*kp[be] for al in range(2) for be in range(2)])
        M+=aj**2*np.outer(v,v)
    return M
if __name__=='__main__':
    for lab,x in [('6-fold QC',np.load('qc12s_x.npy')),('12-fold sym',np.repeat(np.load('qc12sym_x.npy'),2))]:
        a=orbit_amps(x); M=elastic(a); w=np.linalg.eigvalsh(M)
        Cuw=M[:4,4:]
        # isotropic phonon constants: eps_xx eps_yy shear
        uxx=np.array([1,0,0,0.]); uyy=np.array([0,0,0,1.]); sh=np.array([0,1,1,0])/2; rot=np.array([0,1,-1,0])/2
        lam_2mu=uxx@M[:4,:4]@uxx; lam=uxx@M[:4,:4]@uyy; mu=sh@M[:4,:4]@sh/ (2*0.5) 
        print(f'--- {lab}: eigenvalues of 8x8 elastic matrix (units K s^2):',np.round(w,4))
        print('   rotation mode energy',round(rot@M[:4,:4]@rot,6),'  lambda+2mu=%.4f lambda=%.4f'%(lam_2mu,lam))
        print('   phonon-phason coupling block norm',round(np.linalg.norm(Cuw),5))
        dil=np.array([1,0,0,1.])/np.sqrt(2)
        print('   dilation-phason coupling vector',np.round(dil@Cuw,5))
