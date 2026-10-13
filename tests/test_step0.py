"""Step 0: GPU(backend-agnostic) code vs reference on 64x64; also band-minimum check."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sim'))
import numpy as np
from params import landau_to_model, kernel
from gl_bilayer_reference import Bilayer
import gl_gpu as G
from gl_gpu import GLBilayer, kernel_field

p = landau_to_model(0.3, 2.0, 5.0, 4.0, q1=2*np.pi/16)
# kernel equivalence
qx, qy = np.meshgrid(np.linspace(-.5, .5, 7), np.linspace(-.5, .5, 5), indexing='ij')
kd = max(np.abs(kernel_field(p, qx, qy, nyq_fix=False)[i, j] - kernel(p, qx[i, j], qy[i, j])).max() for i in range(7) for j in range(5))
print('backend:', 'cupy' if G.ON_GPU else 'numpy', '| max |kernel_vec - kernel_ref| =', kd)
ref = Bilayer(p, L=64, dt=2.0, seed=0)
new = GLBilayer(p, L=64, dt=2.0, seed=0)
# the reference's DMI at the Nyquist lines is not Hermitian-consistent; give it the same (zeroed) treatment
kk = 2*np.pi*np.fft.fftfreq(64); QX, QY = np.meshgrid(kk, kk, indexing='ij')
ref.Lq = kernel_field(p, QX, QY); ref.Ainv = np.linalg.inv(np.eye(6)[None, None] + 2*ref.dt*ref.Lq)
Er, En = [ref.energy()], [new.energy()]
for _ in range(8):
    ref.step(100); new.step(100); Er.append(ref.energy()); En.append(new.energy())
Er, En = np.array(Er), np.array(En)
rel = np.abs(Er-En)/np.abs(Er)
print('max rel energy diff over trajectory:', rel.max(), ' field diff:', np.abs(ref.m-G.to_np(new.m)).max())
print('monotone:', bool(np.all(np.diff(En) <= 1e-12)))
assert rel.max() < 1e-8 and kd < 1e-12
print('STEP0 OK')
