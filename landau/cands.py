import numpy as np
from rings import *
R12=2*np.cos(np.pi/12); R8=2*np.cos(np.pi/8)
def build():
    S={}
    S['QC12']=State('QC12',[ring_dirs(1,np.arange(0,180,30)),ring_dirs(R12,np.arange(15,180,30))])
    S['HEX(1,2)']=State('HEX(1,2)',[ring_dirs(1,[0,120,240]),ring_dirs(2,[0,120,240])])
    S['HEX(1,√3,2)']=State('HEX(1,√3,2)',[ring_dirs(1,[0,120,240]),ring_dirs(np.sqrt(3),[30,150,270]),ring_dirs(2,[0,120,240])])
    S['QC8']=State('QC8',[ring_dirs(1,np.arange(0,180,45)),ring_dirs(R8,np.arange(22.5,180,45))])
    o49=orbits(shell(49))
    big=[o for o in o49 if np.isclose(abs(o[0][1]),0) or True]
    s70=[o for o in o49 if any(np.isclose(v[1],0) for v in o)][0]
    s53=[v for o in o49 if o is not s70 for v in o]
    S['APX(13|48,49)']=State('APX(13|48,49)',[shell(13),shell(48),s70,s53])
    return S

def build2():
    S=build()
    S['QC12s']=State('QC12s',[ring_dirs(1,[0,60,120]),ring_dirs(1,[30,90,150]),ring_dirs(R12,[15,75,135]),ring_dirs(R12,[45,105,165])])
    s13=shell(13); o13=orbits(s13)
    o49=orbits(shell(49)); s70=[o for o in o49 if any(np.isclose(v[1],0) for v in o)][0]; s53=[v for o in o49 if o is not s70 for v in o]
    S['APXs']=State('APXs',[o13[0],o13[1],shell(48),s70,s53])
    return S
