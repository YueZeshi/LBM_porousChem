import sys
import taichi as ti
import time
import numpy as np
from LBM.LBM3D import LBM3DSolver
from LBM.util.flag import BC_FLOW

X = 0.100
Y = 0.111
Z= 0.123
def MLUPS(dx,verbose= False):
    time_list = []
    dt = dx
    nLattice = 0
    ti.reset()
    ti.init(arch=ti.gpu)
    if verbose:
        print(ti.cfg.arch)
    lb = LBM3DSolver(X,Y,Z,dx,dt)
    lb.set_BCs([BC_FLOW.inlet, BC_FLOW.outlet, BC_FLOW.wall, BC_FLOW.wall, BC_FLOW.wall, BC_FLOW.wall])
    # lb.set_viscosity()
    lb.set_v_BC_value(0,[0.01,0,0])
    lb.set_vtk_path("res")
    time_list.append(time.time())
    lb.init_simulation()
    time_list.append(time.time())
    if verbose:
        print(lb.description())
    time_list.append(time.time())
    lb.step()
    time_list.append(time.time())
    for i in range(10):
        time_list.append(time.time())
        for j in range(10000):
            lb.step()
            ti.sync()
        time_list.append(time.time())
        if verbose:
            print(lb.log_info())
            # lb.export_VTK_pyvista()
    if verbose:
        print(time_list)
    # v = lb.v[lb.nx//2,lb.ny//2,lb.nz//2]
    nLattice = lb.nx*lb.ny*lb.nz
    time_init = time_list[1]-time_list[0]
    time_first = time_list[3]-time_list[2]
    time_10000 = []
    for i in range(10):
        time_10000.append(time_list[2*i+5]-time_list[2*i+4])
    time_avg_10000 = np.array(time_10000).mean()
    MLUPS = nLattice*10000/time_avg_10000/1e6
    return nLattice,MLUPS,time_avg_10000,time_first,time_init,time_10000

if __name__=="__main__":
    verbose = False
    if len(sys.argv)>1:
        verbose = True
    i = 0
    nskip = 0
    for dx in np.logspace(-2,-4,100):
        if i < nskip:
            i += 1
            print("Skipping:", dx)
            continue
        info = MLUPS(dx,verbose)
        for item in info:
            print(item)
        print("-----",flush=True)
