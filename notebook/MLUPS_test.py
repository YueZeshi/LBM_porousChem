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
        print(ti.cfg.arch) # type: ignore 
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
        time_list.append(time.time())
        if verbose:
            print(lb.log_info())
            # lb.export_VTK_pyvista()
    if verbose:
        print(time_list)
    nLattice = lb.nx*lb.ny*lb.nz
    time_init = time_list[1]-time_list[0]
    time_first = time_list[3]-time_list[2]
    time_10000 = []
    MLUPS_list = []
    for i in range(10):
        exec_time = time_list[2*i+5]-time_list[2*i+4]
        time_10000.append(exec_time)
        MLUPS_list.append(nLattice*10000/exec_time/1e6)
    avgMLUPS = np.mean(MLUPS_list)
    stdMLUPS = np.std(MLUPS_list)
    return nLattice,avgMLUPS,stdMLUPS,time_first,time_init,time_10000

if __name__=="__main__":
    verbose = False
    if len(sys.argv)>1:
        verbose = True
    i = 0
    nskip = 0
    data_name = ["LatticeNumber","AvgMLUPS","StdMLUPS","CompileTime","InitTime","Exe10000Times"]
    for dx in np.logspace(-2,-4,100):
        if i < nskip:
            i += 1
            print("Skipping:", dx)
            continue
        info = MLUPS(dx,verbose)
        k = 0
        for item in info:
            print(data_name[k],":",item)
            k += 1
        print("-----",flush=True)
