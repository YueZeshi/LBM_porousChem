import sys
import os
import taichi as ti
import time
import numpy as np
import cli 
from LBM.LBM3D import LBM3DSolver
from LBM.util.flag import BC_FLOW

global nlattice_list
X = 0.100
Y = 0.111
Z= 0.123

def parse_log_v2(log_file):
    n_lattice_list = []
    avg_MLUPS_list = []
    std_MLUPS_list = []
    init_time_list = []
    compile_time_list = []
    with open(log_file, 'r') as f:
        line = f.readline() # 第一行忽略
        line = f.readline()
        while not line.startswith("[Taichi]"):
            line = f.readline()    # skip skipping
        while line: 
            line2 = f.readline() # n lattice
            if line2.startswith("Already"):
                line = f.readline()
            elif  line2.startswith("[Taichi]"):
                continue
            else:
                line3 = f.readline() # avg MLUPS
                line4 = f.readline() # std MLUPS
                line5 = f.readline() # compile time
                line6 = f.readline() # init time
                line8 = f.readline() # execution time list
                line9 = f.readline() # -----
                line = f.readline() # EOF or next arch
                try:
                    nLattice = int(line2.split(":")[1].strip())
                    n_lattice_list.append(nLattice)
                    compile_time_list.append(float(line5.split(":")[1].strip()))
                    init_time_list.append(float(line6.split(":")[1].strip()))
                    # exec_time_list = [float(t) for t in line8.split(':')[1] if t.strip()!='']
                    avg_MLUPS_list.append(float(line3.split(":")[1].strip()))
                    std_MLUPS_list.append(float(line4.split(":")[1].strip()))
                except: # invalid ending due to interruption
                    pass
        index = np.argsort(n_lattice_list)
        n_lattice_list = np.array(n_lattice_list)[index].tolist()
        avg_MLUPS_list = np.array(avg_MLUPS_list)[index].tolist()
        std_MLUPS_list = np.array(std_MLUPS_list)[index].tolist()
        compile_time_list = np.array(compile_time_list)[index].tolist()
        init_time_list = np.array(init_time_list)[index].tolist()
    return n_lattice_list, avg_MLUPS_list, std_MLUPS_list, compile_time_list, init_time_list
def MLUPS(dx,verbose= False,debug = False):
    global nlattice_list
    time_list = []
    dt = dx
    nLattice = 0
    N = np.round(X/dx).astype(int)*np.round(Y/dx).astype(int)*np.round(Z/dx).astype(int)
    if debug:
        print(lb.nx*lb.ny*lb.nz,flush=True)
        if lb.nx*lb.ny*lb.nz not in nlattice_list:
            return lb.nx*lb.ny*lb.nz,0,0,0,0,[0]
        else:
            print("Already have data for lattice number:", lb.nx*lb.ny*lb.nz)
            return None
    else:
        if N not in nlattice_list: 
            ti.reset()
            ti.init(arch=ti.gpu)
            if verbose:
                print(ti.cfg.arch) # type: ignore
            time_list.append(time.time())
            lb = LBM3DSolver(X,Y,Z,dx,dt)
            lb.set_BCs([BC_FLOW.inlet, BC_FLOW.outlet, BC_FLOW.wall, BC_FLOW.wall, BC_FLOW.wall, BC_FLOW.wall])
            # lb.set_viscosity()
            lb.set_v_BC_value(0,[0.01,0,0])
            lb.set_vtk_path("res")
    
            lb.init_simulation()
            ti.sync()
            time_list.append(time.time())
            # print(lb.description())
            if verbose:
                print(lb.nx*lb.ny*lb.nz)
                print(lb.description())
            time_list.append(time.time())
            lb.step()
            time_list.append(time.time())
            for j in range(100):
                lb.step()
            for i in range(5):
                time_list.append(time.time())
                for j in range(100):
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
            time_100 = []
            MLUPS_list = []
            for i in range(5):
                exec_time = time_list[2*i+5]-time_list[2*i+4]
                time_100.append(exec_time)
                MLUPS_list.append(nLattice*100/exec_time/1e6)
            avgMLUPS = np.mean(MLUPS_list)
            stdMLUPS = np.std(MLUPS_list)
            return nLattice,avgMLUPS,stdMLUPS,time_first,time_init,time_100
        else:
            if verbose:
                print("Already have data for lattice number:", N,flush=True)
            return None

if __name__=="__main__":
    verbose = False
    debug = False
    if len(sys.argv)==2:
        verbose = True
    if len(sys.argv)==3:  
        debug = True
        
    i = 0
    nskip = 0
    data_name = ["LatticeNumber","AvgMLUPS","StdMLUPS","CompileTime","InitTime","Exe100Times"]
    nlattice_list = [] # parse_log_v2("log\\YZS_legion_cuda_AA_all.log")[0]

    for dx in np.logspace(-2,-5,100):
        if i < nskip:
            i += 1
            if verbose:
                print("Skipping:", dx,flush=True)
            continue
        else:
            i += 1
        try:
            info = MLUPS(dx,verbose,debug)
        except RuntimeError as e:
            print("Error in MLUPS for dx =", dx, ":", e,flush=True)
            break
        if info is not None:
            nlattice_list.append(int(info[0]))
            k = 0
            for item in info:
                print(data_name[k],":",item)
                k += 1
            print("-----",flush=True)
    
