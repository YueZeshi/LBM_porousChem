import argparse
from ast import arg, parse
import json
from LBM3D import LBM3DSolver

import taichi as ti
from taichi import profiler
import time


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--case",help="case.json",default="case.json")
    parser.add_argument("-o","--output",help="output location",default="output")
    parser.add_argument("--control",help="control.json",default="control.json")
    parser.add_argument("-a","--arch",help="arch platform",default="cpu")
    parser.add_argument("-v","--verbose",help="verbose level",type=int,default=1)
    parser.add_argument("-e","--export",help="export path",default="lbm_snapshot.json")
    args = parser.parse_args()
    verbose = args.verbose
    time_init = time.time()
    if args.arch=="gpu":
        ti.init(arch=ti.gpu)
    else:
        ti.init(arch=ti.cpu)
    lbm3d = LBM3DSolver.CreateLBM3D(args.case)
    lbm3d.init_simulation()
    if verbose>0:
        print("Initialization completed. Execution time(s):",time.time()-time_init)
    if verbose>1:
        lbm3d.print_information()
    time_cur = time.time()
    with open(args.control,"r") as f:
        controlConfig = json.load(f)
    T = controlConfig["T"]
    writeInterval = controlConfig["writeInterval"]
    printInterval = controlConfig["printInterval"]
    for i in range(int(T/lbm3d.dt)+1):
        if(i==1 and verbose>0):
            print("First step completed. All kernels are compiled. Execution time(s):",time.time()-time_cur,"Total time(s):",time.time()-time_init)
            time_cur = time.time()
            print("")
        if (i%int(writeInterval/lbm3d.dt)==0):
            lbm3d.export_VTK(args.output,i)
            lbm3d.export_LBM(args.export)
        if (i%int(printInterval/lbm3d.dt)==0):
            print("Step:",i,", Simulation time:",i*lbm3d.dt," s, Execution time:",time.time()-time_cur,"s, Total time(s):",time.time()-time_init," s")
            max_v = lbm3d.get_max_v()
            min_T = lbm3d.get_min_T()
            print("MAX VELOCITY(LU):",max_v,"MIN TEMPERATURE:",min_T)
            time_cur = time.time()
            print("")
        lbm3d.step()
    if verbose>0:
        print("Simulation completed. Total execution time:",time.time()-time_init," s")
    if verbose>1:
        profiler.print_scoped_profiler_info()
        profiler.print_kernel_profiler_info()


if __name__=="__main__":
    run()
