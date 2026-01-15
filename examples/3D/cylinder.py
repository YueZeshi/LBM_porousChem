from re import T
import time
import taichi as ti
import taichi.profiler as profiler
import os
import sys
import numpy as np
# 指定求解器
from LBM.LBM3D import LBM3DSolver as LB3D
from LBM.util.flag import *
from LBM.GEO.STL import StlGenerator
def main(DX,DT,variant):
    # 获取环境变量是否启用debug模式
    # debug模式计算较少的步数 用于检验算例是否快速发散
    print(os.getenv("DEBUG","false"))
    DEBUG = os.getenv("DEBUG","False").lower() == "true" # 默认非debug
    ARCH = os.getenv("ARCH","cpu").lower() # 默认CPU
    ## 用于获取运算时间信息
    time_init = time.time()
    time_now = time.time()
    time_pre = time.time() 

    # GLOBAL VARIABLE
    # 定义计算域 SI
    X = 500
    Y = 100
    Z = 100
    R = 10
    locX= 100
    locY=50
    locZ= 50
    height = 80
    U= 0.02
    Re = 100
    nu = U*R*2/Re

    name = "validation_cylinder"

    # 初始化taichi
    ## arch=ti.cpu 启用cpu计算；arch=ti.gpu启用gpu运算 (cuda>vulkan)
    if ARCH=="gpu":
        ti.init(arch=ti.gpu, kernel_profiler=True, print_ir=False)
    else:
        ti.init(arch=ti.cpu, kernel_profiler=True, print_ir=False)
    # 初始化lbm模型
    lb3d = LB3D(X,Y,Z,dx=DX,dt=DT,isPoro=False,isChemical=False,isThermal=False,isRadiation=False,name = name)
    # 基础设置
    lb3d.set_viscosity(nu)

    lb3d.set_BCs([BC_FLOW.inlet,BC_FLOW.outlet,BC_FLOW.wall,BC_FLOW.wall,BC_FLOW.wall,BC_FLOW.wall])
    lb3d.set_v_BCs_value([[U,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
    lb3d.set_rho_BCs_value([1,1,1,1,1,1])

    ## 初始化场 
    cylinder = StlGenerator().create_cylinder()
    s,l = lb3d.load_stl(cylinder,scale = [2*R,2*R,height],translate = [locX,locY,locZ])
    lb3d.init_field(lb3d.solid,s)
    lb3d.init_field(lb3d.rho,1)
    # lb3d.init_field3(lb3d.v,U,0,0)
    # 初始化lbm
    lb3d.init_simulation()
    print(lb3d.description())

    total_iteration =   1000
    export_interval = 50
    print_interval = 10
    if DEBUG:
        total_iteration = 2
        export_interval = 1
        print_interval = 1
        print("debug")

    for iter in range(int(total_iteration/DT)+1):
        if iter==1:
            print("init, complie and execute once time:",time.time()-time_init)
        if (iter%int(print_interval/DT)==0):
            time_pre = time_now
            time_now = time.time()
            diff_time = int(time_now-time_pre)
            elap_time = int(time_now-time_init)
            m_diff, s_diff = divmod(diff_time, 60)
            h_diff, m_diff = divmod(m_diff, 60)
            m_elap, s_elap = divmod(elap_time, 60)
            h_elap, m_elap = divmod(m_elap, 60)
            max_v = lb3d.get_max_v()
            min_T = lb3d.get_min_T()
            print(lb3d.log_info(), flush=True)
        if (iter%int(export_interval/DT)==0):
            if DEBUG:
                lb3d.export_VTK_pyvista()
                # lb3d.export_variable(f"simulation_{name}_{int(variant)}_{nx}_{int(T_exp)}",iter)
            else:
                lb3d.export_VTK_pyvista()
                
        lb3d.step()


    profiler.print_kernel_profiler_info()
    # profiler.print_memory_profiler_info()
if __name__ == "__main__":
    dx = sys.argv[1] if len(sys.argv) > 1 else 1
    dt = sys.argv[2] if len(sys.argv) > 2 else 1
    variant = sys.argv[3] if len(sys.argv) > 3 else 0
    main(DX=float(dx), DT=float(dt), variant=variant)