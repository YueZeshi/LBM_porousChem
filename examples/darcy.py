import time
import taichi as ti
import taichi.profiler as profiler
import os
import sys
import numpy as np
# 指定求解器
from LBM.LBM2D import LBM2DSolver as LB2D
from LBM.GEO.G2D import Mesh2D
from LBM.util.flag import *
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
    X = 100
    Y = 100
    name = "validation_darcy"

    # 初始化taichi
    ## arch=ti.cpu 启用cpu计算；arch=ti.gpu启用gpu运算 (cuda>vulkan)
    if ARCH=="gpu":
        ti.init(arch=ti.gpu, kernel_profiler=True, print_ir=False)
    else:
        ti.init(arch=ti.cpu, kernel_profiler=True, print_ir=False)
    # 初始化lbm模型
    lb2d = LB2D(X,Y,dx=DX,dt=DT,isPoro=True,isChemical=False,isThermal=False,isRadiation=False)
    # 基础设置
    # lb2d.source_term_model = SOURCE_TERM.MICRO
    Re  = 0.1
    Da = 1e-7
    eps = 0.5
    nv = 0.1
    rho = 1
    v = Re*nv/Y/rho
    K = Da*Y*Y
    print(1.5-ti.exp(-eps*nv/K)/2,1+eps*nv/K/2)
    lb2d.force_term_model = FORCE_TERM.GUO
    lb2d.set_poro_Darcy(1/K)
    # lb2d.poro_model=PORO_MODEL.DARCY_HIGH
    lb2d.EOS = FLUID_STATE_EQUATION.INCOMPRESSIBLE
    lb2d.bondary_condition_model = BC_MODEL.NEE
    lb2d.set_viscosity(nv)
    # 设置边界条件
    lb2d.set_BCs([BC_FLOW.inlet,BC_FLOW.outlet,BC_FLOW.wall,BC_FLOW.wall])
    lb2d.set_v_BCs_value([[v,0,0],[0,0,0],[0,0,0],[0,0,0]])
    lb2d.set_rho_BCs_value([1,1,1,1])
    ## 初始化场 
    lb2d.init_field(lb2d.rho,1)
    m2d = Mesh2D(lb2d.nx,lb2d.ny)
    m2d.CreateMesh2DRectangle(ti.Vector([40,-10,0]),ti.Vector([60,200,0]))
    s,l =m2d.export_numpy()
    lb2d.init_field(lb2d.solid,(1-eps)*s)
    # 初始化lbm
    lb2d.init_simulation()
    lb2d.print_information()

    total_iteration =   100000
    export_interval = 1000
    measure_interval= 100
    print_interval = 100
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
            max_v = lb2d.get_max_v()
            min_T = lb2d.get_min_T()
            print(name,flush=True)
            print('----------Time between two outputs is %dh %dm %ds; elapsed time is %dh %dm %ds----------------------' %(h_diff, m_diff, s_diff,h_elap,m_elap,s_elap))
            print('The %dth iteration, Max Force = %f,  Min Temperature = %f\n\n ' %(iter, max_v,  min_T))            
        if (iter%int(export_interval/DT)==0):
        
            if DEBUG:
                lb2d.export_VTK(f"debug_{name}_{variant}_{DX}",iter)
                # lb2d.export_variable(f"simulation_{name}_{int(variant)}_{nx}_{int(T_exp)}",iter)
            else:
                lb2d.export_VTK(f"simulation_{name}_{variant}_{DX}",iter)
        if (iter%int(measure_interval/DT)==0):
                lb2d.export_variable(f"simulation_{name}_{variant}_{DX}",iter)
        lb2d.step()


    profiler.print_kernel_profiler_info()
    # profiler.print_memory_profiler_info()
if __name__ == "__main__":
    DX = sys.argv[1]
    DT = sys.argv[2]
    variant = sys.argv[3]
    main(float(DX),float(DT),variant)