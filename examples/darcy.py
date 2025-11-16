from re import M, T
import time
import taichi as ti
import taichi.profiler as profiler
import os
import sys
import numpy as np
# 指定求解器
from LBM.LBM2D import LBM2DSolver as LB2D
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
    X = 400
    Y = 300
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
    lb2d.force_term_model = FORCE_TERM.GUO
    lb2d.set_poro_Darcy(1)
    lb2d.EOS = FLUID_STATE_EQUATION.INCOMPRESSIBLE
    lb2d.bondary_condition_model = BC_MODEL.NEE
    lb2d.set_viscosity(0.9)
    # lb2d.set_radiation_model(RADIATION_MODEL.SURFACE_UNIFORM,T_exp)

    # # 设置物质
    # ## 物种及其状态
    # lb2d.set_specie("N2",False)
    # lb2d.set_species(["wood(S)","tar" ,"char(S)"],
    #                 [True     ,False,True     ])
    # 设置边界条件
    lb2d.set_BCs([BC_FLOW.inlet,BC_FLOW.outlet,BC_FLOW.wall,BC_FLOW.wall])
    lb2d.set_v_BCs_value([[0.01,0,0],[0,0,0],[0,0,0],[0,0,0]])
    lb2d.set_rho_BCs_value([1.1,1,1,1])
    # ## 杂项
    # ## 可变边界条件
    # def setChangingBC():
    #     def TBC(lbm:LB2D_PYRO,t:float):
    #         # 先稳定流场再升温，防止数值扰动
    #         if t<=10:
    #             lbm.TF.s_BC[0] = T_init
    #             lbm.TF.s_BC[2] = T_init
    #             lbm.TF.s_BC[3] = T_init
    #         else:
    #             lbm.TF.s_BC[0] = T_exp
    #             lbm.TF.s_BC[2] = T_exp
    #             lbm.TF.s_BC[3] = T_exp
    #     # lb2d.UpdateBCfunc.append(TBC)
    # setChangingBC()
    # # 导出局部或者全局变量 观测量
    # ## 中心温度 表面温度 转化率 
    # ### 定义变量实现CPU GPU数据传输
    # allWood = ti.field(ti.f32,shape=())
    # conv = ti.field(ti.f32,shape=())
    # total_rad_surface = ti.field(ti.f32,shape=())  
    # @ti.func
    # def cal_wood():
    #     wood = 0.0
    #     for i in ti.grouped(lb2d.rho):
    #         wood += lb2d.species["wood(S)"].S[i]
    #     return wood
    # @ti.kernel
    # def cal_allWood():
    #     allWood[None] = cal_wood()
    #     print("all wood:",allWood[None])
    moment = ti.field(ti.f32,shape=())

    def setVariables():
        def get_moment(lbm:LB2D):
            cal_moment()
            return "moment", moment[None]
        lb2d.GetVariableFunc.append(get_moment)
        @ti.kernel
        def cal_moment():
            moment[None]=0.0
            for j,k in ti.ndrange(lb2d.ny,lb2d.nz):
                moment[None]+=lb2d.v[lb2d.nx-10,j,k][0]
    setVariables()
    ## 初始化场 
    lb2d.init_field(lb2d.rho,1)
    lb2d.init_field(lb2d.solid,0.0)
    # lb2d.init_field(lb2d.TF.S,T_exp)
    # lb2d.init_field(lb2d.TS.S,T_init)

    # lb2d.init_specie("N2",1)
    # lb2d.init_specie("wood(S)",biomass_file)
    # lb2d.init_field(lb2d.radiation_surface,radiation_surface_file)# vague
    # lb2d.init_field(lb2d.heat_trasnfer_surface,1) # vague
    # 初始化lbm
    lb2d.init_simulation()
    lb2d.print_information()

    # cal_allWood() # 计算总木材质量

    total_iteration =   50000
    export_interval = 1000
    measure_interval= 100
    print_interval = 1000
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