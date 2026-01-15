import time
import taichi as ti
import taichi.profiler as profiler
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
# 指定求解器
from LBM.LBM2D import LBM2DSolver
from LBM.GEO.STL import StlGenerator
from LBM.util.flag import *
def main(DX,DT,T_exp,variant="default"):
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
    X = 0.6
    Y = 0.2
    R = 0.02

    ## convert to lattice unit
    T_init = 303

    print("executing ",__name__)
    name = "pyrolysis_Park"
    # 初始化taichi
    ## arch=ti.cpu 启用cpu计算；arch=ti.gpu启用gpu运算 (cuda>vulkan)
    default_fp  = ti.f32
    if ARCH=="gpu":
        ti.init(arch=ti.gpu, kernel_profiler=True, print_ir=False,default_fp=default_fp)
    else:
        ti.init(arch=ti.cpu, kernel_profiler=True, print_ir=False,default_fp=default_fp)
    # 初始化lbm模型
    lb2d = LBM2DSolver(X,Y,dx=DX,dt=DT,isPoro=True,isChemical=True,isThermal=True,isRadiation=True)
    # 基础设置
    lb2d.source_term_model = SOURCE_TERM.MICRO
    lb2d.force_term_model = FORCE_TERM.GUO
    lb2d.EOS = FLUID_STATE_EQUATION.IDEAL_GAS
    lb2d.set_viscosity(0.1)
    lb2d.set_radiation(RADIATION_MODEL.SURFACE_UNIFORM,T_exp)
    # 设置物质
    ## 物种及其状态
    lb2d.set_specie("N2",False,28)
    lb2d.set_species(["wood(S)","intermSolid(S)","tar" ,"gas","char(S)"],
                    [True     ,     True       ,False ,False,True     ],
                    [])
    # 设置边界条件
    lb2d.set_BCs([BC_FLOW.inlet,BC_FLOW.outlet,BC_FLOW.wall,BC_FLOW.wall])
    lb2d.set_v_BCs_value([[0.01,0,0],[0,0,0],[0,0,0],[0,0,0]])
    lb2d.set_rho_BCs_value([1]*4)
    lb2d.set_TF_BCs([BC.fixedValue,BC.zeroGradient,BC.fixedValue,BC.fixedValue])
    lb2d.set_TF_BCs_value([T_exp]*4)
    lb2d.set_TS_BCs([BC.fixedValue,BC.zeroGradient,BC.zeroGradient,BC.zeroGradient])
    lb2d.set_TS_BCs_value([T_init]*4)
    lb2d.set_species_BCs([BC.fixedValue,BC.zeroGradient,BC.zeroGradient,BC.zeroGradient])
    lb2d.set_specie_BCs_value("N2",[1]*4)

    # 添加化学反应
    lb2d.add_reaction("wood(S)=tar",1.08e10,148000,0,300,80000)
    lb2d.add_reaction("wood(S)=gas",4.38e9,152700,0,300,80000)
    lb2d.add_reaction("wood(S)=intermSolid(S)",3.75e6,111700,0,300,80000)
    lb2d.add_reaction("intermSolid(S)=char(S)",1.38e10,161000,0,300,-300000)
    lb2d.add_reaction("tar=char(S)",1e5,108000,0,300,-42000)
    lb2d.add_reaction("tar=gas",4.28e6,108000,0,300,-42000)
    # 设置物种物性
    ## 扩散
    lb2d.set_specie_diff("tar",1e-6,unit="SI")
    lb2d.set_specie_diff("gas",1e-5,unit="SI")
    lb2d.set_specie_diff("N2",1e-5,unit="SI")
    ## 热容
    @ti.func
    def intermSolid_capacity(self,i):
        T = self.LBM.TS.S[i]
        return 1500+T
    lb2d.set_specie_capacity_func("intermSolid(S)",intermSolid_capacity)
    @ti.func
    def wood_capacity(self,i):
        T = self.LBM.TS.S[i]
        return 1500+T
    lb2d.set_specie_capacity_func("wood(S)",wood_capacity)
    @ti.func
    def char_capacity(self,i):
        T = self.LBM.TS.S[i]
        return 420+2.09*T+6.85e-4*T**2
    lb2d.set_specie_capacity_func("char(S)",char_capacity)
    @ti.func
    def tar_capacity(self,i):
        T = self.LBM.TF.S[i]
        return -100+4.4*T-1.57e-3*T**2
    lb2d.set_specie_capacity_func("tar",tar_capacity)
    @ti.func
    def syngas_capacity(self,i):
        T = self.LBM.TF.S[i]
        return 770+0.629*T-1.91e-4*T**2
    lb2d.set_specie_capacity("gas",syngas_capacity)
    lb2d.set_specie_capacity("N2",742)
    ## 热导
    lb2d.set_specie_conductivity("intermSolid(S)", 0.20487)
    lb2d.set_specie_conductivity("wood(S)", 0.20487)
    lb2d.set_specie_conductivity("char(S)", 0.0937)
    lb2d.set_specie_conductivity("tar", 0.0258)
    lb2d.set_specie_conductivity("gas", 0.0258)
    lb2d.set_specie_conductivity("N2",0.0258)
    ## 杂项

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
    allWood = ti.field(float,shape=())
    conv = ti.field(float,shape=())
    total_rad_surface = ti.field(float,shape=())  
    @ti.func
    def cal_wood():
        wood = 0.0
        for i in ti.grouped(lb2d.rho):
            wood += lb2d.species[1].S[i]
        return wood
    @ti.kernel
    def cal_allWood():
        allWood[None] = cal_wood()
        print("all wood:",allWood[None])
    
    def setVariables():
        def Tcenter(lbm:LBM2DSolver):
            T_center = lbm.TS.S[(int(lbm.nx/2),int(lbm.ny/2),int(lbm.nz/2))]
            return "Ts_center", T_center
        lb2d.GetVariableFunc.append(Tcenter)
        def Tsurface(lbm:LBM2DSolver):
            T_surface = lbm.TF.S[(int(lbm.nx/2+R/DX-1),int(lbm.ny/2),int(lbm.nz/2))]
            return "Tf_surface", T_surface
        lb2d.GetVariableFunc.append(Tsurface)
        @ti.kernel
        def cal_conversion():
            conv[None] = (allWood[None]-cal_wood())/allWood[None]
        def conversion(lbm:LBM2DSolver):
            cal_conversion()
            return "conversion", conv[None]
        lb2d.GetVariableFunc.append(conversion)
    setVariables()
    ## 初始化场 
    lb2d.init_field(lb2d.rho,1)
    cylinder  = StlGenerator().create_cylinder(2*R/DX,float(lb2d.ny)/2,float(lb2d.nx)/2, float(lb2d.ny)/2)
    s,l = lb2d.load_stl(cylinder)
    lb2d.set_poro_Darcy(s,2.5e10)
    lb2d.init_field(lb2d.solid,s*0.4)
    lb2d.init_field(lb2d.TF.S,T_init)
    lb2d.init_field(lb2d.TS.S,T_init)
    lb2d.init_field(lb2d.TS.exchangeSurface,100)
    lb2d.init_field(lb2d.TS.exchangeCoef,1000)
    lb2d.init_field(lb2d.TS.emissivity,0.7)
    lb2d.init_specie("N2",1)
    lb2d.init_specie("wood(S)",s*400)
    lb2d.init_field(lb2d.TS.radiation_surface, l*0.6)# vague
    # 初始化lbm
    lb2d.init_simulation()
    print(lb2d.description())
    lb2d.check_python()
    # cal_allWood() # 计算总木材质量
    total_iteration =   1000
    export_interval = 10
    measure_interval= 1
    print_interval = 10
    if DEBUG:
        total_iteration = 0.01
        export_interval = 0.01
        print_interval = 0.01
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
            if max_v>1:
                print("diverge")
                break          
        if (iter%int(export_interval/DT)==0):
            if DEBUG:
                lb2d.export_VTK_pyvista()
                # lb2d.export_variable(f"simulation_{name}_{int(variant)}_{nx}_{int(T_exp)}",iter)
            else:
                lb2d.export_VTK_pyvista()
        if (iter%int(measure_interval/DT)==0):
                lb2d.export_variable()
        lb2d.step()


    profiler.print_kernel_profiler_info()
    # profiler.print_memory_profiler_info()
if __name__=="__main__":
    os.environ["ARCH"] = "CPU"
    dx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    T_exp = float(sys.argv[3]) if len(sys.argv) > 3 else 500
    variant = sys.argv[4] if len(sys.argv) > 4 else "default"
    startTime = time.time()
    main(dx, dt, T_exp, variant)
    print("execution time: ", time.time() - startTime)
    