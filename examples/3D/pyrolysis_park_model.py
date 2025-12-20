import time
import taichi as ti
import taichi.profiler as profiler
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
# 指定求解器
from LBM.LBM3D import LBM3DSolver
from LBM.GEO.G3D import Mesh3D
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
    Z = 0.2
    R = 0.02

    ## convert to lattice unit
    T_init = 303

    print("executing ",__name__)
    name = "pyrolysis_Park_3D"
    # 初始化taichi
    ## arch=ti.cpu 启用cpu计算；arch=ti.gpu启用gpu运算 (cuda>vulkan)
    default_fp  = ti.f32
    if ARCH=="gpu":
        ti.init(arch=ti.gpu, kernel_profiler=True, print_ir=False,default_fp=default_fp)
    else:
        ti.init(arch=ti.cpu, kernel_profiler=True, print_ir=False,default_fp=default_fp)
    # 初始化lbm模型
    lb3D = LBM3DSolver(X,Y,Z,dx=DX,dt=DT,isPoro=True,isChemical=True,isThermal=True,isRadiation=True)
    # 基础设置
    lb3D.source_term_model = SOURCE_TERM.MICRO
    lb3D.force_term_model = FORCE_TERM.GUO
    lb3D.EOS = FLUID_STATE_EQUATION.IDEAL_GAS
    lb3D.set_viscosity(0.1)
    lb3D.set_poro_Darcy(2.5e11,unit="SI")
    lb3D.set_radiation(RADIATION_MODEL.SURFACE_UNIFORM,T_exp)
    # 设置物质
    ## 物种及其状态
    lb3D.set_specie("N2",False)
    lb3D.set_species(["wood(S)","intermSolid(S)","tar" ,"gas","char(S)"],
                    [True     ,     True       ,False ,False,True     ])
    # 设置边界条件
    lb3D.set_BCs([BC_FLOW.inlet,BC_FLOW.outlet,BC_FLOW.wall,BC_FLOW.wall,BC_FLOW.wall,BC_FLOW.wall])
    lb3D.set_v_BCs_value([[0.01,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
    lb3D.set_rho_BCs_value([1]*6)
    lb3D.set_TF_BCs([BC.fixedValue,BC.zeroGradient,BC.fixedValue,BC.fixedValue,BC.fixedValue,BC.fixedValue])
    lb3D.set_TF_BCs_value([T_exp]*6)
    lb3D.set_TS_BCs([BC.fixedValue,BC.zeroGradient,BC.zeroGradient,BC.zeroGradient,BC.zeroGradient,BC.zeroGradient])
    lb3D.set_TS_BCs_value([T_init]*6)
    lb3D.set_species_BCs([BC.fixedValue,BC.zeroGradient,BC.zeroGradient,BC.zeroGradient,BC.zeroGradient,BC.zeroGradient])
    lb3D.set_specie_BCs_value("N2",[1]*6)

    # 添加化学反应
    # lb3D.add_reaction("w2t",[("wood(S)",1)],[("tar",1)],(1.08e10,0,148000,300,80000))
    # lb3D.add_reaction("w2syn",[("wood(S)",1)],[("gas",1)],(4.38e9,0,152700,300,80000))
    # lb3D.add_reaction("w2is",[("wood(S)",1)],[("intermSolid(S)",1)],(3.75e6,0,111700,300,80000))
    # lb3D.add_reaction("is2c",[("intermSolid(S)",1)],[("char(S)",1)],(1.38e10,0,161000,300,-300000))
    # lb3D.add_reaction("t2c",[("tar",1)],[("char(S)",1)],(1e5,0,108000,300,-42000))
    # lb3D.add_reaction("t2syn",[("tar",1)],[("gas",1)],(4.28e6,0,108000,300,-42000))
    # 设置物种物性
    ## 扩散
    lb3D.set_specie_diff("tar",1e-6,unit="SI")
    lb3D.set_specie_diff("gas",1e-5,unit="SI")
    lb3D.set_specie_diff("N2",1e-5,unit="SI")
    ## 热容
    @ti.func
    def intermSolid_capacity(self,i):
        T = self.LBM.TS.S[i]
        return 1500+T
    lb3D.set_specie_capacity_func("intermSolid(S)",intermSolid_capacity)
    @ti.func
    def wood_capacity(self,i):
        T = self.LBM.TS.S[i]
        return 1500+T
    lb3D.set_specie_capacity_func("wood(S)",wood_capacity)
    @ti.func
    def char_capacity(self,i):
        T = self.LBM.TS.S[i]
        return 420+2.09*T+6.85e-4*T**2
    lb3D.set_specie_capacity_func("char(S)",char_capacity)
    @ti.func
    def tar_capacity(self,i):
        T = self.LBM.TF.S[i]
        return -100+4.4*T-1.57e-3*T**2
    lb3D.set_specie_capacity_func("tar",tar_capacity)
    @ti.func
    def syngas_capacity(self,i):
        T = self.LBM.TF.S[i]
        return 770+0.629*T-1.91e-4*T**2
    lb3D.set_specie_capacity_func("gas",syngas_capacity)
    lb3D.set_specie_capacity("N2",742)
    ## 热导
    lb3D.set_specie_conductivity("intermSolid(S)", 0.20487)
    lb3D.set_specie_conductivity("wood(S)", 0.20487)
    lb3D.set_specie_conductivity("char(S)", 0.0937)
    lb3D.set_specie_conductivity("tar", 0.0258)
    lb3D.set_specie_conductivity("gas", 0.0258)
    lb3D.set_specie_conductivity("N2",0.0258)
    ## 杂项

    # ## 可变边界条件
    # def setChangingBC():
    #     def TBC(lbm:lb3D_PYRO,t:float):
    #         # 先稳定流场再升温，防止数值扰动
    #         if t<=10:
    #             lbm.TF.s_BC[0] = T_init
    #             lbm.TF.s_BC[2] = T_init
    #             lbm.TF.s_BC[3] = T_init
    #         else:
    #             lbm.TF.s_BC[0] = T_exp
    #             lbm.TF.s_BC[2] = T_exp
    #             lbm.TF.s_BC[3] = T_exp
    #     # lb3D.UpdateBCfunc.append(TBC)
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
        for i in ti.grouped(lb3D.rho):
            wood += lb3D.species[1].S[i]
        return wood
    @ti.kernel
    def cal_allWood():
        allWood[None] = cal_wood()
        print("all wood:",allWood[None])
    
    def setVariables():
        def Tcenter(lbm:LBM3DSolver):
            T_center = lbm.TS.S[(int(lbm.nx/2),int(lbm.ny/2),int(lbm.nz/2))]
            return "Ts_center", T_center
        lb3D.GetVariableFunc.append(Tcenter)
        def Tsurface(lbm:LBM3DSolver):
            T_surface = lbm.TF.S[(int(lbm.nx/2+R/DX-1),int(lbm.ny/2),int(lbm.nz/2))]
            return "Tf_surface", T_surface
        lb3D.GetVariableFunc.append(Tsurface)
        @ti.kernel
        def cal_conversion():
            conv[None] = (allWood[None]-cal_wood())/allWood[None]
        def conversion(lbm:LBM3DSolver):
            cal_conversion()
            return "conversion", conv[None]
        lb3D.GetVariableFunc.append(conversion)
    setVariables()
    ## 初始化场 
    lb3D.init_field(lb3D.rho,1)
    m3d  = Mesh3D(lb3D.nx,lb3D.ny,lb3D.nz)
    m3d.CreateMesh3D_Sphere_Decimal(float(lb3D.nx)/2,float(lb3D.ny)/2,float(lb3D.nz)/2,R/DX)
    v,s = m3d.export_numpy()
    lb3D.init_field(lb3D.solid,v*0.4)
    lb3D.init_field(lb3D.TF.S,T_init)
    lb3D.init_field(lb3D.TS.S,T_init)
    lb3D.init_field(lb3D.TS.exchangeSurface,100)
    lb3D.init_field(lb3D.TS.exchangeCoef,1000)
    lb3D.init_field(lb3D.TS.emissivity,0.7)
    lb3D.init_specie("N2",1)
    lb3D.init_specie("wood(S)",s*400)
    lb3D.init_field(lb3D.TS.radiation_surface, s*0.6)# vague
    # 初始化lbm
    lb3D.init_simulation()
    lb3D.check_python()
    # cal_allWood() # 计算总木材质量
    total_iteration =   100
    export_interval = 1
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
            max_v = lb3D.get_max_v()
            min_T = lb3D.get_min_T()
            print(name,flush=True)
            print('----------Time between two outputs is %dh %dm %ds; elapsed time is %dh %dm %ds----------------------' %(h_diff, m_diff, s_diff,h_elap,m_elap,s_elap))
            print('The %dth iteration, Max Force = %f,  Min Temperature = %f\n\n ' %(iter, max_v,  min_T))  
            if max_v>1:
                print("diverge")
                break          
        if (iter%int(export_interval/DT)==0):
            if DEBUG:
                lb3D.export_VTK(f"debug_{name}_{variant}_{DX}",iter)
                # lb3D.export_variable(f"simulation_{name}_{int(variant)}_{nx}_{int(T_exp)}",iter)
            else:
                lb3D.export_VTK(f"simulation_{name}_{variant}_{DX}",iter)
        if (iter%int(measure_interval/DT)==0):
                lb3D.export_variable(f"simulation_{name}_{variant}_{DX}",iter)
        lb3D.step()


    profiler.print_kernel_profiler_info()
    # profiler.print_memory_profiler_info()
if __name__=="__main__":
    os.environ["ARCH"]="CPU"
    DX = float(sys.argv[1])
    DT = float(sys.argv[2])
    T_exp = float(sys.argv[3])
    variant = sys.argv[4]
    startTime = time.time()
    main(DX,DT,T_exp,variant)
    print("execution time: ", time.time()-startTime)
    