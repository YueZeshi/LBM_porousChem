import time
import taichi as ti
from taichi import profiler
import os
# 指定求解器
from solver.LBM_2D_pyrolysis import *
from geo.utils import *

def main(DX,DT,T_exp,n):
    # 获取环境变量是否启用debug模式
    # debug模式计算较少的步数 用于检验算例是否快速发散
    DEBUG = os.getenv("DEBUG","False").lower() == "true" # 默认非debug
    ARCH = os.getenv("ARCH","cpu").lower() # 默认CPU
    ## 用于获取运算时间信息
    time_init = time.time()
    time_now = time.time()
    time_pre = time.time() 

    # GLOBAL VARIABLE
    # 定义计算域 SI
    X = 0.2
    Y = 0.1
    R = 0.011

    ## convert to lattice unit
    nx = int(X/DX)
    ny = int(Y/DX)
    r = R/DX

    T_init = 303.

    os.makedirs("temp",exist_ok=True)
    # 定义域
    solid_file = "temp/solid.dat" # porosity
    biomass_file = "temp/biomass.dat"
    radiation_surface_file = "temp/radiation_surface.dat"
    with open(solid_file,"w") as sfile: # 先y后x
        with open(biomass_file,"w") as bfile:
            with open(radiation_surface_file,"w") as rsfile:
                for j in range(ny):
                    for i in range(nx):
                        s,l = Geo2D.circle(nx/2,ny/2,r,i,j)
                        sfile.write(f"{0.4*s}\n")
                        bfile.write(f"{500.0*s}\n")
                        rsfile.write(f"{l}\n") 
    print("executing ",__name__)

    name = "pyrolysis_two_reaction"
    variant = n

    # 初始化taichi
    ## arch=ti.cpu 启用cpu计算；arch=ti.gpu启用gpu运算 (cuda>vulkan)
    if ARCH=="gpu":
        ti.init(arch=ti.gpu, kernel_profiler=True, print_ir=False)
    else:
        ti.init(arch=ti.cpu, kernel_profiler=True, print_ir=False)

    # 初始化lbm模型
    lb2d = LB2D_PYRO(nx,ny,dx=DX,dt=DT,isPoro=True,isChemical=True,isThermal=True,isRadiation=True)

    # 基础设置
    lb2d.source_term_model = SOURCE_TERM.MICRO
    lb2d.force_term_model = FORCE_TERM.MACRO
    lb2d.set_poro_Darcy(2.5e10,unit="SI")
    lb2d.set_radiation_model(RADIATION_MODEL.SURFACE_UNIFORM,T_exp)

    # 设置物质
    ## 物种及其状态
    lb2d.set_specie("N2",False)
    lb2d.set_species(["wood(S)","intermSolid(S)","tar" ,"gas","char(S)"],
                 [True     ,     True       ,False ,False,True     ])
    # 设置边界条件
    lb2d.set_BCs([BC_V.FIXVALUE,BC_V.FIXPRESSURE,BC_V.NOSLIP,BC_V.NOSLIP])
    lb2d.set_v_BCs_value([[0.01,0,0],[0,0,0],[0,0,0],[0,0,0]],unit="SI")
    lb2d.set_rho_BCs_value([0.5]*4)
    lb2d.set_TF_BCs([BC_S.FIXVALUE,BC_S.OPEN,BC_S.FIXVALUE,BC_S.FIXVALUE])
    lb2d.set_TF_BCs_value([T_exp]*4)
    lb2d.set_species_BCs([BC_S.FIXVALUE,BC_S.OPEN,BC_S.WALL,BC_S.WALL])
    lb2d.set_specie_BCs_value("N2",[1]*4)

    # 添加化学反应
    gamma = 0.65
    lb2d.add_reaction("w2t",[("wood(S)",1)],[("tar",1)],(2e10,0,148000,200,110000))
    lb2d.add_reaction("w2is",[("wood(S)",1)],[("intermSolid(S)",1)],(2.51e7,0,111700,200,0))
    lb2d.add_reaction("is2c",[("intermSolid(S)",1)],[("char(S)",gamma),("gas",1-gamma)],(1.38e10,0,161000,200,-121000))

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
    lb2d.set_specie_capacity_func("gas",syngas_capacity)
    lb2d.set_specie_capacity("N2",1040)
    ## 热导
    lb2d.set_specie_conductivity("intermSolid(S)", 0.20487)
    lb2d.set_specie_conductivity("wood(S)", 0.20487)
    lb2d.set_specie_conductivity("char(S)", 0.0937)
    lb2d.set_specie_conductivity("tar", 0.0258)
    lb2d.set_specie_conductivity("gas", 0.0258)
    lb2d.set_specie_conductivity("N2",0.0258)

    ## 杂项
    ## 可变边界条件
    def setChangingBC():
        def TBC(lbm:LB2D_PYRO,t:float):
            # 先稳定流场再升温，防止数值扰动
            if t<=10:
                lbm.TF.s_BC[0] = T_init
                lbm.TF.s_BC[2] = T_init
                lbm.TF.s_BC[3] = T_init
            else:
                lbm.TF.s_BC[0] = T_exp
                lbm.TF.s_BC[2] = T_exp
                lbm.TF.s_BC[3] = T_exp
        # lb2d.UpdateBCfunc.append(TBC)
    setChangingBC()
    # 导出局部或者全局变量 观测量
    ## 中心温度 表面温度 转化率 
    ### 定义变量实现CPU GPU数据传输
    allWood = ti.field(ti.f32,shape=())
    conv = ti.field(ti.f32,shape=())
    @ti.func
    def cal_wood():
        wood = 0.0
        for i in ti.grouped(lb2d.rho):
            wood += lb2d.species["wood(S)"].S[i]
        return wood
    @ti.kernel
    def cal_allWood():
        allWood[None] = cal_wood()

    def setVariables():
        def Tcenter(lbm:LB2D_PYRO):
            T_center = lbm.TS.S[(int(lbm.nx/2),int(lbm.ny/2),int(lbm.nz/2))]
            return "Ts_center", T_center
        lb2d.GetVariableFunc.append(Tcenter)
        def Tsurface(lbm:LB2D_PYRO):
            T_surface = lbm.TF.S[(int(lbm.nx/2+r-1),int(lbm.ny/2),int(lbm.nz/2))]
            return "Tf_surface", T_surface
        lb2d.GetVariableFunc.append(Tsurface)
        @ti.kernel
        def cal_conversion():
            conv[None] = (allWood[None]-cal_wood())/allWood[None]
        def conversion(lbm:LB2D_PYRO):
            cal_conversion()
            return "conversion", conv[None]
        lb2d.GetVariableFunc.append(conversion)
    setVariables()
    ## 初始化场 
    lb2d.init_field(lb2d.rho,0.5)
    lb2d.init_field(lb2d.solid,solid_file)
    lb2d.init_field(lb2d.TF.S,T_init)
    lb2d.init_field(lb2d.TS.S,T_init)

    lb2d.init_specie("N2",1)
    lb2d.init_specie("wood(S)",biomass_file)
    lb2d.init_field(lb2d.radiation_surface,radiation_surface_file)# vague
    lb2d.init_field(lb2d.heat_trasnfer_surface,1) # vague
    # 初始化lbm
    lb2d.init_simulation()

    cal_allWood() # 计算总木材质量

    total_iteration = 1000
    export_interval = 10
    print_interval = 100
    if DEBUG:
        total_iteration = 2
        export_interval = 0.1
        print_interval = 1

    for iter in range(int(total_iteration/DT)+1):
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
                lb2d.export_VTK(f"debug_{name}_{int(variant)}_{nx}_{int(T_exp)}",iter)
                lb2d.check()
            else:
                lb2d.export_VTK(f"simulation_{name}_{int(variant)}_{nx}_{int(T_exp)}",iter)
                lb2d.export_variable(f"simulation_{name}_{int(variant)}_{nx}_{int(T_exp)}",iter)
        lb2d.step()            

    profiler.print_kernel_profiler_info()