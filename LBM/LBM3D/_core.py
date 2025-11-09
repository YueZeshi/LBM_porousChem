import taichi as ti
import numpy as np
from util.flag import *
from ._thermal import TemperatureFluid,TemperatureSolid
from ._chemical import Specie,Reaction
@ti.data_oriented
class LBM3DSolver:
    """
    LBM3DSolver: core
    """
    def __init__(self, X, Y, Z,dx = 0.001,dt = 0.001,name="LBM",isThermal = False,isChemical = False,isPoro = False,isRadiation = False):
        self.name = name
        self.t = 0.0
        # 模型参数
        self.enable_projection = True
        self.X = X
        self.Y = Y
        self.Z = Z
        self.dx,self.dt = dx,dt #格子尺度 步进时间
        self.nx=int(self.X/self.dx)
        self.ny=int(self.Y/self.dx)
        self.nz = int(self.Z/self.dx) # 模型大小
        self.max_v=ti.field(ti.f32,shape=())
        self.source_term_model = SOURCE_TERM.NONE
        self.force_term_model = FORCE_TERM.NONE
        # LBM使用常量
        self.e19 = ti.Vector.field(3,ti.i32, shape=(19)) # e 方向向量
        self.w19 = ti.field(ti.f32, shape=(19)) # 权重
        self.LR = [0,2,1,4,3,6,5,10,9,8,7,14,13,12,11,18,17,16,15] #对称索引
        self.e7 = ti.Vector.field(3,ti.i32, shape=(7)) # e 方向向量
        self.w7 = ti.field(ti.f32, shape=(7)) # 权重
        # self.LR5 = [0,3,4,1,2] #对称索引
        self.x = np.linspace(0, self.nx, self.nx)
        self.y = np.linspace(0, self.ny, self.ny)
        self.z = np.linspace(0, self.nz, self.nz)
        self.ext_f = ti.Vector.field(3,ti.f32,shape=()) # 外部力
        
        #X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # 声明物理场
        self.rho = ti.field(ti.f32, shape=(self.nx,self.ny,self.nz))
        self.v = ti.Vector.field(3,ti.f32, shape=(self.nx,self.ny,self.nz))
        self.solid = ti.field(ti.f32,shape = (self.nx,self.ny,self.nz))
        self.f = ti.Vector.field(19,ti.f32,shape=(self.nx,self.ny,self.nz))
        self.F = ti.Vector.field(19,ti.f32,shape=(self.nx,self.ny,self.nz))

        # 定义边界条件
        self.bc = [BC_FLOW.periodic]*6
        self.bc_v = [BC.periodic]*6
        self.bc_rho = [BC.periodic]*6
        self.v_BC = ti.Vector.field(3,ti.f32,shape = (6))
        self.rho_BC = ti.field(ti.f32,shape = (6))
        self.sideName = ["left","right","front","back","bottom","top"]
        self.UpdateBCfunc = [] # 可变边界条件
        self.GetVariableFunc = [] # 获取观测量的函数
        
        # 定义标量场
        self.THERMAL = isThermal
        self.CHEMICAL = isChemical
        self.PORO = isPoro
        self.RADIATION = isRadiation and isThermal
        if self.THERMAL:
            self.TF = TemperatureFluid("Temperature of Fluid",self.nx,self.ny,self.nz,self)
            self.TS = TemperatureSolid("Temperature of Solid",self.nx,self.ny,self.nz,self)
            self.heat_trasnfer_surface = ti.field(ti.f32,shape = (self.nx,self.ny,self.nz))
            self.min_T = ti.field(ti.f32,shape=())
            if self.RADIATION:
                self.radiation_model = RADIATION_MODEL.NONE # 辐射模型
                # self.radiation = ti.field(ti.f32,shape=(self.nx,self.ny,self.nz))
                self.radiation_surface = ti.field(ti.f32,shape = (self.nx,self.ny,self.nz)) # S/V L-1
        if self.CHEMICAL:
            self.species = list[Specie]
            self.reactions = dict()
        if self.PORO:        
            self.poro_model = PORO_MODEL.SPHERICAL # 使用的多孔介质模型 如球孔介质模型 Darcy Darcy-Forhheimer
            self.coefDarcy = ti.field(ti.f32,shape=(self.nx,self.ny,self.nz))
            self.coefForchheimer = ti.field(ti.f32,shape=(self.nx,self.ny,self.nz))
            self.rhos = ti.field(ti.f32,shape=(self.nx,self.ny,self.nz))
            self.rho1 = ti.field(ti.f32,shape=(self.nx,self.ny,self.self.nz))
        # 默认初始化所有场
        self.default_init()    
    # 内置函数
    def __repr__(self):
        return self.__str__()
    def __call__(self):
        print(self)
    def __str__(self):
        return "多孔生物质热解LBM2D求解器。包含多孔介质达西定律，温度场，温度流场耦合单松弛因子LBM，可根据Darcy定律模拟多孔介质行为。可定义不同区域物性"

    