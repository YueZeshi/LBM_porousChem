import taichi as ti
import numpy as np
from ..util.flag import *
from ._thermal import TemperatureFluid,TemperatureSolid
from ._chemical import Specie,Reaction,Reactions
from ._info import INFO
@ti.data_oriented
class LBM3D_BASE:
    """
    LBM3D base class
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
        self.max_v=ti.field(float,shape=())
        self.source_term_model = SOURCE_TERM.NONE
        self.force_term_model = FORCE_TERM.NONE
        self.boundary_condition_model = BC_MODEL.NEE
        self.EOS = FLUID_STATE_EQUATION.IDEAL_GAS
        # LBM使用常量
        self.e19 = ti.Vector.field(3,int, shape=(19)) # e 方向向量
        self.w19 = ti.field(float, shape=(19)) # 权重
        self.LR = [0,2,1,4,3,6,5,10,9,8,7,14,13,12,11,18,17,16,15] #对称索引
        self.e7 = ti.Vector.field(3,int, shape=(7)) # e 方向向量
        self.w7 = ti.field(float, shape=(7)) # 权重
        # self.LR5 = [0,3,4,1,2] #对称索引
        self.x = np.linspace(0, self.X, self.nx)
        self.y = np.linspace(0, self.Y, self.ny)
        self.z = np.linspace(0, self.Z, self.nz)
        self.ext_f = ti.Vector.field(3,float,shape=()) # 外部力
        
        #X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # 声明物理场
        self.rho = ti.field(float, shape=(self.nx,self.ny,self.nz))
        self.v = ti.Vector.field(3,float, shape=(self.nx,self.ny,self.nz))
        self.solid = ti.field(float,shape = (self.nx,self.ny,self.nz))
        self.f = ti.Vector.field(19,float,shape=(self.nx,self.ny,self.nz))
        self.F = ti.Vector.field(19,float,shape=(self.nx,self.ny,self.nz))

        # 定义边界条件
        self.bc = [BC_FLOW.periodic]*6
        self.bc_v = [BC.periodic]*6
        self.bc_rho = [BC.periodic]*6
        self.v_BC = ti.Vector.field(3,float,shape = (6))
        self.rho_BC = ti.field(float,shape = (6))
        self.sideName = ["left","right","front","back","bottom","top"]
        self.UpdateBCfunc = [] # 可变边界条件
        self.GetVariableFunc = [] # 获取观测量的函数
        
        # 定义标量场
        self.TEMPERATURE= isThermal
        self.CHEMISTRY = isChemical
        self.PORO = isPoro
        self.RADIATION = isRadiation and isThermal
        if self.TEMPERATURE:
            self.TF:TemperatureFluid = TemperatureFluid("Temperature of Fluid",self.nx,self.ny,self.nz,self)
            self.TS:TemperatureSolid = TemperatureSolid("Temperature of Solid",self.nx,self.ny,self.nz,self,isRadiation = self.RADIATION)
            self.min_T = ti.field(float,shape=())
        if self.CHEMISTRY:
            self.specieName = []
            self.species:list[str,Specie] = []
            self.reactions = Reactions(self) 
        if self.PORO:        
            self.poro_model = PORO_MODEL.SPHERICAL # 使用的多孔介质模型 如球孔介质模型 Darcy Darcy-Forhheimer
            self.coefDarcy = ti.field(float,shape=(self.nx,self.ny,self.nz))
            self.coefForchheimer = ti.field(float,shape=(self.nx,self.ny,self.nz))
            self.rhos = ti.field(float,shape=(self.nx,self.ny,self.nz))
            self.rho1 = ti.field(float,shape=(self.nx,self.ny,self.nz))
        # default init all fields
        self.default_init()
        if self.TEMPERATURE:
            self.TS.default_init()
            self.TF.default_init()
        if self.CHEMISTRY:
            for specie in self.species:
                specie.default_init()
    # 内置函数
    def __repr__(self):
        return self.__str__()
    def __call__(self):
        print(self)
    def __str__(self):
        return INFO
    @ti.func
    def feq19(self,s,i,j,k):
        pass
    def default_init(self):
        pass

    @ti.func
    def Boundary_condition(self):
        pass
    @ti.func
    def Boundary_condition_NEBB(self):
        pass
        
    @ti.func
    def Boundary_condition_NEE(self):
        pass
    @ti.func 
    def Boundary_condition_ES(self):
        pass
    