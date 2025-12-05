from typing import Literal
import taichi as ti
import numpy as np
from ..util.flag import *
from ._thermal import TemperatureFluid,TemperatureSolid
from ._chemical import Specie,Reaction,Reactions
from ._info import INFO
@ti.data_oriented
class LBM2D_BASE:
    def __init__(self, X, Y ,dx = 0.001,dt = 0.001,name="LBM",isThermal = False,isChemical = False,isPoro = False,isRadiation = False):
        self.name = name
        self.t = 0.0
        # 模型参数
        self.X = X
        self.Y = Y
        self.t = 0
        self.dx,self.dt = dx,dt #格子尺度 步进时间
        self.nx=int(self.X/self.dx)
        self.ny=int(self.Y/self.dx)
        self.nz = 1 
        self.max_v=ti.field(float,shape=())
        self.source_term_model = SOURCE_TERM.NONE
        self.force_term_model = FORCE_TERM.NONE
        self.boundary_condition_model = BC_MODEL.NEE
        self.EOS = FLUID_STATE_EQUATION.IDEAL_GAS
        # LBM使用常量
        self.e9 = ti.Vector.field(3,ti.i32, shape=(9)) # e 方向向量
        self.w9 = ti.field(float, shape=(9)) # 权重
        self.LR = [0,3,4,1,2,7,8,5,6] #对称索引
        self.e5 = ti.Vector.field(3,ti.i32, shape=(5)) # e 方向向量
        self.w5 = ti.field(float, shape=(5)) # 权重
        self.x = np.linspace(0, self.nx, self.nx)
        self.y = np.linspace(0, self.ny, self.ny)
        self.z = np.linspace(0, self.nz, self.nz)
        self.ext_f = ti.Vector.field(3,float,shape=()) # 外部力
        
        #X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # 声明物理场
        self.rho = ti.field(float, shape=(self.nx,self.ny,self.nz))
        self.v = ti.Vector.field(3,float, shape=(self.nx,self.ny,self.nz))
        self.solid = ti.field(float,shape = (self.nx,self.ny,self.nz))
        self.f = ti.Vector.field(9,float,shape=(self.nx,self.ny,self.nz))
        self.F = ti.Vector.field(9,float,shape=(self.nx,self.ny,self.nz))

        # 定义边界条件
        self.bc = [BC_FLOW.periodic]*4
        self.bc_v = [BC.periodic]*4
        self.bc_rho = [BC.periodic]*4
        self.v_BC = ti.Vector.field(3,float,shape = (4))
        self.rho_BC = ti.field(float,shape = (4))
        self.sideName = ["left","right","bottom","top"]
        self.UpdateBCfunc = [] # 可变边界条件
        self.GetVariableFunc = [] # 获取观测量的函数
        
        # 定义标量场
        self.TEMPERATURE= isThermal
        self.CHEMISTRY = isChemical
        self.PORO = isPoro
        self.RADIATION = isRadiation and isThermal
        if self.TEMPERATURE:
            self.TF = TemperatureFluid("Temperature of Fluid",self.nx,self.ny,self.nz,self)
            self.TS = TemperatureSolid("Temperature of Solid",self.nx,self.ny,self.nz,self,isRadiation = self.RADIATION)
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
    def feq9(self,s,i,j,k):
        pass
    @ti.kernel
    def Boundary_condition(self):
        pass
    @ti.kernel
    def Boundary_condition_NEBB(self):
        pass
        
    @ti.kernel
    def Boundary_condition_NEE(self):
        pass
    @ti.kernel 
    def Boundary_condition_ES(self):
        pass