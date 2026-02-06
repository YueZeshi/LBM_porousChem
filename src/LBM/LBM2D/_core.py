import taichi as ti
import numpy as np
from ..util.flag import *
from ._thermal import TemperatureFluid,TemperatureSolid
from ._chemical import Specie,Reaction,Reactions
from visualization_tool.PVD import PVDWriter
@ti.data_oriented
class LBM2D_BASE:
    """LBM2D base class
    """
    def __init__(self, X, Y,Z,dx = 0.001,dt = 0.001,name="LBM",isThermal = False,isChemical = False,isPoro = False,isRadiation = False):
        self.name = name
        self.t = 0.0
        self.tol = 1e-6
        # 模型参数
        self.X = X
        self.Y = Y
        self.Z = dx
        self.tLattice : int = 0
        self.dx = float(dx) # 格子尺度 
        self.dt = float(dt) # 步进时间
        self.nx = np.round(self.X/self.dx).astype(int)
        self.ny = np.round(self.Y/self.dx).astype(int)
        self.nz = 1 
        self.max_v=ti.field(float,shape=())
        self.viscosity_model = VISCOSITY_MODEL.NONE
        self.visco = 2e-5
        self.sutherland_coef = [1.6e-6,170]
        self.boundary_condition_model = BC_MODEL.NEE # only NEE implemented
        self.EOS = FLUID_STATE_EQUATION.IDEAL_GAS
        # LBM使用常量
        self.e9 = ti.Vector.field(3,ti.i32, shape=(9)) # e 方向向量
        self.w9 = ti.field(float, shape=(9)) # 权重
        self.LR = [0,3,4,1,2,7,8,5,6] #对称索引
        self.e5 = ti.Vector.field(3,ti.i32, shape=(5)) # e 方向向量
        self.w5 = ti.field(float, shape=(5)) # 权重
        self.x = np.linspace(0, self.X, self.nx)
        self.y = np.linspace(0, self.Y, self.ny)
        self.z = np.linspace(0, self.Z, self.nz)
        self.ext_f = ti.Vector.field(3,float,shape=()) # 外部力
        
        self.meshX, self.meshY, self.meshZ = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # 声明物理场
        self.rho = ti.field(float, shape=(self.nx,self.ny,self.nz))
        self.v = ti.Vector.field(3,float, shape=(self.nx,self.ny,self.nz))
        self.solid = ti.field(float,shape = (self.nx,self.ny,self.nz))
        
        self.rhos = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.f = ti.Vector.field(9,float,shape=(self.nx,self.ny,self.nz)) # 分布函数

        # 定义边界条件
        self.bc = [BC_FLOW.periodic]*4 # 左右上下边界条件类型
        self.bc_v = [BC.periodic]*4 # 左右上下速度边界条件类型
        self.bc_rho = [BC.periodic]*4 # 左右上下密度边界条件类型
        # 左右上下边界速度值/密度值/流量值
        self.v_BC = ti.Vector.field(3,float,shape = (4))
        self.rho_BC = ti.field(float,shape = (4))
        self.flow_BC = ti.field(float,shape = (4))
        # 边界速度剖面
        self.v_bc_profile = [ti.Vector.field(3,float,shape = (1,self.ny,self.nz)),
                             ti.Vector.field(3,float,shape = (1,self.ny,self.nz)),
                             ti.Vector.field(3,float,shape = (self.nx,1,self.nz)),
                             ti.Vector.field(3,float,shape = (self.nx,1,self.nz)),]
        # 边界密度剖面
        self.rho_bc_profile = [ti.field(float,shape = (1,self.ny,self.nz)),
                               ti.field(float,shape = (1,self.ny,self.nz)),
                               ti.field(float,shape = (self.nx,1,self.nz)),
                               ti.field(float,shape = (self.nx,1,self.nz)),] 
        # 边界名称
        self.sideName = ["left","right","bottom","top"]
        self.UpdateBCfunc:list[callable] = [] # 可变边界条件
        self.GetVariableFunc:list[callable] = [] # 获取观测量的函数 
        
        # 定义标量场
        self.TEMPERATURE= isThermal
        self.CHEMISTRY = isChemical
        self.PORO = isPoro
        self.RADIATION = isRadiation and isThermal
        # Esoteric Twist (ET) 单数组算法奇偶步标记：0=偶数步, 1=奇数步
        self.even_step = ti.field(dtype=ti.i32, shape=())
        if self.TEMPERATURE:
            self.TF = TemperatureFluid("Temperature of Fluid",self)
            self.TS = TemperatureSolid("Temperature of Solid",self,isRadiation = self.RADIATION)
            self.min_T = ti.field(float,shape=())
            self.max_T = ti.field(float,shape=())
        if self.CHEMISTRY:
            self.specieName = []
            self.species:list[Specie] = []
            self.reactions = Reactions(self) 
        if self.PORO:        
            self.poro_model = PORO_MODEL.SPHERICAL # 使用的多孔介质模型 如球孔介质模型 Darcy Darcy-Forhheimer
            self.coefDarcy = ti.field(float,shape=(self.nx,self.ny,self.nz))
            self.coefForchheimer = ti.field(float,shape=(self.nx,self.ny,self.nz))
        # default init all fields
        self.default_init()
        self.static_init_kernel()
        if self.TEMPERATURE:
            self.TS.default_init()
            self.TF.default_init()
        if self.CHEMISTRY:
            for specie in self.species:
                specie.default_init()
                
        self.PVD = PVDWriter(name=self.name)
        self.exportPath = "result"
        self.snapshotPath = "snapshot.yaml"
        # 初始化ET步进标记
        self.even_step[None] = 0
    # 内置函数
    def __repr__(self):
        return self.__str__()
    def __call__(self):
        print(self)
    @ti.func
    def tau(self,i):
        return 300
    @ti.func
    def viscosity(self,i):
        return 300
        
    @ti.func
    def GetTF(self,i):
        return 300
    @ti.func
    def GetTS(self,i):
        return 300
    def default_init(self):
        pass
    @ti.kernel
    def static_init_kernel(self): # 初始化静态变量
        pass
    @ti.func
    def feq9(self,s,i,j,k):
        pass
    @ti.func
    def feq9_no_poro(self,s,i,j,k):
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
    @ti.func
    def Boundary_condition_NEE_AA(self):
        pass
    def updateBC(self,t):
        pass