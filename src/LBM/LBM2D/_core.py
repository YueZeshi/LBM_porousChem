import taichi as ti
import numpy as np
from ..util.flag import *
from ._thermal import TemperatureFluid, TemperatureSolid
from ._chemical import Specie, Reaction, Reactions
from visualization_tool.PVD import PVDWriter


@ti.data_oriented
class LBM2D_BASE:
    """LBM2D基类 —— 二维LBM求解器核心。

    管理D2Q9格子、物理场、标量输运(D2Q5)以及碰撞模型选择。

    Parameters
    ----------
    X : float
        物理域X方向长度 (m)
    Y : float
        物理域Y方向长度 (m)
    Z : float
        物理域Z方向长度 (m) — 二维中代表"厚度"，仍保留用于3D接口兼容
    dx : float, default=0.001
        空间步长 (m)
    dt : float, default=0.001
        时间步长 (s)
    name : str, default="LBM"
        求解器名称，用于输出文件命名
    isThermal : bool, default=False
        启用温度/热传导模块
    isChemical : bool, default=False
        启用化学反应/物种输运模块
    isPoro : bool, default=False
        启用多孔介质模型
    isRadiation : bool, default=False
        启用辐射模型
    collision_model : COLLISION_MODEL, default=COLLISION_MODEL.BGK
        碰撞模型类型:
        - BGK (0): 单松弛时间 Bhatnagar-Gross-Krook 模型
        - MRT (1): 多松弛时间模型（需要额外设置松弛率）

    Notes
    -----
    MRT相关场（s_mrt, tau_mrt, s_mrt_q5, tau_mrt_q5）仅在
    ``collision_model == COLLISION_MODEL.MRT`` 时分配，BGK模式下
    不占用显存。
    """
    def __init__(self, X, Y, Z, dx=0.001, dt=0.001, name="LBM", isThermal=False, isChemical=False, isPoro=False, isRadiation=False, collision_model=COLLISION_MODEL.BGK):
        self.name = name
        self.t = ti.field(float,shape=())
        self.tol = 1e-6
        # Model parameters
        self.X = float(X)
        self.Y = float(Y)
        self.Z = dx
        self.tLattice :int= 0
        self.dx = float(dx)
        self.dt = float(dt)

        self.nx = np.round(self.X / self.dx).astype(int)
        self.ny = np.round(self.Y / self.dx).astype(int)
        self.nz = 1
        self.max_v = ti.field(float, shape=())
        self.viscosity_model = VISCOSITY_MODEL.CONSTANT
        self.visco = ti.field(float,shape=())
        self.visco[None] = 2e-5
        self.sutherland_coef = [0.0,0.0] #ti.field(float,shape=(2)) # field性能不如python
        self.sutherland_coef[0],self.sutherland_coef[1] = 1.6e-6,170
        self.boundary_condition_model = BC_MODEL.NEE  # Only NEE implemented
        self.collision_model = collision_model
        self.EOS = FLUID_STATE_EQUATION.IDEAL_GAS

        # LBM constants
        self.e9 = ti.Vector.field(3, ti.i32, shape=(9))  # Direction vectors
        self.w9 = ti.field(float, shape=(9))  # Weights
        self.LR = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # Opposite indices for D2Q9
        self.e5 = ti.Vector.field(3, ti.i32, shape=(5))  # Direction vectors for D2Q5
        self.w5 = ti.field(float, shape=(5))  # Weights

        # Grid information
        self.x = np.linspace(0, self.X, self.nx)
        self.y = np.linspace(0, self.Y, self.ny)
        self.z = np.linspace(0, self.Z, self.nz)
        self.meshX, self.meshY, self.meshZ = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # 声明物理场
        
        self.rho = ti.field(float, shape=(self.nx,self.ny,self.nz))
        self.drho = ti.field(float, shape=(self.nx,self.ny,self.nz))
        self.v = ti.Vector.field(3,float, shape=(self.nx,self.ny,self.nz))
        self.solid = ti.field(float,shape = (self.nx,self.ny,self.nz))
        
        self.rhos = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.rhos0 = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.f = ti.Vector.field(9,float,shape=(self.nx,self.ny,self.nz)) # 分布函数

        self.ext_f = ti.Vector.field(3, float, shape=())  # External force
        # Define boundary conditions
        self.bc = [BC_FLOW.periodic]*4  # Left, right, top, bottom boundary conditions
        self.bc_v = [BC.periodic]*4  # Velocity boundary conditions
        self.bc_rho = [BC.periodic]*4  # Density boundary conditions
        # Left, right, top, bottom boundary velocity/density/flow values
        self.v_BC = ti.Vector.field(3, float, shape=(4))
        self.rho_BC = ti.field(float, shape=(4))
        self.flow_BC = ti.field(float, shape=(4))
        # Boundary velocity profiles
        self.v_bc_profile = [ti.Vector.field(3, float, shape=(1,self.ny,self.nz)),
                             ti.Vector.field(3, float, shape=(1,self.ny,self.nz)),
                             ti.Vector.field(3, float, shape=(self.nx,1,self.nz)),
                             ti.Vector.field(3, float, shape=(self.nx,1,self.nz)),]
        # Boundary density profiles
        self.rho_bc_profile = [ti.field(float, shape=(1,self.ny,self.nz)),
                               ti.field(float, shape=(1,self.ny,self.nz)),
                               ti.field(float, shape=(self.nx,1,self.nz)),
                               ti.field(float, shape=(self.nx,1,self.nz)),] 
        # Boundary names
        self.sideName = ["left","right","bottom","top"]
        self.UpdateBCfunc:list[callable] = []  # Variable boundary conditions
        self.GetVariableFunc:list[callable] = []  # Functions to get observables
        
        # MRT collision model fields (only allocated when MRT is selected)
        if self.collision_model == COLLISION_MODEL.MRT:
            self.M9 = ti.field(dtype=ti.f32, shape=(9, 9))  # D2Q9 MRT变换矩阵
            self.invM9 = ti.field(dtype=ti.f32, shape=(9, 9))  # D2Q9 MRT逆变换矩阵
            self.M5 = ti.field(dtype=ti.f32, shape=(5, 5))  # D2Q5 MRT变换矩阵
            self.invM5 = ti.field(dtype=ti.f32, shape=(5, 5))  # D2Q5 MRT逆变换矩阵
            self.s_mrt = ti.field(float, shape=(9))       # Relaxation rates s0-s8 for D2Q9 except s7,s8 which are variable
            self.tau_mrt = ti.field(float, shape=(9))     # Relaxation times for D2Q9 except s7,s8 which are variable
            # D2Q5 MRT for scalar transport (temperature, species)
            self.s_mrt_q5 = ti.field(float, shape=(5))    # Relaxation rates for D2Q5 except s1,s2 which are variable
            self.tau_mrt_q5 = ti.field(float, shape=(5))  # Relaxation times for D2Q5 except s1,s2 which are variable
        
        # Define scalar fields
        self.TEMPERATURE = isThermal
        self.CHEMISTRY = isChemical
        self.PORO = isPoro
        self.RADIATION = isRadiation and isThermal
        # Esoteric Twist (ET) Single array algorithm even-odd step marker: 0=even step, 1=odd step
        self.even_step = ti.field(dtype=ti.u1, shape=())
        if self.TEMPERATURE:
            self.TF_delay = ti.field(float,shape=())  # Temperature field delay time steps
            self.TS_delay = ti.field(float,shape=())  # Temperature field delay time steps
            self.TF = TemperatureFluid("Temperature of Fluid",self)
            self.TS = TemperatureSolid("Temperature of Solid",self,isRadiation = self.RADIATION)
            self.min_T = ti.field(float,shape=())
            self.max_T = ti.field(float,shape=())
        if self.CHEMISTRY:
            self.chemistry_field_delay = ti.field(float,shape=())  # Chemical reaction field delay time steps

            self.specieName = []
            self.species:list[Specie] = []
            self.reactions = Reactions(self) 
            self.inertSpecie = None # 惰性物质
        if self.PORO:        
            self.poro_model = PORO_MODEL.SPHERICAL  # Porous medium model
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
        # Initialize ET step marker
        self.even_step[None] = False

    # Built-in functions
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
    def static_init_kernel(self):  # Initialize static variables
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
