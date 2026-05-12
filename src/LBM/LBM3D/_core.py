from sympy import im
import taichi as ti
import numpy as np
from ..util.flag import *
from ._thermal import TemperatureFluid,TemperatureSolid
from ._chemical import Specie,Reaction,Reactions
from ._mrt_matrix_D3Q19 import M19, invM19
from visualization_tool.PVD import PVDWriter
@ti.data_oriented
class LBM3D_BASE:
    """
    LBM3D base class
    """
    def __init__(self, X, Y, Z,dx = 0.001,dt = 0.001,name="LBM",isThermal = False,isChemical = False,isPoro = False,isRadiation = False,collision_model=COLLISION_MODEL.BGK):
        self.name = name
        self.t = ti.field(float,shape=())
        self.tol = 1e-12
        # 模型参数
        self.X = float(X)
        self.Y = float(Y)
        self.Z = float(Z)
        self.tLattice : int = 0
        self.dx = float(dx) # 格子尺度 
        self.dt = float(dt) # 步进时间
        self.nx = np.round(self.X/self.dx).astype(int)
        self.ny = np.round(self.Y/self.dx).astype(int)
        self.nz = np.round(self.Z/self.dx).astype(int) # 模型大小
        self.max_v=ti.field(float,shape=())
        self.viscosity_model = VISCOSITY_MODEL.CONSTANT
        self.visco = 1e-5
        self.sutherland_coef = [1.6e-6,170]
        self.boundary_condition_model = BC_MODEL.NEE
        self.EOS = FLUID_STATE_EQUATION.IDEAL_GAS
        # LBM使用常量
        self.e19 = ti.Vector.field(3,int, shape=(19)) # e 方向向量
        self.w19 = ti.field(float, shape=(19)) # 权重
        self.LR = [0,2,1,4,3,6,5,10,9,8,7,14,13,12,11,18,17,16,15] #对称索引
        self.e7 = ti.Vector.field(3,int, shape=(7)) # e 方向向量
        self.w7 = ti.field(float, shape=(7)) # 权重
        self.x = np.linspace(0, self.X, self.nx)
        self.y = np.linspace(0, self.Y, self.ny)
        self.z = np.linspace(0, self.Z, self.nz)
        self.ext_f = ti.Vector.field(3,float,shape=()) # 外部力
        
        self.meshX, self.meshY, self.meshZ = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # 声明物理场
        self.rho = ti.field(float, shape=(self.nx,self.ny,self.nz))
        self.drho = ti.field(float, shape=(self.nx,self.ny,self.nz))
        self.v = ti.Vector.field(3,float, shape=(self.nx,self.ny,self.nz))
        self.solid = ti.field(float,shape = (self.nx,self.ny,self.nz))
        self.rhos = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.rhos0 = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.f = ti.Vector.field(19,float,shape=(self.nx,self.ny,self.nz)) # 分布函数
        self.F = ti.Vector.field(19,float,shape=(self.nx,self.ny,self.nz)) # 后碰撞分布函数
        # 定义边界条件
        self.bc = [BC_FLOW.periodic]*6 # 左右前后上下边界条件类型
        self.bc_v = [BC.periodic]*6 # 左右前后上下速度边界条件类型
        self.bc_rho = [BC.periodic]*6 # 左右前后上下密度边界条件类型
        # 左右前后上下边界速度值/密度值/流量值
        self.v_BC = ti.Vector.field(3,float,shape = (6))
        self.rho_BC = ti.field(float,shape = (6))
        self.flow_BC = ti.field(float,shape = (6))
        # 边界速度剖面
        self.v_bc_profile = [ti.Vector.field(3,float,shape = (1,self.ny,self.nz)),
                             ti.Vector.field(3,float,shape = (1,self.ny,self.nz)),
                             ti.Vector.field(3,float,shape = (self.nx,1,self.nz)),
                             ti.Vector.field(3,float,shape = (self.nx,1,self.nz)),
                             ti.Vector.field(3,float,shape = (self.nx,self.ny,1)),
                             ti.Vector.field(3,float,shape = (self.nx,self.ny,1)),]
        # 边界密度剖面
        self.rho_bc_profile = [ti.field(float,shape = (1,self.ny,self.nz)),
                               ti.field(float,shape = (1,self.ny,self.nz)),
                               ti.field(float,shape = (self.nx,1,self.nz)),
                               ti.field(float,shape = (self.nx,1,self.nz)),
                               ti.field(float,shape = (self.nx,self.ny,1)),
                               ti.field(float,shape = (self.nx,self.ny,1)),] 
        
        self.sideName = ["left","right","front","back","bottom","top"]
        self.UpdateBCfunc:list[callable] = [] # 可变边界条件
        self.GetVariableFunc:list[callable] = [] # 获取观测量的函数
        
        self.TEMPERATURE= isThermal
        self.CHEMISTRY = isChemical
        self.PORO = isPoro
        self.RADIATION = isRadiation and isThermal
        self.collision_model = collision_model
        # Esoteric Twist (ET) 单数组算法奇偶步标记：0=偶数步, 1=奇数步
        self.even_step = ti.field(dtype=ti.i32, shape=())
        if self.TEMPERATURE:
            self.TF_delay = ti.field(float,shape=()) # 温度场延迟时间步数
            self.TS_delay = ti.field(float,shape=()) # 温度场延迟时间步数
            self.TF:TemperatureFluid = TemperatureFluid("Temperature of Fluid",self)
            self.TS:TemperatureSolid = TemperatureSolid("Temperature of Solid",self,isRadiation = self.RADIATION)
            self.min_T = ti.field(float,shape=())
            self.max_T = ti.field(float,shape=())
        if self.CHEMISTRY:
            self.chemistry_field_delay = ti.field(float,shape=()) # 化学反应场延迟时间步数
            self.specieName = []
            self.species:list[Specie] = []
            self.reactions = Reactions(self) 
        if self.PORO:        
            self.poro_model = PORO_MODEL.SPHERICAL # 使用的多孔介质模型 如球孔介质模型 Darcy Darcy-Forhheimer
            self.coefDarcy = ti.field(float,shape=(self.nx,self.ny,self.nz))
            self.coefForchheimer = ti.field(float,shape=(self.nx,self.ny,self.nz))
        if self.collision_model == COLLISION_MODEL.MRT:
            self.s_mrt_D3Q19 = ti.field(float, shape=(19))
            # ★ Bugfix: 初始化MRT松弛率（参照LBM2D分类模式）
            #   守恒矩(ρ,jz,jy,jx)→s=0, 剪切矩(7模跨越5D迹零张量)→s=ω,
            #   Ghost矩(8个高阶模)→s=s_magic=1.63
            visco_lu = self.visco * self.dt / (self.dx ** 2)
            tau0 = 3.0 * visco_lu + 0.5
            omega = 1.0 / tau0
            S_MAGIC = 1.63
            s_np = np.full(19, S_MAGIC, dtype=np.float32)
            # 守恒矩 (s=0)
            s_np[0] = 0.0  # ρ
            s_np[1] = 0.0  # jz
            s_np[3] = 0.0  # jy
            s_np[9] = 0.0  # jx
            # 剪切矩 (s=omega, 7个矩跨越5D迹零二阶张量子空间)
            s_np[2] = omega   # cz²-based
            s_np[4] = omega   # σ_yz
            s_np[6] = omega   # cy²-based
            s_np[8] = omega   # cx²-based
            s_np[10] = omega  # σ_xz
            s_np[12] = omega  # σ_xy
            s_np[14] = omega  # cx²-based
            # 其余8个Ghost矩保持s_magic: m5,m7,m11,m13,m15,m16,m17,m18
            self.s_mrt_D3Q19.from_numpy(s_np)
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
        self.exportPath = "vtk"
        self.snapshotPath = "snapshot.yaml"
        # 初始化ET步进标记
        self.even_step[None] = 0
        # 初始化delay
        if self.TEMPERATURE:
            self.TF_delay = 0.0
            self.TS_delay = 0.0
        if self.CHEMISTRY:
            self.chemistry_field_delay = 0.0
    # 内置函数
    def __repr__(self):
        return self.__str__()
    def __call__(self):
        print(self)
    @ti.func
    def feq19(self,s,i,j,k):
        pass
    @ti.func
    def feq19_no_poro(self,s,i,j,k):
        pass
    @ti.kernel
    def default_init(self):
        pass
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
    @ti.kernel
    def static_init_kernel(self): # 初始化静态变量
        pass
    @ti.func
    def feq9(self,s,i,j,k):
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
    @ti.func
    def Boundary_condition_NEE_AA(self):
        pass
    
    @ti.func
    def setup_mrt_rates_D3Q19(self, tau, tau_bulk=None, s_magic=1.63):
        """
        设置 D3Q19 MRT 松弛率（参照LBM2D分类模式）
        守恒矩 (s=0): m0(ρ), m1(jz), m3(jy), m9(jx)
        剪切矩 (s=ω, 碰撞核实时用局部τ覆盖): m2,m4,m6,m8,m10,m12,m14
            (7个矩跨越5D迹零二阶张量子空间)
        Ghost矩 (s=s_magic): m5,m7,m11,m13,m15,m16,m17,m18 (8个高阶模)
        """
        omega = 1.0 / tau
        if tau_bulk is None:
            tau_bulk = tau
        omega_bulk = 1.0 / tau_bulk
        for i in ti.static(range(19)):
            self.s_mrt_D3Q19[i] = s_magic
        # 守恒矩 (s=0)
        self.s_mrt_D3Q19[0] = 0.0  # ρ
        self.s_mrt_D3Q19[1] = 0.0  # jz
        self.s_mrt_D3Q19[3] = 0.0  # jy
        self.s_mrt_D3Q19[9] = 0.0  # jx
        # 剪切矩 (s=omega)
        self.s_mrt_D3Q19[2] = omega    # cz²-based
        self.s_mrt_D3Q19[4] = omega    # σ_yz
        self.s_mrt_D3Q19[6] = omega    # cy²-based
        self.s_mrt_D3Q19[8] = omega    # cx²-based
        self.s_mrt_D3Q19[10] = omega   # σ_xz
        self.s_mrt_D3Q19[12] = omega   # σ_xy
        self.s_mrt_D3Q19[14] = omega   # cx²-based
        # 其余8个Ghost矩保留s_magic: m5,m7,m11,m13,m15,m16,m17,m18

    @ti.func
    def mrt_collide_D3Q19(self, f, feq, idx, tau):
        """
        D3Q19 MRT 碰撞: m' = m - S*(m - meq) → f' = invM19 @ m'
        使用预构建的正交矩阵 M19/invM19
        守恒矩(s=0)保持, 剪切矩(s=ω实时覆盖), Ghost矩(s=预设s_magic)
        
        tau: 局部松弛时间（支持非牛顿/变粘度流体实时ω覆盖）
        """
        omega_local = 1.0 / tau  # 实时局部ω, 覆盖剪切模
        # 1) 变换到矩空间: m = M19 @ f, meq = M19 @ feq
        m = ti.Vector([0.0 for _ in ti.static(range(19))])
        meq = ti.Vector([0.0 for _ in ti.static(range(19))])
        for i in ti.static(range(19)):
            m_sum = 0.0
            meq_sum = 0.0
            for j in ti.static(range(19)):
                m_sum += M19[i, j] * f[j]
                meq_sum += M19[i, j] * feq[j]
            m[i] = m_sum
            meq[i] = meq_sum
        # 2) 矩空间松弛: 分类使用松弛率
        m_relaxed = ti.Vector([0.0 for _ in ti.static(range(19))])
        for i in ti.static(range(19)):
            # 剪切模(7个): 用实时ω覆盖, 支持非牛顿变粘度
            if ti.static(i in (2, 4, 6, 8, 10, 12, 14)):
                s_i = omega_local
            else:
                s_i = self.s_mrt_D3Q19[i]
            m_relaxed[i] = m[i] - s_i * (m[i] - meq[i])
        # 3) 反变换回速度空间: f' = invM19 @ m_relaxed
        f_collided = ti.Vector([0.0 for _ in ti.static(range(19))])
        for i in ti.static(range(19)):
            f_sum = 0.0
            for j in ti.static(range(19)):
                f_sum += invM19[i, j] * m_relaxed[j]
            f_collided[i] = f_sum
        return f_collided

    def updateBC(self,t):
        pass