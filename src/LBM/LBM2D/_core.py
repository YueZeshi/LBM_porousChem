import taichi as ti
import numpy as np
from ..util.flag import *
from ._thermal import TemperatureFluid, TemperatureSolid
from ._chemical import Specie, Reaction, Reactions
from visualization_tool.PVD import PVDWriter

# ========= MRT 变换矩阵 (taichi 静态矩阵, 用于 @ 矩阵乘法) =========
# D2Q9: M (矩变换), invM (逆变换)
M9 = ti.Matrix.rows([
    [1,  1,  1,  1,  1,  1,  1,  1,  1],
    [-4, -1, -1, -1, -1,  2,  2,  2,  2],
    [4, -2, -2, -2, -2,  1,  1,  1,  1],
    [0,  1,  0, -1,  0,  1, -1, -1,  1],
    [0, -2,  0,  2,  0,  1, -1, -1,  1],
    [0,  0,  1,  0, -1,  1,  1, -1, -1],
    [0,  0, -2,  0,  2,  1,  1, -1, -1],
    [0,  1, -1,  1, -1,  1,  1,  1,  1],
    [0,  0,  0,  0,  0,  1, -1,  1, -1]
])

invM9 = ti.Matrix.rows([
    [1/9, -1/9,  1/9,    0,    0,     0,     0,   0,   0],
    [  0, -1/12,-1/12, 1/6,-1/6,     0,     0, 1/4,   0],
    [2/9, 1/36,-1/36,   0,    0,   1/6,  -1/6,-1/4,   0],
    [  0, -1/12,-1/12,-1/6, 1/6,     0,     0, 1/4,   0],
    [2/9, 1/36,-1/36,   0,    0,  -1/6,   1/6,-1/4,   0],
    [1/9, 1/18, 1/36, 1/6, 1/12,  1/6,  1/12,   0, 1/4],
    [1/9, 1/18, 1/36,-1/6,-1/12,  1/6,  1/12,   0,-1/4],
    [1/9, 1/18, 1/36,-1/6,-1/12, -1/6, -1/12,   0, 1/4],
    [1/9, 1/18, 1/36, 1/6, 1/12, -1/6, -1/12,   0,-1/4]
])

# D2Q5: M5 (矩变换), invM5 (逆变换, 标量输运)
M5 = ti.Matrix.rows([
    [1,  1,  1,  1,  1],
    [0,  1,  0, -1,  0],
    [0,  0,  1,  0, -1],
    [-4, 1,  1,  1,  1],
    [0,  1, -1,  1, -1]
])

invM5 = ti.Matrix.rows([
    [1/5,   0,   0,-1/5,   0],
    [1/5, 1/2,   0, 1/20, 1/4],
    [1/5,   0, 1/2, 1/20,-1/4],
    [1/5,-1/2,   0, 1/20, 1/4],
    [1/5,   0,-1/2, 1/20,-1/4]
])

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
        self.viscosity_model = VISCOSITY_MODEL.NONE
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
            self.s_mrt = ti.field(float, shape=(9))       # Relaxation rates s0-s8 for D2Q9
            self.tau_mrt = ti.field(float, shape=(9))     # Relaxation times for D2Q9 (τ_i = 1/s_i)
            # D2Q5 MRT for scalar transport (temperature, species)
            self.s_mrt_q5 = ti.field(float, shape=(5))    # Relaxation rates for D2Q5
            self.tau_mrt_q5 = ti.field(float, shape=(5))  # Relaxation times for D2Q5
        
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
    
    # ========= MRT Collision Operators =========
    @ti.func
    def mrt_transform_D2Q9(self, f):
        """D2Q9 分布函数 → 矩空间变换 m = M @ f
        使用已验证的M矩阵：m0=ρ, m1=e, m2=ε, m3=jx, m4=qx, m5=jy, m6=qy, m7=pxx, m8=pxy
        """
        return M9 @ f
    
    @ti.func
    def invmrt_transform_D2Q9(self, m):
        """矩空间 → D2Q9 分布函数逆变换 f = invM @ m"""
        return invM9 @ m
    
    @ti.func
    def mrt_eq_moments_D2Q9(self, rho, u, v):
        """计算 D2Q9 矩空间平衡态 m_eq
        u: x方向速度分量, v: y方向速度分量
        """
        usqr = u*u + v*v
        m_eq = ti.Vector([0.0] * 9)
        m_eq[0] = rho                           # m0 = ρ (密度)
        m_eq[1] = rho * (-2.0 + 3.0 * usqr)     # m1 = e
        m_eq[2] = rho * (1.0 - 3.0 * usqr)      # m2 = ε
        m_eq[3] = rho * u                       # m3 = jx
        m_eq[4] = rho * (-u)                    # m4 = qx
        m_eq[5] = rho * v                       # m5 = jy
        m_eq[6] = rho * (-v)                    # m6 = qy
        m_eq[7] = rho * (u*u - v*v)             # m7 = pxx
        m_eq[8] = rho * (u*v)                   # m8 = pxy
        return m_eq
    
    @ti.func
    def collision_MRT_D2Q9(self, f, rho, u, v):
        """D2Q9 MRT 碰撞算子：f* = invM @ (I - S) @ M @ f + invM @ S @ m_eq
        
        返回碰撞后的分布函数
        """
        m = self.mrt_transform_D2Q9(f)
        m_eq = self.mrt_eq_moments_D2Q9(rho, u, v)
        
        # 矩空间松弛: m* = m - s*(m - m_eq)
        # 其中 s[0..8] 对应松弛率，守恒矩(m0,m3,m5)松弛率为0
        for q in ti.static(range(9)):
            m[q] = m[q] - self.s_mrt[q] * (m[q] - m_eq[q])
        
        # 逆变换回分布函数
        return self.invmrt_transform_D2Q9(m)
    
    @ti.func
    def mrt_transform_D2Q5(self, g):
        """D2Q5 分布函数 → 矩空间变换 (用于标量输运)"""
        return M5 @ g
    
    @ti.func
    def invmrt_transform_D2Q5(self, m):
        """矩空间 → D2Q5 分布函数逆变换"""
        return invM5 @ m
    
    @ti.func
    def invmrt_transform_D2Q5(self, m):
        """矩空间 → D2Q5 分布函数逆变换"""
        inv5 = 0.2
        inv4 = 0.25
        f = ti.Vector([0.0] * 5)
        f[0] = inv5*m[0] - inv5*m[3]
        f[1] = inv5*m[0] + inv4*m[1] + 0.2*m[3] + inv4*m[4]
        f[2] = inv5*m[0] + inv4*m[2] + 0.2*m[3] - inv4*m[4]
        f[3] = inv5*m[0] - inv4*m[1] + 0.2*m[3] + inv4*m[4]
        f[4] = inv5*m[0] - inv4*m[2] + 0.2*m[3] - inv4*m[4]
        return f
    
    @ti.func
    def mrt_eq_moments_D2Q5(self, scalar, ux, uy):
        """计算 D2Q5 矩空间平衡态"""
        m_eq = ti.Vector([0.0] * 5)
        m_eq[0] = scalar                                   # m0 = φ
        m_eq[1] = scalar * ux * (1.0/3.0)                  # m1 = jx_eq
        m_eq[2] = scalar * uy * (1.0/3.0)                  # m2 = jy_eq
        m_eq[3] = scalar * (-2.0/3.0)                      # m3 = e_eq
        m_eq[4] = 0.0                                       # m4 = ε_eq
        return m_eq
    
    @ti.func
    def collision_MRT_D2Q5(self, g, scalar, ux, uy):
        """D2Q5 MRT 碰撞算子 (用于标量输运如温度/组分)"""
        m = self.mrt_transform_D2Q5(g)
        m_eq = self.mrt_eq_moments_D2Q5(scalar, ux, uy)
        
        for q in ti.static(range(5)):
            m[q] = m[q] - self.s_mrt_q5[q] * (m[q] - m_eq[q])
        
        return self.invmrt_transform_D2Q5(m)
    
    def setup_mrt_rates(self, tau, tau_bulk=None, s_magic=1.63):
        """设置 D2Q9/D2Q5 MRT 松弛率 (仅 MRT 碰撞模型可用)。
        
        Args:
            tau: 剪切粘性松弛时间 (与BGK的τ相同)
            tau_bulk: 体粘性松弛时间 (默认与tau相同)
            s_magic: magic parameter for s1/s2 (default 1.63 for optimal stability)
        
        标准MRT参数:
            s0=0 (密度, 守恒)
            s1=s2  (能量模, 与体粘性相关)
            s3=s5=0 (动量, 守恒)
            s4, s6  (能量平方模)
            s7=s8=1/tau (剪切应力, 决定运动粘度)
            
        D2Q5 MRT参数 (标量输运):
            s0=0 (标量密度, 守恒)
            s1=s2=1/tau_Q (通量, 决定扩散系数)
            s3=s4 (高阶矩)
            
        Raises:
            RuntimeError: 如果碰撞模型不是 MRT
        """
        if self.collision_model != COLLISION_MODEL.MRT:
            raise RuntimeError(
                f"setup_mrt_rates() 仅对 MRT 碰撞模型可用, "
                f"当前模型: {self.collision_model.name}"
            )
        omega = 1.0 / tau  # BGK relaxation frequency
        if tau_bulk is None:
            tau_bulk = tau
        omega_bulk = 1.0 / tau_bulk
        
        # D2Q9 MRT 松弛率
        # s0 (密度): 守恒矩, 松弛率为1
        self.s_mrt[0] = 1.0
        # s1 (能量e): 与体粘性相关
        self.s_mrt[1] = omega_bulk
        # s2 (能量平方ε): 与体粘性相关
        self.s_mrt[2] = omega_bulk
        # s3 (jx): 守恒矩
        self.s_mrt[3] = 1.0
        # s4 (qx): 高阶矩, 使用magic参数
        self.s_mrt[4] = s_magic
        # s5 (jy): 守恒矩
        self.s_mrt[5] = 1.0
        # s6 (qy): 高阶矩
        self.s_mrt[6] = s_magic
        # s7 (pxx): 剪切应力, 决定运动粘度 ν = cs^2*(τ-0.5)*dt
        self.s_mrt[7] = omega
        # s8 (pxy): 剪切应力
        self.s_mrt[8] = omega
        
        # D2Q5 MRT 松弛率 (标量输运)
        # s0 (φ): 守恒矩
        self.s_mrt_q5[0] = 1.0
        # s1 (jx), s2 (jy): 通量, 决定扩散系数 D = cs^2*(τ-0.5)*dt
        self.s_mrt_q5[1] = omega
        self.s_mrt_q5[2] = omega
        # s3 (e), s4 (ε): 高阶矩
        self.s_mrt_q5[3] = s_magic
        self.s_mrt_q5[4] = s_magic