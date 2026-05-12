import taichi as ti
import os
from ..util.flag import *
from ._core import LBM2D_BASE
@ti.data_oriented
class LBM2D_INITIALIZATION(LBM2D_BASE):
    # 初始化
    @ti.kernel
    def default_init(self): # 创建时初始化
        for i in ti.ndrange(4):
            self.rho_BC[i] = 1.0
            self.v_BC[i] = ti.Vector([0,0,0])
        for i in ti.grouped(self.solid):
            self.solid[i] = 0.0
            self.v[i] = ti.Vector([0,0,0])
            self.rho[i] = 1.0
    def init_simulation(self):
        self.init_python()
        self.init_kernel()
    def init_python(self):
        if self.collision_model == COLLISION_MODEL.MRT:
            import numpy as np
            # 用ti.field代替ti.Matrix.rows避免9×9矩阵(81 entries)编译展开警告
            self.M9.from_numpy(np.array([
                [1,  1,  1,  1,  1,  1,  1,  1,  1],
                [-4, -1, -1, -1, -1,  2,  2,  2,  2],
                [4, -2, -2, -2, -2,  1,  1,  1,  1],
                [0,  1,  0, -1,  0,  1, -1, -1,  1],
                [0, -2,  0,  2,  0,  1, -1, -1,  1],
                [0,  0,  1,  0, -1,  1,  1, -1, -1],
                [0,  0, -2,  0,  2,  1,  1, -1, -1],
                [0,  1, -1,  1, -1,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  1, -1,  1, -1]
            ], dtype=np.float32))
            self.invM9 = ti.field(dtype=ti.f32, shape=(9, 9))
            self.invM9.from_numpy(np.array([
                [1/9, -1/9,  1/9,    0,    0,     0,     0,   0,   0],
                [1/9,-1/36,-1/18, 1/6,-1/6,     0,     0, 1/4,   0],
                [1/9,-1/36,-1/18,   0,    0,   1/6,  -1/6,-1/4,   0],
                [1/9,-1/36,-1/18,-1/6, 1/6,     0,     0, 1/4,   0],
                [1/9,-1/36,-1/18,   0,    0,  -1/6,   1/6,-1/4,   0],
                [1/9, 1/18, 1/36, 1/6, 1/12,  1/6,  1/12,   0, 1/4],
                [1/9, 1/18, 1/36,-1/6,-1/12,  1/6,  1/12,   0,-1/4],
                [1/9, 1/18, 1/36,-1/6,-1/12, -1/6, -1/12,   0, 1/4],
                [1/9, 1/18, 1/36, 1/6, 1/12, -1/6, -1/12,   0,-1/4]
            ], dtype=np.float32))
            self.M5 = ti.field(dtype=ti.f32, shape=(5, 5))
            self.M5.from_numpy(np.array([
                [1,  1,  1,  1,  1],
                [0,  1,  0, -1,  0],
                [0,  0,  1,  0, -1],
                [-4, 1,  1,  1,  1],
                [0,  1, -1,  1, -1]
            ], dtype=np.float32))
            self.invM5 = ti.field(dtype=ti.f32, shape=(5, 5))
            self.invM5.from_numpy(np.array([
                [1/5,   0,   0,-1/5,   0],
                [1/5, 1/2,   0, 1/20, 1/4],
                [1/5,   0, 1/2, 1/20,-1/4],
                [1/5,-1/2,   0, 1/20, 1/4],
                [1/5,   0,-1/2, 1/20,-1/4]
            ], dtype=np.float32))
            self.setup_mrt_rates(tau=1.0) # 初始化MRT松弛率，默认tau=1.0 (可根据需要调整)
        # initialize reactions class , knowing all species
        if self.CHEMISTRY:
            self.reactions.dS = ti.Vector.field(len(self.species) + 1, dtype=float, shape=self.rho.shape) # specie num + 1
            self.reactions.specieNum = len(self.species)

        # create export directory
        os.makedirs(self.exportPath, exist_ok=True)
        # handle exception
        ## mixture but no chemistry
        if not self.CHEMISTRY and self.TEMPERATURE:
            if self.TF.capacity_model == THERMO_MODEL.MIXTURE:
                self.TF.capacity_model = THERMO_MODEL.CONSTANT
                print("Warning: No chemistry module enabled, switching fluid capacity model to constant.")
            if self.TF.conductivity_model == CONDUCTIVITY_MODEL.MIXTURE:
                self.TF.conductivity_model = CONDUCTIVITY_MODEL.CONSTANT
                print("Warning: No chemistry module enabled, switching fluid conductivity model to constant.")
        fixSpecieNum = sum(1 for specie in self.species if specie.FIX) if self.CHEMISTRY else 0
        if (not self.CHEMISTRY or fixSpecieNum == 0) and self.TEMPERATURE:
            if self.TS.capacity_model == THERMO_MODEL.MIXTURE:
                self.TS.capacity_model = THERMO_MODEL.CONSTANT
                print("Warning: No chemistry module enabled, switching solid capacity model to constant.")
            if self.TS.conductivity_model == CONDUCTIVITY_MODEL.MIXTURE:
                self.TS.conductivity_model = CONDUCTIVITY_MODEL.CONSTANT
                print("Warning: No chemistry module enabled, switching solid conductivity model to constant.")
    @ti.kernel
    def static_init_kernel(self): # 初始化静态变量
        self.e9[0] = ti.Vector([0,0,0])
        self.e9[1] = ti.Vector([1,0,0]); self.e9[2] = ti.Vector([0,1,0]); self.e9[3] = ti.Vector([-1,0,0]); self.e9[4] = ti.Vector([0,-1,0])
        self.e9[5] = ti.Vector([1,1,0]); self.e9[6] = ti.Vector([-1,1,0]); self.e9[7] = ti.Vector([-1,-1,0]); self.e9[8] = ti.Vector([1,-1,0])
        self.w9[0] = 4.0/9.0
        self.w9[1] = 1.0/9.0; self.w9[2] = 1.0/9.0; self.w9[3] = 1.0/9.0; self.w9[4] = 1.0/9.0
        self.w9[5] = 1.0/36.0; self.w9[6] = 1.0/36.0; self.w9[7] = 1.0/36.0; self.w9[8] = 1.0/36.0
        self.e5[0] = ti.Vector([0,0,0])
        self.e5[1] = ti.Vector([1,0,0]); self.e5[2] = ti.Vector([0,1,0]); self.e5[3] = ti.Vector([-1,0,0]); self.e5[4] = ti.Vector([0,-1,0])
        self.w5[0] = 1.0/3.0
        self.w5[1] = 1.0/6.0; self.w5[2] = 1.0/6.0; self.w5[3] = 1.0/6.0; self.w5[4] = 1.0/6.0

    @ti.kernel
    def init_kernel(self): # 初始化所有分布函数
        for i in ti.grouped(self.solid):
            eps = 1-self.solid[i]
            if ti.static(self.PORO):
                if ti.static(self.CHEMISTRY): # 计算固体物质密度
                    self.rhos[i]=0.0
                    for specie in ti.static(list(self.species)):
                        if ti.static(specie.FIX):
                            self.rhos[i] += specie.S[i]
                self.rhos0[i] = self.rhos[i]/(1-eps) if eps < 1 else 0.0
            # 需要添加rhos0
            for s in ti.static(range(9)):
                self.f[i][s] = self.feq9_no_poro(s,self.rho[i],i[0],i[1],i[2])
                if ti.static(self.CHEMISTRY):
                    if s<5:
                        for specie in ti.static(list(self.species)):
                            if ti.static(not specie.FIX):
                                specie.g[i][s] = specie.geq5(s,specie.S[i],i[0],i[1],i[2])
                if ti.static(self.TEMPERATURE):
                    if s<5:
                        self.TF.g[i][s] = self.TF.geq5(s,self.TF.S[i],i[0],i[1],i[2])
                        self.TS.g[i][s] = self.TS.geq5(s,self.TS.S[i],i[0],i[1],i[2])
        self.init_boundary()
    @ti.func
    def init_boundary(self): # 固定流量边界需要特殊初始化
        # boundary 0 (left) - velocity init
        if ti.static(self.bc[0]==BC_FLOW.inlet_flow):
            v0 = self.flow_BC[0]/self.ny/self.nz
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[0][0,j,k] = ti.Vector([v0,0,0])
                self.v[0,j,k] = ti.Vector([v0,0,0])
                self.v[1,j,k] = ti.Vector([v0,0,0])
        if ti.static(self.bc[0]==BC_FLOW.inlet or self.bc[0]==BC_FLOW.wall):
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[0][0,j,k] = self.v_BC[0]
        # boundary 1 (right) - velocity init
        if ti.static(self.bc[1]==BC_FLOW.inlet_flow):
            v1 = self.flow_BC[1]/self.ny/self.nz
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[1][0,j,k] = ti.Vector([v1,0,0])
        if ti.static(self.bc[1]==BC_FLOW.inlet or self.bc[1]==BC_FLOW.wall):
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[1][0,j,k] = self.v_BC[1]
        # boundary 2 (bottom) - velocity init
        if ti.static(self.bc[2]==BC_FLOW.inlet_flow):
            v2 = self.flow_BC[2]/self.nx/self.nz
            for i,k in ti.ndrange(self.nx,self.nz):
                self.v_bc_profile[2][i,0,k] = ti.Vector([0,v2,0])
        if ti.static(self.bc[2]==BC_FLOW.inlet or self.bc[2]==BC_FLOW.wall):
            for i,k in ti.ndrange(self.nx,self.nz):
                self.v_bc_profile[2][i,0,k] = self.v_BC[2]
        # boundary 3 (top) - velocity init
        if ti.static(self.bc[3]==BC_FLOW.inlet_flow):
            v3 = self.flow_BC[3]/self.nx/self.nz
            for i,k in ti.ndrange(self.nx,self.nz):
                self.v_bc_profile[3][i,0,k] = ti.Vector([0,v3,0])
        if ti.static(self.bc[3]==BC_FLOW.inlet or self.bc[3]==BC_FLOW.wall):
            for i,k in ti.ndrange(self.nx,self.nz):
                self.v_bc_profile[3][i,0,k] = self.v_BC[3]
        # density profiles (all boundaries)
        for j,k in ti.ndrange(self.ny,self.nz):
            self.rho_bc_profile[0][0,j,k] = self.rho_BC[0]
            self.rho_bc_profile[1][0,j,k] = self.rho_BC[1]
        for i,k in ti.ndrange(self.nx,self.nz):
            self.rho_bc_profile[2][i,0,k] = self.rho_BC[2]
            self.rho_bc_profile[3][i,0,k] = self.rho_BC[3]

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
        # s0 (密度): 守恒矩, 松弛率为0
        self.s_mrt[0] = 0.0
        # s1 (能量e): 与体粘性相关
        self.s_mrt[1] = omega_bulk
        # s2 (能量平方ε): 与体粘性相关
        self.s_mrt[2] = omega_bulk
        # s3 (jx): 守恒矩
        self.s_mrt[3] = 0.0
        # s4 (qx): 高阶矩, 使用magic参数
        self.s_mrt[4] = s_magic
        # s5 (jy): 守恒矩
        self.s_mrt[5] = 0.0
        # s6 (qy): 高阶矩
        self.s_mrt[6] = s_magic
        # s7 (pxx): 剪切应力, 决定运动粘度 ν = cs^2*(τ-0.5)*dt
        self.s_mrt[7] = omega
        # s8 (pxy): 剪切应力
        self.s_mrt[8] = omega
        
        # D2Q5 MRT 松弛率 (标量输运)
        # s0 (φ): 守恒矩
        self.s_mrt_q5[0] = 0.0
        # s1 (jx), s2 (jy): 通量, 决定扩散系数 D = cs^2*(τ-0.5)*dt
        self.s_mrt_q5[1] = omega
        self.s_mrt_q5[2] = omega
        # s3 (e), s4 (ε): 高阶矩
        self.s_mrt_q5[3] = s_magic
        self.s_mrt_q5[4] = s_magic