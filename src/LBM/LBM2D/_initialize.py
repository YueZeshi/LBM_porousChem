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
        # boundary 0
        if ti.static(self.bc[0]==BC_FLOW.inlet_flow):
            v0 = self.flow_BC[0]/self.ny/self.nz
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[0][0,j,k] = ti.Vector([v0,0,0])
                self.v[0,j,k] = ti.Vector([v0,0,0])
                self.v[1,j,k] = ti.Vector([v0,0,0])
        if ti.static(self.bc[0]==BC_FLOW.inlet):
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[0][0,j,k] = self.v_BC[0]
        for j,k in ti.ndrange(self.ny,self.nz):
            self.rho_bc_profile[0][0,j,k] = self.rho_BC[0]
            self.rho_bc_profile[1][0,j,k] = self.rho_BC[1]
        for i,k in ti.ndrange(self.nx,self.nz):
            self.rho_bc_profile[2][i,0,k] = self.rho_BC[2]
            self.rho_bc_profile[3][i,0,k] = self.rho_BC[3]

