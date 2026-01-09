from ._core import LBM3D_BASE
from ..util.flag import *
import taichi as ti

@ti.data_oriented
class LBM3D_INITIALIZATION(LBM3D_BASE):
    # 初始化
    @ti.kernel
    def default_init(self): # 创建时初始化
        for i in ti.ndrange(6):
            self.rho_BC[i] = 0.0
            self.v_BC[i] = ti.Vector([0,0,0])
        for i in ti.grouped(self.solid):
            self.solid[i] = 0.0
            self.v[i] = ti.Vector([0,0,0])
            self.rho[i] = 1.0
    def init_simulation(self):
        self.init_python()
        self.init_kernel()
    def init_python(self):
        if self.CHEMISTRY:
            self.reactions.dS = ti.Vector.field(len(self.species),dtype = float,shape=self.rho.shape)
            self.reactions.specieNum = len(self.species)
        # handle exception
        ## mixture but no chemistry
        if not self.CHEMISTRY and self.TEMPERATURE:
            if self.TF.capacity_model == THERMO_MODEL.MIXTURE:
                self.TF.capacity_model = THERMO_MODEL.CONSTANT
                print("Warning: No chemistry module enabled, but mixture fluid capacity model selected. Switching to constant model.")
            if self.TF.conductivity_model == CONDUCTIVITY_MODEL.MIXTURE:
                self.TF.conductivity_model = CONDUCTIVITY_MODEL.CONSTANT
                print("Warning: No chemistry module enabled, but mixture fluid conductivity model selected. Switching to constant model.")
            if self.TS.capacity_model == THERMO_MODEL.MIXTURE:
                self.TS.capacity_model = THERMO_MODEL.CONSTANT
                print("Warning: No chemistry module enabled, but mixture solid capacity model selected. Switching to constant model.")
            if self.TS.conductivity_model == CONDUCTIVITY_MODEL.MIXTURE:
                self.TS.conductivity_model = CONDUCTIVITY_MODEL.CONSTANT
                print("Warning: No chemistry module enabled, but mixture solid conductivity model selected. Switching to constant model.")
    @ti.kernel
    def static_init_kernel(self): # 初始化静态变量
            self.e19[0] = ti.Vector([0,0,0])
            self.e19[1] = ti.Vector([1,0,0]); self.e19[2] = ti.Vector([-1,0,0]); self.e19[3] = ti.Vector([0,1,0]); self.e19[4] = ti.Vector([0,-1,0]);self.e19[5]=ti.Vector([0,0,1]);self.e19[6]=ti.Vector([0,0,-1]) 
            self.e19[7] = ti.Vector([1,1,0]); self.e19[8] = ti.Vector([-1,1,0]); self.e19[9] = ti.Vector([1,-1,0]); self.e19[10] = ti.Vector([-1,-1,0])
            self.e19[11] = ti.Vector([1,0,1]); self.e19[12] = ti.Vector([-1,0,1]); self.e19[13] = ti.Vector([1,0,-1]); self.e19[14] = ti.Vector([-1,0,-1])
            self.e19[15] = ti.Vector([0,1,1]); self.e19[16] = ti.Vector([0,-1,1]); self.e19[17] = ti.Vector([0,1,-1]); self.e19[18] = ti.Vector([0,-1,-1])
            
            self.w19[0] = 1.0/3.0
            self.w19[1] = 1.0/18.0; self.w19[2] = 1.0/18.0; self.w19[3] = 1.0/18.0; self.w19[4] = 1.0/18.0; self.w19[5] = 1.0/18.0; self.w19[6] = 1.0/18.0
            self.w19[7] = 1.0/36.0; self.w19[8] = 1.0/36.0; self.w19[9] = 1.0/36.0; self.w19[10] = 1.0/36.0
            self.w19[11] = 1.0/36.0; self.w19[12] = 1.0/36.0; self.w19[13] = 1.0/36.0; self.w19[14] = 1.0/36.0
            self.w19[15] = 1.0/36.0; self.w19[16] = 1.0/36.0; self.w19[17] = 1.0/36.0; self.w19[18] = 1.0/36.0

            self.e7[0] = ti.Vector([0,0,0])
            self.e7[1] = ti.Vector([1,0,0]); self.e7[2] = ti.Vector([-1,0,0]); self.e7[3] = ti.Vector([0,1,0]); self.e7[4] = ti.Vector([0,-1,0]);self.e7[5] = ti.Vector([0,0,1]); self.e7[6] = ti.Vector([0,0,-1])
            
            self.w7[0] = 1.0/4.0
            self.w7[1] = 1.0/8.0; self.w7[2] = 1.0/8.0; self.w7[3] = 1.0/8.0; self.w7[4] = 1.0/8.0; self.w7[5] = 1.0/8.0; self.w7[6] = 1.0/8.0
            
    @ti.kernel
    def init_kernel(self): # 初始化所有分布函数
        for i in ti.grouped(self.rho):
            eps = 1-self.solid[i]
            if ti.static(self.PORO):
                if ti.static(self.CHEMISTRY): # 计算固体物质密度
                    self.rhos[i]=0.0
                    for specie in ti.static(list(self.species)):
                        if ti.static(specie.FIX):
                            self.rhos[i] += specie.S[i]
                # if self.solid[i]!=0: # 计算孔隙率为1的密度作为参考
                #     self.rho1[i] = self.rhos[i]/self.solid[i]
                #     if self.rho1[i]==0:
                #         self.rho1[i]=1.0
            for s in ti.static(range(19)):
                self.f[i][s] = self.feq19(s,i[0],i[1],i[2])
                self.F[i][s] = self.f[i][s]
            # if ti.static(self.TEMPERATURE):
            #     self.IE.uS[i] = self.IE.S[i]*self.v[i]
            for s in ti.static(range(7)):
                if ti.static(self.CHEMISTRY):
                    for specie in ti.static(list(self.species)):
                        if ti.static(not specie.FIX):
                            # specie.init()
                            specie.g[i][s] = specie.geq7(s,specie.S[i],i[0],i[1],i[2])
                            specie.G[i][s] = specie.g[i][s]
                if ti.static(self.TEMPERATURE):
                # self.IE.init()
                    self.TF.g[i][s] = self.TF.geq7(s,self.TF.S[i],i[0],i[1],i[2])
                    self.TF.G[i][s] = self.TF.g[i][s]
                    self.TS.g[i][s] = self.TS.geq7(s,self.TS.S[i],i[0],i[1],i[2])
                    self.TS.G[i][s] = self.TS.g[i][s]
        self.init_boundary()
    @ti.func
    def init_boundary(self): # 固定流量边界需要特殊初始化
        if ti.static(self.bc[0]==BC_FLOW.inlet_flow):
            v0 = self.flow_BC[0]/self.ny/self.nz
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[0][0,j,k] = ti.Vector([v0,0,0])
                self.v[0,j,k] = ti.Vector([v0,0,0])
                self.v[1,j,k] = ti.Vector([v0,0,0])
        if ti.static(self.bc[0]==BC_FLOW.inlet):
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[0][0,j,k] = self.v_BC[0]