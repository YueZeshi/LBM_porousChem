import taichi as ti
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
        if ti.static(self.CHEMISTRY):
            # initialize reactions class , knowing all species
            self.reactions.dS = ti.Vector.field(len(self.species)+1,dtype = float,shape=self.rho.shape) # specie num + 1
            self.reactions.specieNum = len(self.species)
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
                # if self.solid[i]!=0: # 计算孔隙率为1的密度作为参考
                #     self.rho1[i] = self.rhos[i]/self.solid[i]
                #     if self.rho1[i]==0:
                #         self.rho1[i]=1.0
            for s in ti.static(range(9)):
                self.f[i][s] = self.feq9(s,i[0],i[1],i[2])
                self.F[i][s] = self.f[i][s]
            for s in ti.static(range(5)):
                if ti.static(self.CHEMISTRY):
                    for specie in ti.static(list(self.species)):
                        if ti.static(not specie.FIX):
                            specie.g[i][s] = specie.geq5(s,specie.S[i],i[0],i[1],i[2])
                            specie.G[i][s] = specie.g[i][s]
                if ti.static(self.TEMPERATURE):
                    self.TF.g[i][s] = self.TF.geq5(s,self.TF.S[i],i[0],i[1],i[2])
                    self.TF.G[i][s] = self.TF.g[i][s]
                    self.TS.g[i][s] = self.TS.geq5(s,self.TS.S[i],i[0],i[1],i[2])
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
            print("inlet flow v0:",v0)
        if ti.static(self.bc[0]==BC_FLOW.inlet):
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[0][0,j,k] = self.v_BC[0]

