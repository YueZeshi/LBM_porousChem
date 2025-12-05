import taichi as ti
from ..util.flag import *
from ._core import LBM2D_BASE
@ti.data_oriented
class LBM2D_INITIALIZATION(LBM2D_BASE):
    nx:int
    ny:int
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
    def print_information(self):
        print("\n \nBasic information of the LBM simulation:")
        print(self.name,f": Size: {self.nx} x {self.ny} x {self.nz}, dx: {self.dx}, dt: {self.dt}")
        print("It contains :")
        if self.PORO:
            print(" -porous medium")
        if self.TEMPERATURE:
            print(" -thermal transfer")
        if self.RADIATION:
            print(" -radiation")
        if ti.static(self.CHEMISTRY):
            print(" -chemical reaction")
            print("The species concerned: ",end="")
            for specie in ti.static(list(self.species.keys())):
                print(specie,end=" ")
            print("")
        print("Boundary condition model:",end=" ")
        print(BC_MODEL(self.bondary_condition_model).name)
        print("The boundary conditions of the flow field are set to :")
        sideName = ti.static(["left","right","bottom","top"])
        for i in ti.static(range(4)):
            # print(self.bc[i])
            if ti.static(self.bc[i]==BC_FLOW.periodic):
                print(sideName[i]+": PERIODIC")
            if ti.static(self.bc[i]==BC_FLOW.wall):
                print(sideName[i]+": WALL")
            if ti.static(self.bc[i]==BC_FLOW.inlet):
                print(sideName[i]+": INLET")
            if ti.static(self.bc[i]==BC_FLOW.outlet):
                print(sideName[i]+": OUTLET")
            if ti.static(self.bc[i]==BC_FLOW.symmetric):
                print(sideName[i]+": SYMMETRIC")
    def init_simulation(self):
        self.init_python()
        self.init_taichi()
    def init_python(self):
        if ti.static(self.CHEMISTRY):
            self.reactions.dS = ti.Vector.field(len(self.species),dtype = float,shape=self.rho.shape)
            self.reactions.specieNum = len(self.species)
    @ti.kernel
    def init_taichi(self):# 用户设置完数值之后进行手动初始化
        self.static_init_kernel()
        self.init_kernel()
        # self.macro()
    @ti.func
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
       
    @ti.func
    def init_kernel(self): # 初始化所有分布函数
        for i in ti.grouped(self.solid):
            eps = 1-self.solid[i]
            if ti.static(self.PORO):
                if ti.static(self.CHEMISTRY): # 计算固体物质密度
                    self.rhos[i]=0.0
                    for specie in ti.static(list(self.species)):
                        if ti.static(specie.FIX):
                            self.rhos[i] += specie.S[i]
                if self.solid[i]!=0: # 计算孔隙率为1的密度作为参考
                    self.rho1[i] = self.rhos[i]/self.solid[i]
                    if self.rho1[i]==0:
                        self.rho1[i]=1.0
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
        
