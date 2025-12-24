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
    def description(self):
        des = "\n------------------------------------------\n"
        des += "Basic information of the LBM simulation:\n"
        des += " name : " + self.name+f"\n Size : {self.nx} x {self.ny} x {self.nz}\n dx : {self.dx}\n dt : {self.dt}\n"
        des += "Modules activated :\n"
        des += " -flow field\n"
        if self.PORO:
            des+=" -porous medium\n"
        if self.TEMPERATURE:
            des+=" -thermal transfer\n"
        if self.RADIATION:
            des+=" -radiation\n"
        if ti.static(self.CHEMISTRY):
            des+=" -chemical reaction\n"

        des+="Boundary condition model : "+ BC_MODEL(self.boundary_condition_model).name+"\n"
        des+="The boundary conditions of the flow field are set to :\n" 
        sideName = ti.static(["left","right","front","back","bottom","top"])

        for i in ti.static(range(6)):
            # des+=self.bc[i])
            if ti.static(self.bc[i]==BC_FLOW.periodic):
                des+="    "+sideName[i]+": PERIODIC\n"
            if ti.static(self.bc[i]==BC_FLOW.wall):
                des+="    "+sideName[i]+": WALL\n"
            if ti.static(self.bc[i]==BC_FLOW.inlet):
                des+="    "+sideName[i]+": INLET\n"
            if ti.static(self.bc[i]==BC_FLOW.outlet):
                des+="    "+sideName[i]+": OUTLET\n"
            if ti.static(self.bc[i]==BC_FLOW.symmetric):
                des+="    "+sideName[i]+": SYMMETRIC\n"
            if ti.static(self.bc[i]==BC_FLOW.inlet_flow):
                des+="    "+sideName[i]+": INLET FLOW\n"
        if ti.static(self.CHEMISTRY):
            des+="The species involved are: "
            for specie in self.species:
                des += specie.__str__()+" "
            des += "\n"+self.reactions.__str__()
        des += "------------------------------------------\n\n"
        return des
    def init_simulation(self):
        self.init_python()
        self.init_taichi()
    def init_python(self):
        if ti.static(self.CHEMISTRY):
            print(self.CHEMISTRY)
            self.reactions.dS = ti.Vector.field(len(self.species),dtype = float,shape=self.rho.shape)
            self.reactions.specieNum = len(self.species)
    @ti.kernel
    def init_taichi(self):# 用户设置完数值之后进行手动初始化
        self.static_init_kernel()
        self.init_kernel()
        # self.macro()
    
    @ti.func
    def static_init_kernel(self): # 初始化静态变量
        if ti.static(self.enable_projection): # No runtime overhead
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
            
    @ti.func
    def init_kernel(self): # 初始化所有分布函数
        for i in ti.grouped(self.rho):
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
        
