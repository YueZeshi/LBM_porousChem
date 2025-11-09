import taichi as ti
from ..util.flag import BC

@ti.data_oriented    
class ScalarField:
    '''
    标量场 D2Q5
    '''
    def __init__(self,name,nx,ny,nz,lb2d,FIX = False):
        self.name = name
        self.nx,self.ny,self.nz = nx,ny,nz
        self.LBM = lb2d
        self.FIX = FIX
        self.S = ti.field(ti.f32,shape=(nx,ny,nz))
        if not self.FIX:
            self.g = ti.Vector.field(5,ti.f32,shape=(nx,ny,nz))
            self.G = ti.Vector.field(5,ti.f32,shape=(nx,ny,nz))
            self.BC = [BC.PERIODIC]*4
            self.flux_BC =ti.field(ti.f32,shape = (4))
            self.s_BC = ti.field(ti.f32,shape = (4))
            for i in range(4):
                self.flux_BC[i]=0.0
                self.s_BC[i]=0.0
    @ti.kernel
    def default_init(self):
        for i in range(4):
            self.flux_BC[i]=0.0
            self.s_BC[i]=0.0
        for i in ti.grouped(self.S):
            self.S[i]=0.0
            if not self.FIX:
                for k in ti.static(range(5)):
                    self.g[i][k]=0.0
                    self.G[i][k]=0.0        
    @ti.func
    def coefDiff(self,i):
        return 0.1
    # boundary condition
    def set_BC(self,index_boundary,BC):
        self.BC[index_boundary]=BC
    def set_BCs(self,BCs):
        self.BC = BCs
    def set_s_BC_value(self,index,s):
        self.s_BC[index]=s
    def set_s_BCs_value(self,s):
        for i in range(4):
            self.s_BC[i]=s[i]
    def set_s_BC_flux(self,index,f):
        self.flux_BC[index]=f
    def set_s_BCs_flux(self,f):
        for i in range(4):
            self.flux_BC[i]=f[i]
    @ti.func
    def geq5(self,k,S,x,y,z):
        return 0
    @ti.func
    def Boundary_condition_scalar_0(self,x,y,z):
        if ti.static(self.BC[0]==BC.fixedValue):
            self.S[0,y,z] = self.s_BC[0]
        if ti.static(self.BC[0]==BC.zeroGadient):
            self.S[0,y,z] = self.S[1,y,z]  
        # todo
        for s in ti.static(range(5)):
            self.G[0,y,z][s] = self.geq5(s,self.S[0,y,z],0,y,z)+(self.G[0,y,z][s]-self.geq5(s,self.S[1,y,z],1,y,z))
    @ti.func
    def Boundary_condition_scalar_1(self,x,y,z):
        if ti.static(self.BC[1]==BC.fixedValue):
            self.S[self.nx-1,y,z] = self.s_BC[1]  
        if ti.static(self.BC[1]==BC.zeroGadient):
            self.S[self.nx-1,y,z] = self.S[self.nx-2,y,z]
        # todo
        for s in ti.static(range(5)):
            self.G[self.nx-1,y,z][s] = self.geq5(s,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.G[self.nx-2,y,z][s]-self.geq5(s,self.S[self.nx-2,y,z],self.nx-2,y,z))
    @ti.func
    def Boundary_condition_scalar_2(self,x,y,z):
        if ti.static(self.BC[0]==BC.fixedValue):
            self.S[x,0,z] = self.s_BC[0]
        if ti.static(self.BC[0]==BC.zeroGadient):
            self.S[x,0,z] = self.S[x,1,z]  
        # todo
        for s in ti.static(range(5)):
            self.G[x,0,z][s] = self.geq5(s,self.S[x,0,z],x,0,z)+(self.G[x,0,z][s]-self.geq5(s,self.S[x,1,z],x,1,z))
    @ti.func
    def Boundary_condition_scalar_3(self,x,y,z):
        if ti.static(self.BC[3]==BC.fixedValue): 
            self.S[x,self.ny-1,z] = self.s_BC[3]
        if ti.static(self.BC[3]==BC.zeroGadient):
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        # todo
        for s in ti.static(range(5)):
            self.G[x,self.ny-1,z][s] = self.geq5(s,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.G[x,self.ny-2,z][s]-self.geq5(s,self.S[x,self.ny-2,z],x,self.ny-2,z))