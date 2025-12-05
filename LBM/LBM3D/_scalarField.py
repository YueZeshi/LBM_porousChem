import taichi as ti
from util.flag import BC

@ti.data_oriented    
class ScalarField:
    '''
    标量场 D3Q7
    '''
    def __init__(self,name,nx,ny,nz,lb3d,FIX = False):
        self.name = name
        self.nx,self.ny,self.nz = nx,ny,nz
        self.LBM = lb3d
        self.FIX = FIX
        self.S = ti.field(float,shape=(nx,ny,nz))
        if not self.FIX:
            self.g = ti.Vector.field(7,float,shape=(nx,ny,nz))
            self.G = ti.Vector.field(7,float,shape=(nx,ny,nz))
            self.BC = [BC.PERIODIC,BC.PERIODIC,BC.PERIODIC,BC.PERIODIC,BC.PERIODIC,BC.PERIODIC]
            self.flux_BC =ti.field(float,shape = (6))
            self.s_BC = ti.field(float,shape = (6))
            for i in range(6):
                self.flux_BC[i]=0.0
                self.s_BC[i]=0.0
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
        for i in range(6):
            self.s_BC[i]=s[i]
    def set_s_BC_flux(self,index,f):
        self.flux_BC[index]=f
    def set_s_BCs_flux(self,f):
        for i in range(6):
            self.flux_BC[i]=f[i]
    @ti.func
    def geq7(self,k,S,x,y,z):
        return 0
    @ti.func
    def Boundary_condition_scalar_0(self,x,y,z):
        if ti.static(self.BC[0]==BC.FIXVALUE): # fix v
            self.S[x,0,z] = self.s_BC[0]
        if ti.static(self.BC[0]==BC.zeroGradient): # open
            self.S[x,0,z] = self.S[x,1,z]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,0,z][s] = self.geq7(s,self.S[x,0,z],x,0,z)+(self.G[x,0,z][s]-self.geq7(s,self.S[x,1,z],x,1,z))
    @ti.func
    def Boundary_condition_scalar_1(self,x,y,z):
        if ti.static(self.BC[1]==BC.WALL):# no slip
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        if ti.static(self.BC[1]==BC.FIXVALUE): # fix v
            self.S[self.nx-1,y,z] = self.s_BC[1]
        if ti.static(self.BC[1]==BC.ZEROGRADIENT): # open
            self.S[self.nx-1,y,z] = self.S[self.nx-2,y,z]  
        # todo
        for s in ti.static(range(7)):
            self.G[self.nx-1,y,z][s] = self.geq7(s,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.G[self.nx-2,y,z][s]-self.geq7(s,self.S[self.nx-2,y,z],self.nx-2,y,z))
    @ti.func
    def Boundary_condition_scalar_2(self,x,y,z):
        if ti.static(self.BC[2]==BC.WALL):# no slip
            self.S[x,0,z] = self.S[x,1,z]  
        if ti.static(self.BC[2]==BC.FIXVALUE): # fix v
            self.S[x,0,z] = self.s_BC[2]
        if ti.static(self.BC[2]==BC.ZEROGRADIENT): # open
            self.S[x,0,z] = self.S[x,1,z]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,0,z][s] = self.geq7(s,self.S[x,0,z],x,0,z)+(self.G[x,1,z][s]-self.geq7(s,self.S[x,1,z],x,1,z))
    @ti.func
    def Boundary_condition_scalar_3(self,x,y,z):
        if ti.static(self.BC[3]==BC.WALL):# no slip
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        if ti.static(self.BC[3]==BC.FIXVALUE): # fix v
            self.S[x,self.ny-1,z] = self.s_BC[3]
        if ti.static(self.BC[3]==BC.ZEROGRADIENT): # open
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,self.ny-1,z][s] = self.geq7(s,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.G[x,self.ny-2,z][s]-self.geq7(s,self.S[x,self.ny-2,z],x,self.ny-2,z))
    @ti.func
    def Boundary_condition_scalar_4(self,x,y,z):
        if ti.static(self.BC[4]==BC.WALL):# no slip
            self.S[x,y,0] = self.S[x,y,1]  
        if ti.static(self.BC[4]==BC.FIXVALUE): # fix v
            self.S[x,y,0] = self.s_BC[4]
        if ti.static(self.BC[4]==BC.ZEROGRADIENT): # open
            self.S[x,y,0] = self.S[x,y,1]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,y,0][s] = self.geq7(s,self.S[x,y,0],x,y,0)+(self.G[x,y,1][s]-self.geq7(s,self.S[x,y,1],x,y,1))
    @ti.func
    def Boundary_condition_scalar_5(self,x,y,z):
        if ti.static(self.BC[5]==BC.FIXVALUE): # fix v
            self.S[x,y,self.nz-1] = self.s_BC[5]
        if ti.static(self.BC[5]==BC.ZEROGRADIENT): # open
            self.S[x,y,self.nz-1] = self.S[x,y,self.nz-2]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,y,self.nz-1][s] = self.geq7(s,self.S[x,y,self.nz-1],x,y,self.nz-1)+(self.G[x,y,self.nz-2][s]-self.geq7(s,self.S[x,y,self.nz-2],x,y,self.nz-2))
