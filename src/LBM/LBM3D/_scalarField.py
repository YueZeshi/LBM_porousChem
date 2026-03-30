import taichi as ti
from ..util.flag import BC

@ti.data_oriented    
class ScalarField:
    '''
    标量场 D3Q7
    '''
    def __init__(self,name,lb3d,FIX = False):
        self.name = name
        self.nx,self.ny,self.nz = lb3d.nx,lb3d.ny,lb3d.nz
        self.LBM = lb3d
        self.FIX = FIX
        self.S = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.dS = ti.field(float,shape = (self.nx,self.ny,self.nz))
        self.v_ref = ti.field(float,shape=())
        self.v_scale = ti.field(float,shape=())
        self.v_ref[None] = 0.0
        self.v_scale[None] = 1.0
        if not self.FIX:
            self.g = ti.Vector.field(7,float,shape=(self.nx,self.ny,self.nz))
            self.G = ti.Vector.field(7,float,shape=(self.nx,self.ny,self.nz))
            self.BC = [BC.periodic]*6
            self.flux_BC =ti.field(float,shape = (6))
            self.s_BC = ti.field(float,shape = (6))
            for i in range(6):
                self.flux_BC[i]=0.0
                self.s_BC[i]=0.0
    def get_physical_value(self,v): # change unit and reference
        v_phys = v*self.v_scale[None] + self.v_ref[None]
        return v_phys

    def get_normalized_value(self,v_phys):
        v = (v_phys - self.v_ref[None])/self.v_scale[None]
        return v    
    
    @ti.func
    def normalized_value(self,v_phys):
        v = (v_phys - self.v_ref[None])/self.v_scale[None]
        return v
    @ti.func
    def physical_value(self,v):
        v_phys = v*self.v_scale[None] + self.v_ref[None]
        return v_phys

    @ti.kernel
    def default_init(self):
        for i in range(6):
            self.flux_BC[i]=0.0
            self.s_BC[i]=0.0
        for i in ti.grouped(self.S):
            self.S[i]=0.0
            if not self.FIX:
                for k in ti.static(range(7)):
                    self.g[i][k]=0.0
                    self.G[i][k]=0.0
    @ti.func
    def tau(self,i):
        return 4*self.coefDiff(i)+0.5
    @ti.func
    def coefDiff(self,i):
        return 0.1
    # boundary condition
    def set_BC(self,index_boundary,BC):
        if not self.FIX:
            self.BC[index_boundary]=BC
    def set_BCs(self,BCs):
        for i in range(6):
            self.set_BC(i,BCs[i])
    def set_s_BC_value(self,index,s):
        if not self.FIX:
            self.s_BC[index]=self.get_normalized_value(s)
    def set_s_BCs_value(self,s):
        for i in range(6):
            self.set_s_BC_value(i,s[i])
    def set_s_BC_flux(self,index,f):
        self.flux_BC[index]=f
    def set_s_BCs_flux(self,f):
        for i in range(6):
            self.set_s_BC_flux(i,f[i])
    @ti.func
    def geq7(self,k,S,x,y,z):
        u = self.LBM.v[x,y,z]
        eu = self.LBM.e7[k].dot(u)
        uv = u.dot(u)
        geqout = 0.0
        geqout += self.LBM.w7[k]*S*(1+4.0*eu+8.0*eu*eu-2*uv)
        return geqout
    @ti.func
    def Boundary_condition_scalar_0(self,x,y,z):
        if ti.static(self.BC[0]==BC.fixedValue): # fix v
            self.S[0,y,z] = self.s_BC[0]
        if ti.static(self.BC[0]==BC.zeroGradient): # open
            self.S[0,y,z] = self.S[1,y,z]  
        # todo
        for s in ti.static(range(7)):
            self.G[0,y,z][s] = self.geq7(s,self.S[0,y,z],0,y,z)+(self.G[1,y,z][s]-self.geq7(s,self.S[1,y,z],1,y,z))
            self.g[0,y,z][s] = self.G[0,y,z][s]
    @ti.func
    def Boundary_condition_scalar_1(self,x,y,z):
        if ti.static(self.BC[1]==BC.fixedValue): # fix v
            self.S[self.nx-1,y,z] = self.s_BC[1]
        if ti.static(self.BC[1]==BC.zeroGradient): # open
            self.S[self.nx-1,y,z] = self.S[self.nx-2,y,z]  
        # todo
        for s in ti.static(range(7)):
            self.G[self.nx-1,y,z][s] = self.geq7(s,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.G[self.nx-2,y,z][s]-self.geq7(s,self.S[self.nx-2,y,z],self.nx-2,y,z))
            self.g[self.nx-1,y,z] = self.G[self.nx-1,y,z]
    @ti.func
    def Boundary_condition_scalar_2(self,x,y,z):
        if ti.static(self.BC[2]==BC.fixedValue): # fix v
            self.S[x,0,z] = self.s_BC[2]
        if ti.static(self.BC[2]==BC.zeroGradient): # open
            self.S[x,0,z] = self.S[x,1,z]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,0,z][s] = self.geq7(s,self.S[x,0,z],x,0,z)+(self.G[x,1,z][s]-self.geq7(s,self.S[x,1,z],x,1,z))
            self.g[x,0,z] = self.G[x,0,z]
    @ti.func
    def Boundary_condition_scalar_3(self,x,y,z):
        if ti.static(self.BC[3]==BC.fixedValue): # fix v
            self.S[x,self.ny-1,z] = self.s_BC[3]
        if ti.static(self.BC[3]==BC.zeroGradient): # open
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,self.ny-1,z][s] = self.geq7(s,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.G[x,self.ny-2,z][s]-self.geq7(s,self.S[x,self.ny-2,z],x,self.ny-2,z))
            self.g[x,self.ny-1,z] = self.G[x,self.ny-1,z]
    @ti.func
    def Boundary_condition_scalar_4(self,x,y,z):
        if ti.static(self.BC[4]==BC.fixedValue): # fix v
            self.S[x,y,0] = self.s_BC[4]
        if ti.static(self.BC[4]==BC.zeroGradient): # open
            self.S[x,y,0] = self.S[x,y,1]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,y,0][s] = self.geq7(s,self.S[x,y,0],x,y,0)+(self.G[x,y,1][s]-self.geq7(s,self.S[x,y,1],x,y,1))
            self.g[x,y,0] = self.G[x,y,0]
    @ti.func
    def Boundary_condition_scalar_5(self,x,y,z):
        if ti.static(self.BC[5]==BC.fixedValue): # fix v
            self.S[x,y,self.nz-1] = self.s_BC[5]
        if ti.static(self.BC[5]==BC.zeroGradient): # open
            self.S[x,y,self.nz-1] = self.S[x,y,self.nz-2]  
        # todo
        for s in ti.static(range(7)):
            self.G[x,y,self.nz-1][s] = self.geq7(s,self.S[x,y,self.nz-1],x,y,self.nz-1)+(self.G[x,y,self.nz-2][s]-self.geq7(s,self.S[x,y,self.nz-2],x,y,self.nz-2))
            self.g[x,y,self.nz-1][s] = self.G[x,y,self.nz-1][s]
    @ti.func
    def Boundary_condition_scalar_AA_0(self,x,y,z):
        if ti.static(self.BC[0]==BC.fixedValue):
            self.S[0,y,z] = self.s_BC[0]
        if ti.static(self.BC[0]==BC.zeroGradient):
            self.S[0,y,z] = self.S[1,y,z]  
        # AA NEE for boundary 0 (x=0)
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[1,y,z][2] = self.geq7(1,self.S[0,y,z],0,y,z)+(self.g[2,y,z][2]-self.geq7(1,self.S[1,y,z],1,y,z))
        else: # 奇数步
            self.g[0,y,z][1] = self.geq7(1,self.S[0,y,z],0,y,z)+(self.g[1,y,z][1]-self.geq7(1,self.S[1,y,z],1,y,z))
    @ti.func
    def Boundary_condition_scalar_AA_1(self,x,y,z):
        if ti.static(self.BC[1]==BC.fixedValue):
            self.S[self.nx-1,y,z] = self.s_BC[1]
        if ti.static(self.BC[1]==BC.zeroGradient):
            self.S[self.nx-1,y,z] = self.S[self.nx-2,y,z]
        # AA NEE for boundary 1 (x=nx-1)
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[self.nx-2,y,z][1] = self.geq7(2,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.g[self.nx-3,y,z][1]-self.geq7(2,self.S[self.nx-2,y,z],self.nx-2,y,z))
        else: # 奇数步
            self.g[self.nx-1,y,z][2] = self.geq7(2,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.g[self.nx-2,y,z][2]-self.geq7(2,self.S[self.nx-2,y,z],self.nx-2,y,z))
    @ti.func
    def Boundary_condition_scalar_AA_2(self,x,y,z):
        if ti.static(self.BC[2]==BC.fixedValue):
            self.S[x,0,z] = self.s_BC[2]
        if ti.static(self.BC[2]==BC.zeroGradient):
            self.S[x,0,z] = self.S[x,1,z]  
        # AA NEE for boundary 2 (y=0)
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[x,1,z][4] = self.geq7(3,self.S[x,0,z],x,0,z)+(self.g[x,2,z][4]-self.geq7(3,self.S[x,1,z],x,1,z))
        else: # 奇数步
            self.g[x,0,z][3] = self.geq7(3,self.S[x,0,z],x,0,z)+(self.g[x,1,z][3]-self.geq7(3,self.S[x,1,z],x,1,z))
    @ti.func
    def Boundary_condition_scalar_AA_3(self,x,y,z):
        if ti.static(self.BC[3]==BC.fixedValue): 
            self.S[x,self.ny-1,z] = self.s_BC[3]
        if ti.static(self.BC[3]==BC.zeroGradient):
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        # AA NEE for boundary 3 (y=ny-1)
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[x,self.ny-2,z][3] = self.geq7(4,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.g[x,self.ny-3,z][3]-self.geq7(4,self.S[x,self.ny-2,z],x,self.ny-2,z))
        else: # 奇数步
            self.g[x,self.ny-1,z][4] = self.geq7(4,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.g[x,self.ny-2,z][4]-self.geq7(4,self.S[x,self.ny-2,z],x,self.ny-2,z))
    @ti.func
    def Boundary_condition_scalar_AA_4(self,x,y,z):
        if ti.static(self.BC[4]==BC.fixedValue):
            self.S[x,y,0] = self.s_BC[4]
        if ti.static(self.BC[4]==BC.zeroGradient):
            self.S[x,y,0] = self.S[x,y,1]  
        # AA NEE for boundary 4 (z=0)
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[x,y,1][6] = self.geq7(5,self.S[x,y,0],x,y,0)+(self.g[x,y,2][6]-self.geq7(5,self.S[x,y,1],x,y,1))
        else: # 奇数步
            self.g[x,y,0][5] = self.geq7(5,self.S[x,y,0],x,y,0)+(self.g[x,y,1][5]-self.geq7(5,self.S[x,y,1],x,y,1))
    @ti.func
    def Boundary_condition_scalar_AA_5(self,x,y,z):
        if ti.static(self.BC[5]==BC.fixedValue):
            self.S[x,y,self.nz-1] = self.s_BC[5]
        if ti.static(self.BC[5]==BC.zeroGradient):
            self.S[x,y,self.nz-1] = self.S[x,y,self.nz-2]  
        # AA NEE for boundary 5 (z=nz-1)
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[x,y,self.nz-2][5] = self.geq7(6,self.S[x,y,self.nz-1],x,y,self.nz-1)+(self.g[x,y,self.nz-3][5]-self.geq7(6,self.S[x,y,self.nz-2],x,y,self.nz-2))
        else: # 奇数步
            self.g[x,y,self.nz-1][6] = self.geq7(6,self.S[x,y,self.nz-1],x,y,self.nz-1)+(self.g[x,y,self.nz-2][6]-self.geq7(6,self.S[x,y,self.nz-2],x,y,self.nz-2))
