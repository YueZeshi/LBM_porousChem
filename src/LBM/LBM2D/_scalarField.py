import taichi as ti
from ..util.flag import BC

@ti.data_oriented    
class ScalarField:
    '''
    标量场 D2Q5
    '''
    def __init__(self,name,lb2d,FIX = False):
        self.name = name
        self.nx,self.ny,self.nz = lb2d.nx,lb2d.ny,lb2d.nz
        self.LBM = lb2d
        self.FIX = FIX
        self.S = ti.field(float,shape = (self.nx,self.ny,self.nz))
        self.dS = ti.field(float,shape = (self.nx,self.ny,self.nz))
        self.v_ref = ti.field(float, shape=())
        self.v_scale = ti.field(float, shape=())
        self.v_ref[None] = 0.0
        self.v_scale[None] = 1.0
        if ti.static(not self.FIX):
            self.g = ti.Vector.field(5,float,shape=(self.nx,self.ny,self.nz))
            self.BC = [BC.periodic]*4
            self.flux_BC =ti.field(float,shape = (4))
            self.s_BC = ti.field(float,shape = (4))
            for i in range(4):
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
        for i in range(4):
            if ti.static(not self.FIX):
                self.flux_BC[i]=0.0
                self.s_BC[i]=0.0
        for i in ti.grouped(self.S):
            self.S[i]=0.0
            if ti.static(not self.FIX):
                for k in ti.static(range(5)):
                    self.g[i][k]=0.0
        
    @ti.func
    def tau(self,i):
        return 3*self.coefDiff(i)+.5
    @ti.func
    def coefDiff(self,i):
        return 0.1
    # boundary condition
    def set_BC(self,index_boundary,BC):
        if not self.FIX:
            self.BC[index_boundary]=BC
    def set_BCs(self,BCs):
        for i in range(4):
            self.set_BC(i,BCs[i])
    def set_s_BC_value(self,index,s):
        if not self.FIX:
            self.s_BC[index]=self.get_normalized_value(s)
    def set_s_BCs_value(self,s):
        for i in range(4):
            self.set_s_BC_value(i,s[i])
    def set_s_BC_flux(self,index,f):
        self.flux_BC[index]=f
    def set_s_BCs_flux(self,f):
        for i in range(4):
            self.set_s_BC_flux(i,f[i])
    @ti.func
    def geq5(self,k,S,x,y,z):
        return 0
    @ti.func
    def Boundary_condition_scalar_0(self,x,y,z):
        if ti.static(self.BC[0]==BC.fixedValue):
            self.S[0,y,z] = self.s_BC[0]
        if ti.static(self.BC[0]==BC.zeroGradient):
            self.S[0,y,z] = self.S[1,y,z]  
        # todo 1
        if self.even_step[None]==0: # 偶数步
            self.g[1,y,z][3] = self.geq5(1,self.S[0,y,z],0,y,z)+(self.g[2,y,z][3]-self.geq5(1,self.S[1,y,z],1,y,z))
        else: # 奇数步
            self.g[0,y,z][1] = self.geq5(1,self.S[0,y,z],0,y,z)+(self.g[1,y,z][1]-self.geq5(1,self.S[1,y,z],1,y,z))
    @ti.func
    def Boundary_condition_scalar_1(self,x,y,z):
        if ti.static(self.BC[1]==BC.fixedValue):
            self.S[self.nx-1,y,z] = self.s_BC[1]  
        if ti.static(self.BC[1]==BC.zeroGradient):
            self.S[self.nx-1,y,z] = self.S[self.nx-2,y,z]
        # todo 3
        if self.even_step[None]==0: # 偶数步
            self.g[self.nx-2,y,z][1] = self.geq5(3,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.g[self.nx-3,y,z][1]-self.geq5(3,self.S[self.nx-2,y,z],self.nx-2,y,z))
        else: # 奇数步
            self.g[self.nx-1,y,z][3] = self.geq5(3,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.g[self.nx-2,y,z][3]-self.geq5(3,self.S[self.nx-2,y,z],self.nx-2,y,z))
    @ti.func
    def Boundary_condition_scalar_2(self,x,y,z):
        if ti.static(self.BC[2]==BC.fixedValue):
            self.S[x,0,z] = self.s_BC[2]
        if ti.static(self.BC[2]==BC.zeroGradient):
            self.S[x,0,z] = self.S[x,1,z]  
        # todo 2
        if self.even_step[None]==0: # 偶数步
            self.g[x,1,z][4] = self.geq5(2,self.S[x,0,z],x,0,z)+(self.g[x,2,z][4]-self.geq5(2,self.S[x,1,z],x,1,z))
        else: # 奇数步
            self.g[x,0,z][2] = self.geq5(2,self.S[x,0,z],x,0,z)+(self.g[x,1,z][2]-self.geq5(2,self.S[x,1,z],x,1,z))
    @ti.func
    def Boundary_condition_scalar_3(self,x,y,z):
        if ti.static(self.BC[3]==BC.fixedValue): 
            self.S[x,self.ny-1,z] = self.s_BC[3]
        if ti.static(self.BC[3]==BC.zeroGradient):
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        # todo 4
        if self.even_step[None]==0: # 偶数步
            self.g[x,self.ny-2,z][2] = self.geq5(4,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.g[x,self.ny-3,z][2]-self.geq5(4,self.S[x,self.ny-2,z],x,self.ny-2,z))
        else: # 奇数步
            self.g[x,self.ny-1,z][4] = self.geq5(4,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.g[x,self.ny-2,z][4]-self.geq5(4,self.S[x,self.ny-2,z],x,self.ny-2,z))
    @ti.func
    def Boundary_condition_scalar_AA_0(self,x,y,z):
        if ti.static(self.BC[0]==BC.fixedValue):
            self.S[0,y,z] = self.s_BC[0]
        if ti.static(self.BC[0]==BC.zeroGradient):
            self.S[0,y,z] = self.S[1,y,z]  
        # todo 1
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[1,y,z][3] = self.geq5(1,self.S[0,y,z],0,y,z)+(self.g[2,y,z][3]-self.geq5(1,self.S[1,y,z],1,y,z))
        else: # 奇数步
            self.g[0,y,z][1] = self.geq5(1,self.S[0,y,z],0,y,z)+(self.g[1,y,z][1]-self.geq5(1,self.S[1,y,z],1,y,z))
    @ti.func
    def Boundary_condition_scalar_AA_1(self,x,y,z):
        if ti.static(self.BC[1]==BC.fixedValue):
            self.S[self.nx-1,y,z] = self.s_BC[1]  
        if ti.static(self.BC[1]==BC.zeroGradient):
            self.S[self.nx-1,y,z] = self.S[self.nx-2,y,z]
        # todo 3
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[self.nx-2,y,z][1] = self.geq5(3,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.g[self.nx-3,y,z][1]-self.geq5(3,self.S[self.nx-2,y,z],self.nx-2,y,z))
        else: # 奇数步
            self.g[self.nx-1,y,z][3] = self.geq5(3,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.g[self.nx-2,y,z][3]-self.geq5(3,self.S[self.nx-2,y,z],self.nx-2,y,z))
    @ti.func
    def Boundary_condition_scalar_AA_2(self,x,y,z):
        if ti.static(self.BC[2]==BC.fixedValue):
            self.S[x,0,z] = self.s_BC[2]
        if ti.static(self.BC[2]==BC.zeroGradient):
            self.S[x,0,z] = self.S[x,1,z]  
        # todo 2
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[x,1,z][4] = self.geq5(2,self.S[x,0,z],x,0,z)+(self.g[x,2,z][4]-self.geq5(2,self.S[x,1,z],x,1,z))
        else: # 奇数步
            self.g[x,0,z][2] = self.geq5(2,self.S[x,0,z],x,0,z)+(self.g[x,1,z][2]-self.geq5(2,self.S[x,1,z],x,1,z))
    @ti.func
    def Boundary_condition_scalar_AA_3(self,x,y,z):
        if ti.static(self.BC[3]==BC.fixedValue): 
            self.S[x,self.ny-1,z] = self.s_BC[3]
        if ti.static(self.BC[3]==BC.zeroGradient):
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        # todo 4
        if self.LBM.even_step[None]==0: # 偶数步
            self.g[x,self.ny-2,z][2] = self.geq5(4,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.g[x,self.ny-3,z][2]-self.geq5(4,self.S[x,self.ny-2,z],x,self.ny-2,z))
        else: # 奇数步
            self.g[x,self.ny-1,z][4] = self.geq5(4,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.g[x,self.ny-2,z][4]-self.geq5(4,self.S[x,self.ny-2,z],x,self.ny-2,z))