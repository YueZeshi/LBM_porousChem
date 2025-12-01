import taichi as ti
from ._chemical import Specie,Reaction
from ._thermal import TemperatureFluid,TemperatureSolid
from util.flag import *
@ti.data_oriented
class LBM3D_EVOLUTION:
    """
    evolution part of LBM3D solver
    """
    SIGMA = 5.67e-8
    @ti.kernel
    def step(self):
        self.collision_source_streaming()
        self.macro()  # F->value
        # self.source_term_macro() # value 
        self.Boundary_condition() # value changed for the boundary condition
    @ti.func
    def collision_source_streaming(self):
        for i in ti.grouped(self.rho):
            # collision f
            if (self.solid[i] < 1):
                for k in ti.static(range(19)):
                    eps = 1-self.solid[i]
                    f =self.f[i][k]-1/(3*self.viscosity(i)+0.5)*(self.f[i][k]-self.feq19(k,self.rho[i],self.v[i],eps))
                    self.f[i][k] =f
                    if ti.static(k<7):
                        if ti.static(self.CHEMICAL):
                            for specie in ti.static(list(self.species.values())):
                                if ti.static(not specie.FIX):
                                    g =specie.g[i][k]-1/(3*specie.coefDiff(i)+0.5)*(specie.g[i][k]-specie.geq7(k,specie.S[i],i[0],i[1],i[2]))
                                    specie.g[i][k] =g 
                        if ti.static(self.THERMAL):
                            g = self.TF.g[i][k]-1/(3*self.TF.coefDiff(i)+0.5)*(self.TF.g[i][k]-self.TF.geq7(k,self.TF.S[i],i[0],i[1],i[2]))
                            self.TF.g[i][k] =g
                            if eps > 0.0:
                                g= self.TS.g[i][k]-1/(3*self.TS.coefDiff(i)+0.5)*(self.TS.g[i][k]-self.TS.geq7(k,self.TS.S[i],i[0],i[1],i[2]))
                                self.TS.g[i][k] =g
                            # source term
            if ti.static(self.force_term_model==FORCE_TERM.GUO):
                if ti.static(self.PORO):
                    if (self.solid[i] < 1 and self.solid[i]>0):
                        for k in ti.ndrange(19):
                            self.f[i][k] +=self.forceTermGuo(k,self.rho[i],self.v[i],self.force(i),3*self.viscosity(i)+0.5) # 如果力太大，刚性太强->需要使用宏观指数衰减
            if ti.static(self.source_term_model==SOURCE_TERM.MICRO):
                if self.solid[i]>1e-6: # 有固体的区域
                    if ti.static(self.THERMAL):
                        q = 100.0*self.heat_trasnfer_surface[i]*(self.TF.S[i]-self.TS.S[i]) # from f to s
                        for k in ti.ndrange(7):
                            self.TS.g[i][k] += self.TS.geq7(k,q/self.TS.capacity_v(i),i[0],i[1],i[2])
                            self.TF.g[i][k] += self.TF.geq7(k,-q/self.TF.capacity_v(i),i[0],i[1],i[2])            
                if ti.static(self.RADIATION): # radiation
                    dT =self.dt*self.radiation(i)/self.TS.capacity_v(i) # 计算温度变化 # s*{rad}/(Jm-3K-1)=K; {rad}=Wm-3
                    for k in ti.ndrange(7):
                        self.TS.g[i][k] += self.TS.geq7(k,dT,i[0],i[1],i[2]) # 辐射吸热 # 直接添加到格子中央可能会带来问题
                if ti.static(self.CHEMICAL):
                    for r in ti.static(list(self.reactions.values())):
                        r.reaction(i) # update species and internal energy
            # streaming f->F
            if (self.solid[i]<1):# 流体更新
                for s in ti.static(range(19)):
                    ip = self.periodic_index(i+self.e19[s]) # 更新之后的位置索引，默认边界视为周期边界
                    if (self.solid[ip]!=1): # 如果更新之后的位置是液体    
                        self.F[ip][s] = self.f[i][s] # 直接更新F
                        if ti.static(s<7):
                            if ti.static(self.THERMAL):
                                self.TF.G[ip][s] = self.TF.g[i][s]
                            if ti.static(self.CHEMICAL):
                                for specie in ti.static(list(self.species.values())):
                                    if ti.static(not specie.FIX):
                                        specie.G[ip][s] = specie.g[i][s]
                    else:
                        self.F[i][self.LR[s]] = self.f[i][s] # 如果不是 反弹 流固表面处理
                        if ti.static(s<7):
                            if ti.static(self.THERMAL):
                                self.TF.G[i][self.LR[s]] = self.TF.g[i][s]
                            if ti.static(self.CHEMICAL):
                                for specie in ti.static(list(self.species.values())):
                                    if ti.static(not specie.FIX):
                                        specie.G[i][self.LR[s]] = specie.g[i][s]
            if (self.solid[i] > 0):# 固体更新
                for s in ti.static(range(7)):
                    ip = self.periodic_index(i+self.e19[s]) # 更新之后的位置索引，默认边界视为周期边界
                    if (self.solid[ip] >0): # 如果更新之后的位置有固体  
                        if ti.static(self.THERMAL):
                            self.TS.G[ip][s] = self.TS.g[i][s]
                    else: # 更新之后无固体 迁移步骤中认为是绝热壁 热交换在源项中
                        if ti.static(self.THERMAL):
                            self.TS.G[i][self.LR[s]] = self.TS.g[i][s]
    @ti.func
    def Boundary_condition(self):
        for j,k in ti.ndrange(self.ny,self.nz):            
            self.Boundary_condition_flow_0(0,j,k)
            if ti.static(self.CHEMICAL):
                for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_0(0,j,k)
            if ti.static(self.THERMAL):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_0(0,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_0(0,j,k)
            self.Boundary_condition_flow_1(self.nx-1,j,k)
            if ti.static(self.CHEMICAL):
                for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_1(self.nx-1,j,k)
            if ti.static(self.THERMAL):
                self.TF.Boundary_condition_scalar_1(self.nx-1,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_1(self.nx-1,j,k)
        for i,k in ti.ndrange(self.nx,self.nz):
            self.Boundary_condition_flow_2(i,0,k)
            if ti.static(self.CHEMICAL):
               for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_2(i,0,k)
            if ti.static(self.THERMAL):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_2(i,0,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_2(i,0,k)
            self.Boundary_condition_flow_3(i,self.ny-1,k)
            if ti.static(self.CHEMICAL):
                for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_3(i,self.ny-1,k)
            if ti.static(self.THERMAL):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_3(i,self.ny-1,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_3(i,self.ny-1,k)
        
        for i,j in ti.ndrange(self.nx,self.ny):
            self.Boundary_condition_flow_4(i,j,0)
            if ti.static(self.CHEMICAL):
               for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_4(i,j,0)
            if ti.static(self.THERMAL):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_4(i,j,0) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_4(i,j,0)
            self.Boundary_condition_flow_5(i,j,self.nz-1)
            if ti.static(self.CHEMICAL):
                for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_5(i,j,self.nz-1)
            if ti.static(self.THERMAL):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_5(i,j,self.nz-1) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_5(i,j,self.nz-1)
    @ti.func
    def Boundary_condition_flow_0(self,x,y,z):
        if ti.static(self.bc_v[0]==BC.fixedValue):
            self.v[0,y,z] = self.v_BC[0]
        elif ti.static(self.bc_v[0]==BC.zeroGradient):
            self.v[0,y,z] = self.v[1,y,z]
        if ti.static(self.bc_rho[0]==BC.fixedValue):
            self.rho[0,y,z] = self.rho_BC[0]
        elif ti.static(self.bc_rho[0]==BC.zeroGradient):
            self.rho[0,y,z] = self.rho[1,y,z]
        
        for s in ti.static(range(19)):
            self.f[0,y,z][s] = self.feq19(s,self.rho[0,y,z],self.v[0,y,z])+(self.f[1,y,z][s]-self.feq19(s,self.rho[1,y,z],self.v[1,y,z]))
    @ti.func
    def Boundary_condition_flow_1(self,x,y,z):
        
        if ti.static(self.bc_v[1]==BC.fixedValue):
            self.v[self.nx-1,y,z] = self.v_BC[1]
        elif ti.static(self.bc_v[1]==BC.zeroGradient):
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
        if ti.static(self.bc_rho[1]==BC.fixedValue):
            self.rho[self.nx-1,y,z] = self.rho_BC[1]
        elif ti.static(self.bc_rho[1]==BC.zeroGradient):
            self.rho[self.nx-1,y,z] = self.rho[self.nx-2,y,z]
        for s in ti.static(range(19)):
            self.f[self.nx-1,y,z][s] = self.feq19(s,self.rho[self.nx-1,y,z],self.v[self.nx-1,y,z])+(self.f[self.nx-2,y,z][s]-self.feq19(s,self.rho[self.nx-2,y,z],self.v[self.nx-2,y,z]))
    @ti.func
    def Boundary_condition_flow_2(self,x,y,z):
        if ti.static(self.bc_v[2]==BC.fixedValue):
            self.v[x,0,z] = self.v_BC[2]
        elif ti.static(self.bc_v[2]==BC.zeroGradient):
            self.v[x,0,z] = self.v[x,1,z]
        if ti.static(self.bc_rho[2]==BC.fixedValue):
            self.rho[x,0,z] = self.rho_BC[2]
        elif ti.static(self.bc_rho[2]==BC.zeroGradient):
            self.rho[x,0,z] = self.rho[x,1,z]
        for s in ti.static(range(19)):
            self.f[x,0,z][s] = self.feq19(s,self.rho[x,0,z],self.v[x,0,z])+(self.f[x,1,z][s]-self.feq19(s,self.rho[x,1,z],self.v[x,1,z]))
    @ti.func
    def Boundary_condition_flow_3(self,x,y,z):
        if ti.static(self.bc_v[3]==BC.fixedValue):
            self.v[x,self.ny-1,z] = self.v_BC[3]
        elif ti.static(self.bc_v[3]==BC.zeroGradient):
            self.v[x,self.ny-1,z] = self.v[x,self.ny-2,z]
        if ti.static(self.bc_rho[3]==BC.fixedValue):
            self.rho[x,self.ny-1,z] = self.rho_BC[3]
        elif ti.static(self.bc_rho[3]==BC.zeroGradient):
            self.rho[x,self.ny-1,z] = self.rho[x,self.ny-2,z]
        for s in ti.static(range(19)):
            self.f[x,self.ny-1,z][s] = self.feq19(s,self.rho[x,self.ny-1,z],self.v[x,self.ny-1,z])+(self.f[x,self.ny-2,z][s]-self.feq19(s,self.rho[x,self.ny-2,z],self.v[x,self.ny-2,z]))
    @ti.func
    def Boundary_condition_flow_4(self,x,y,z):
        if ti.static(self.bc_v[4]==BC.fixedValue):
            self.v[x,y,0] = self.v_BC[4]
        elif ti.static(self.bc_v[4]==BC.zeroGradient):
            self.v[x,y,0] = self.v[x,y,1]
        if ti.static(self.bc_rho[4]==BC.fixedValue):
            self.rho[x,y,0] = self.rho_BC[4]
        elif ti.static(self.bc_rho[4]==BC.zeroGradient):
            self.rho[x,y,0] = self.rho[x,y,1]
        for s in ti.static(range(19)):
            self.f[x,y,0][s] = self.feq19(s,self.rho[x,y,0],self.v[x,y,0])+(self.f[x,y,1][s]-self.feq19(s,self.rho[x,y,1],self.v[x,y,1]))
    @ti.func
    def Boundary_condition_flow_5(self,x,y,z):
        if ti.static(self.bc_v[5]==BC.fixedValue):
            self.v[x,y,self.nz-1] = self.v_BC[5]
        elif ti.static(self.bc_v[5]==BC.zeroGradient):
            self.v[x,y,self.nz-1] = self.v[x,y,self.nz-2]
        if ti.static(self.bc_rho[5]==BC.fixedValue):
            self.rho[x,y,self.nz-1] = self.rho_BC[5]
        elif ti.static(self.bc_rho[5]==BC.zeroGradient):
            self.rho[x,y,self.nz-1] = self.rho[x,y,self.nz-2]
        for s in ti.static(range(19)):
            self.f[x,y,self.nz-1][s] = self.feq19(s,self.rho[x,y,self.nz-1],self.v[x,y,self.nz-1])+(self.f[x,y,self.nz-2][s]-self.feq19(s,self.rho[x,y,self.nz-2],self.v[x,y,self.nz-2]))
    
    def updateBC(self,t):
        for func in self.UpdateBCfunc:
            func(self,t)

    @ti.func
    def macro(self):# F F->f 计算宏观量 F为正确值
        for i in ti.grouped(self.rho):
            if self.solid[i]<1: # 流体/多孔介质区域
                eps = 1-self.solid[i]
                self.v[i] = ti.Vector([0,0,0])
                self.rho[i] = 0.0
                for s in ti.static(range(19)):
                    self.f[i][s] = self.F[i][s] # 更新F
                    self.v[i] += self.e19[s]*self.F[i][s]
                    self.rho[i] += self.F[i][s] # 宏观量重建 计算密度
                self.v[i] /= self.rho[i]
                # if self.v[i].norm()>0.5:
                #     print("warning: high velocity",self.v[i],self.t)
                if ti.static(self.PORO):
                    if eps!=0 and eps!=1: # 涉及多孔介质的速度修正
                        Dc,Fc = 0.0,0.0
                        if ti.static(self.poro_model==PORO_MODEL.SPHERICAL):
                            Dc = 1.0/self.perm(eps)
                            Fc = 1.75/ti.sqrt(150*ti.pow(eps,3))
                        elif ti.static(self.poro_model==PORO_MODEL.DARCY):
                            Dc = self.coefDarcy[i]
                        elif ti.static(self.poro_model==PORO_MODEL.DARCYFORCHHEIMER):
                            Dc = self.coefDarcy[i]
                            Fc = self.coefForchheimer[i]
                        c0 = 0.5*(1+eps*self.viscosity(i)/2*Dc)
                        c1 = 0.5*eps*Fc
                        v = ti.math.length(self.v[i])
                        self.v[i]/=(c0+ti.sqrt(c0**2+c1*v))
                if ti.static(self.CHEMICAL): 
                    if ti.static(self.PORO):
                        self.rhos[i]=0.0
                    Yall = 0.0
                    for specie in ti.static(list(self.species.values())):
                        if ti.static(not specie.FIX): # 更新流体组分
                            specie.S[i] = 0.0
                            for s in ti.static(range(7)):
                                specie.g[i][s] = specie.G[i][s] # 更新G
                                specie.S[i] = specie.G[i][s]
                            Yall += specie.S[i]
                        else:
                            if ti.static(self.PORO): # 计算当前固体物质总密度
                                self.rhos[i]+=specie.S[i]
                    for specie in ti.static(list(self.species.values())):
                        if ti.static(not specie.FIX): # 更新流体组分
                            specie.S[i] /= Yall
                    
                if ti.static(self.PORO):
                    if self.rho1[i] != 0 and self.rhos[i] != 0:
                        self.solid[i] = self.rhos[i]/self.rho1[i] # 更新孔隙结构
                
                if ti.static(self.THERMAL):
                    self.TF.S[i] = 0.0
                    self.TS.S[i] = 0.0
                    for s in ti.static(range(7)):
                        self.TF.g[i][s] = self.TF.G[i][s] # 更新G
                        self.TS.g[i][s] = self.TS.G[i][s] # 更新G
                        self.TF.S[i] = self.TF.G[i][s]# 计算温度
                        self.TS.S[i] = self.TS.G[i][s]   
    @ti.func
    def source_term_macro(self):
        if ti.static(self.force_term_model==FORCE_TERM.MACRO):
            for i in ti.grouped(self.rho):
                if ti.static(self.PORO): 
                    # v  : 直接隐式求解速度 当Darcy阻力过大时，减速太快。显式更新破坏稳定性。最好使用隐式更新。
                    if (self.solid[i] < 1 and self.solid[i]>0): # 只有Darcy力 多孔介质区域
                        if self.poro_model == PORO_MODEL.DARCY:
                            self.v[i] *= ti.math.exp(-self.viscosity(i)*self.coefDarcy[i]*self.dt)
        # chemical reaction 更新化学物质
        if ti.static(self.source_term_model==SOURCE_TERM.MACRO):
            if ti.static(self.RADIATION): # radiation
                for i in ti.grouped(self.rho):          
                        self.TS.S[i] += self.dt*self.radiation(i)/self.TS.capacity_v(i) # 辐射吸热 # 直接添加到格子中央可能会带来问题
            if ti.static(self.CHEMICAL):
                for r in ti.static(list(self.reactions.values())):
                    r.reaction_macro() # update species and internal energy
    
    
    # 使用函数
    @ti.func
    def periodic_index(self,i): # 按照周期边界条件获取索引，避免边界条件判断多余处理
        iout = i
        if i[0]<0:     iout[0] = self.nx-1
        if i[0]>self.nx-1:  iout[0] = 0
        if i[1]<0:     iout[1] = self.ny-1
        if i[1]>self.ny-1:  iout[1] = 0
        if i[2]<0:     iout[2] = self.nz-1
        if i[2]>self.nz-1:  iout[2] = 0
    
        return iout
    @ti.func
    def feq19(self, k,rho, u,eps = 1.0): #计算平衡分布函数 
        eu = self.e19[k].dot(u)
        uv = u.dot(u)
        feqout = self.w19[k]*rho*(1.0+3.0*eu+4.5*eu*eu/(eps+1e-12)-1.5*uv/(eps+1e-12))
        return feqout
    @ti.func
    def feq7(self, k,rho_local, u,eps = 1.0): #计算平衡分布函数 考虑多孔介质
        eu = self.e7[k].dot(u)
        uv = u.dot(u)
        feqout = self.w7[k]*rho_local*(1.0+3.0*eu)
        return feqout
    
    @ti.func
    def viscosity(self,i):
        return 0.1
    @ti.func
    def tau(self,i):
        return 3*self.viscosity(i)/self.rho[i]+0.5
    @ti.func
    def perm(self,eps): # 多孔介质 渗透率
        p = eps**3/(1.000001-eps)**2*1**2
        return p
    @ti.func 
    def force(self,i):
        F = ti.Vector([0.0,0.0,0.0])
        F+=self.volume_force()
        F+=self.darcy_force(i)
        return F
    @ti.func
    def volume_force(self):
        return ti.Vector([0,0,0])
    @ti.func
    def darcy_force(self,i): # 达西定律
        F = ti.Vector([0.0,0.0,0.0])
        if ti.static(self.PORO):
            if(self.solid[i]!=0 and self.solid[i]!=1):
                if ti.static(self.poro_model==PORO_MODEL.SPHERICAL):
                    eps = 1.0-self.solid[i]
                    perm = self.perm(eps)
                    beta = 1.75/ti.sqrt(150*eps*perm)
                    F += (-self.viscosity(i)/perm*eps-beta*ti.math.length(self.v[i]))*self.v[i]
                elif ti.static(self.poro_model==PORO_MODEL.DARCY):
                    F +=  -self.viscosity(i)*self.coefDarcy[i]*self.rho[i]*self.v[i]
                elif ti.static(self.poro_model==PORO_MODEL.DARCYFORCHHEIMER):
                    F += (-self.viscosity(i)*self.coefDarcy[i]-self.coefForchheimer[i]*ti.math.length(self.v[i]))*self.rho[i]*self.v[i]
        return F
    @ti.func
    def forceTermGuo(self,k,rho,u,F,tau): # 将力转化为分布函数源项 Guo Zhao
        return (1.0-1.0/2.0/tau)*rho*self.w19[k]*(3.0*(self.e19[k]-u).dot(F)\
              +9.0*self.e19[k].dot(u)*self.e19[k].dot(F))
    @ti.func
    def scalarCorrectionTerm(self,k,duS,tau):
        return (1.0-1.0/2.0/tau)*3.0*self.w7[k]*self.e7[k].dot(duS)
    @ti.func
    def radiation(self,i):# Wm-2K-4*m-1K4*{m2}=Wm-3 SI # 单位体积辐射 SI
        q = 0.0
        if ti.static(self.radiation_model==RADIATION_MODEL.SURFACE_UNIFORM):
            q += LBM3D_EVOLUTION.SIGMA*self.radiation_surface[i]/self.dx*(ti.pow(self.Tambient,4)-ti.pow(self.TS.S[i],4))
        elif ti.static(self.radiation_model==RADIATION_MODEL.REAL_RADIATION):
            q += self.real_radiation[i]-LBM3D_EVOLUTION.SIGMA*self.radiation_surface[i]/self.dx*ti.pow(self.TS.S[i],4)

        return q
    @ti.func
    def check(self):
        print(self.viscosity([0,0,0]))