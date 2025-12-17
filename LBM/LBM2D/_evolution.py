import taichi as ti
from ._core import LBM2D_BASE
from ._chemical import Specie,Reaction
from ._thermal import TemperatureFluid,TemperatureSolid
from ..util.flag import *
@ti.data_oriented
class LBM2D_EVOLUTION(LBM2D_BASE):
    """
    evolution part of LBM
    The realisation of Boundary condition is written in _boundary.py because of its complexity
    """
    def step(self):
        self.updateBC(self.t)
        self.step_kernel()
        self.tLattice += 1
    @ti.kernel
    def step_kernel(self):
        self.collision_source_streaming() # f->F
        if ti.static(self.boundary_condition_model==BC_MODEL.NEBB):
            self.Boundary_condition_NEBB() # on F
        self.macro()  # F->value f
        if ti.static(self.boundary_condition_model==BC_MODEL.NEE):
            self.Boundary_condition_NEE() # value changed for the boundary condition
        if ti.static(self.boundary_condition_model==BC_MODEL.ES):
            self.Boundary_condition_ES()
    @ti.func
    def collision_source_streaming(self): 
        # collistion + source term + streaming : merge kernels
        for i in ti.grouped(self.rho):
            if (self.solid[i] < 1): # 流体和多孔介质区域 流体 物种 流体温度
                for k in ti.static(range(9)):
                    # collision f
                    f = self.f[i][k]-1/self.tau(i)*(self.f[i][k]-self.feq9(k,i[0],i[1],i[2]))
                    self.f[i][k] = f 
                    if ti.static(k<5):
                        if ti.static(self.TEMPERATURE): # 流体温度更新 有固体更新固体温度
                            # 有气体
                            g = self.TF.g[i][k]-1/self.TF.tau(i)*(self.TF.g[i][k]-self.TF.geq5(k,self.TF.S[i],i[0],i[1],i[2]))
                            self.TF.g[i][k] = g
                            if self.solid[i] > 0.0: # 有固体
                                g = self.TS.g[i][k]-1/self.TS.tau(i)*(self.TS.g[i][k]-self.TS.geq5(k,self.TS.S[i],i[0],i[1],i[2]))
                                self.TS.g[i][k] = g
                        if ti.static(self.CHEMISTRY):      
                            for specie in ti.static(list(self.species)):
                                if ti.static(not specie.FIX): # 气体物质更新
                                    g = specie.g[i][k]-1/specie.tau(i)*(specie.g[i][k]-specie.geq5(k,specie.S[i],i[0],i[1],i[2]))
                                    specie.g[i][k] = g 

                    # source term f # 将预先计算好的源项施加到各个分布函数分量当中 (化学反应 体积热源（辐射）)
                    ## force term for fluid flow: volume force like gravity, darcy drag force
                    if ti.static(self.force_term_model==FORCE_TERM.GUO): # GUO 力模型
                        self.f[i][k] +=self.forceTermGuo(k,i)
                    ## microscopic source term of scalar field
                    if ti.static(self.TEMPERATURE):
                        if ti.static(k<5):
                            self.TF.g[i][k] += self.TF.geq5(k,self.TF.dS[i],i[0],i[1],i[2])
                            if self.solid[i] > 0.0:
                                self.TS.g[i][k] += self.TS.geq5(k,self.TS.dS[i],i[0],i[1],i[2])
                    if ti.static(self.CHEMISTRY):      
                        if ti.static(k<5):
                            for specie in ti.static(list(self.species)):
                                if ti.static(not specie.FIX): # 气体物质更新
                                    specie.g[i][k] += self.TF.geq5(k,specie.dS[i],i[0],i[1],i[2])
                                else:
                                    if ti.static(k==0):
                                        specie.S[i] += specie.dS[i]
                    # streaming f->F
                    ip = self.periodic_index(i+self.e9[k]) # 更新之后的位置索引，默认边界视为周期边界
                    if (self.solid[ip]<1): # 如果更新之后的位置是液体 更新流体 气相温度 浓度 
                        self.F[ip][k] = self.f[i][k] # 直接更新F
                        if ti.static(k<5):
                            if ti.static(self.TEMPERATURE):
                                self.TF.G[ip][k] = self.TF.g[i][k]
                            if ti.static(self.CHEMISTRY):
                                for specie in ti.static(list(self.species)):
                                    if ti.static(not specie.FIX):
                                        specie.G[ip][k] = specie.g[i][k]
                    else: 
                        self.F[i][self.LR[k]] = self.f[i][k] # 如果不是 反弹 流固表面处理
                        if ti.static(k<5):
                            if ti.static(self.TEMPERATURE):
                                self.TF.G[i][self.LR[k]] = self.TF.g[i][k]
                            if ti.static(self.CHEMISTRY):
                                for specie in ti.static(list(self.species)):
                                    if ti.static(not specie.FIX):
                                        specie.G[i][self.LR[k]] = specie.g[i][k]
                    if (self.solid[i]) > 0.0: # 多孔介质区域的固体温度迁移
                        if(self.solid[ip] > 0): # 迁移之后有固体
                            if ti.static(k<5):
                                if ti.static(self.TEMPERATURE):
                                    self.TS.G[ip][k] = self.TS.g[i][k]
                        else: # 迁移之后没有固体当作绝热反弹边界
                            if ti.static(k<5):
                                if ti.static(self.TEMPERATURE):
                                    self.TS.G[i][self.LR[k]] = self.TS.g[i][k]

                        
            else: # 纯固体区域 只更新固体温度
                for k in ti.static(range(9)):
                    # collision
                    if ti.static(k<5):
                        if ti.static(self.TEMPERATURE):
                            g = self.TS.g[i][k]-1/self.TS.tau(i)*(self.TS.g[i][k]-self.TS.geq5(k,self.TS.S[i],i[0],i[1],i[2]))
                            self.TS.g[i][k] =g
                    # source term
                    # streaming
                    ip = self.periodic_index(i+self.e9[k]) # 更新之后的位置索引，默认边界视为周期边界
                    if (self.solid[ip]>0): # 如果更新之后的位置有固体  
                        if ti.static(k<5):
                            if ti.static(self.TEMPERATURE):
                                self.TS.G[ip][k] = self.TS.g[i][k]
                    else: # 更新之后无固体 迁移步骤中认为是绝热壁 热交换在源项中
                        if ti.static(k<5):
                            if ti.static(self.TEMPERATURE):
                                self.TS.G[i][self.LR[k]] = self.TS.g[i][k]

   
    @ti.func
    def macro(self):# F F->f 计算宏观量 F为正确值
        for i in ti.grouped(self.rho):
            if self.solid[i]<1: # 流体/多孔介质区域
                eps = 1-self.solid[i]
                self.v[i] = ti.Vector([0,0,0])
                self.rho[i] = 0.0
                for s in ti.static(range(9)):
                    self.f[i][s] = self.F[i][s] # 更新F
                    self.v[i] += self.e9[s]*self.F[i][s]
                    self.rho[i] += self.F[i][s] # 宏观量重建 计算密度
                if self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS:
                    self.v[i] /= self.rho[i]
                if ti.static(self.PORO):
                    if eps!=0 and eps!=1: # 涉及多孔介质的速度修正
                        Dc,Fc = 0.0,0.0
                        if ti.static(self.poro_model==PORO_MODEL.SPHERICAL):
                            Dc = 1.0/self.perm(eps)
                            Fc = 1.75/ti.sqrt(150*ti.pow(eps,3))
                            c0 = 0.5*(1+eps*self.kinetic_viscosity(i)/2*Dc)
                            c1 = 0.5*eps*Fc
                            v = ti.math.length(self.v[i])
                            self.v[i]/=(c0+ti.sqrt(c0**2+c1*v))
                        elif ti.static(self.poro_model==PORO_MODEL.DARCY):
                            Dc = self.coefDarcy[i]
                            c = 1+eps*self.kinetic_viscosity(i)*Dc/2
                            self.v[i]/=c
                        elif ti.static(self.poro_model==PORO_MODEL.DARCYFORCHHEIMER):
                            Dc = self.coefDarcy[i]
                            Fc = self.coefForchheimer[i]
                            c0 = 0.5*(1+eps*self.kinetic_viscosity(i)/2*Dc)
                            c1 = 0.5*eps*Fc
                            v = ti.math.length(self.v[i])
                            self.v[i]/=(c0+ti.sqrt(c0**2+c1*v))
                        # elif ti.static(self.poro_model==PORO_MODEL.DARCY_HIGH):
                        #     Dc = self.coefDarcy[i]
                        #     c =  1.5-ti.exp(-eps*self.kinetic_viscosity(i)*Dc)/2
                        #     # c =  0.5+ti.exp(eps*self.kinetic_viscosity(i)*Dc)/2
                        #     # c = 1+eps*self.kinetic_viscosity(i)*Dc/2
                        #     self.v[i]/=c
                if ti.static(self.TEMPERATURE):
                    self.TS.dS[i] = 0.0
                    self.TF.dS[i] = 0.0
                if ti.static(self.CHEMISTRY): 
                    if ti.static(self.PORO):
                        self.rhos[i]=0.0
                    Yall = 0.0
                    for specie in ti.static(list(self.species)):
                        if ti.static(not specie.FIX): # 更新流体组分
                            specie.S[i] = 0.0
                            for s in ti.static(range(5)):
                                specie.g[i][s] = specie.G[i][s] # 更新G
                                specie.S[i] += specie.G[i][s]
                            if specie.S[i]<0:
                                specie.S[i]=0
                            Yall += specie.S[i]
                        else:
                            if ti.static(self.PORO): # 计算当前固体物质总密度
                                self.rhos[i]+=specie.S[i]
                    for specie in ti.static(list(self.species)):
                        if ti.static(not specie.FIX): # 更新流体组分
                            specie.S[i] /= Yall # 归一化处理
                    # 化学反应
                    self.reactions.update_dS(i)
                    
                if ti.static(self.PORO):
                    if self.rho1[i] != 0 and self.rhos[i] != 0:
                        self.solid[i] = self.rhos[i]/self.rho1[i] # 更新孔隙结构
                # 宏观恢复温度场 并计算源项
                if ti.static(self.TEMPERATURE):
                    Tf = 0.0
                    Ts = 0.0
                    for s in ti.static(range(5)):
                        self.TF.g[i][s] = self.TF.G[i][s] # 更新G
                        self.TS.g[i][s] = self.TS.G[i][s] # 更新G
                        Tf += self.TF.G[i][s] # 计算温度
                        Ts += self.TS.G[i][s]
                    self.TF.S[i] = Tf
                    self.TS.S[i] = Ts
                    if self.solid[i] > 0: # 有固体
                        dH = self.TS.exchangeCoef[i]*self.TS.exchangeSurface[i]*(self.TF.physical_value(Tf)-self.TS.physical_value(Ts))*self.dt
                        self.TS.dS[i] += dH/self.TS.capacity_v(i)/self.TS.v_scale
                        self.TF.dS[i] += -dH/self.TF.capacity_v(i)/self.TF.v_scale
                    if ti.static(self.RADIATION):
                        self.TS.dS[i] += self.TS.radiation(i)*self.dt/self.TS.capacity_v(i)/self.TS.v_scale

    # 使用函数
    @ti.func
    def periodic_index(self,i): # 按照周期边界条件获取索引，避免边界条件判断多余处理
        iout = i
        if i[0]<0:     iout[0] = self.nx-1
        if i[0]>self.nx-1:  iout[0] = 0
        if i[1]<0:     iout[1] = self.ny-1
        if i[1]>self.ny-1:  iout[1] = 0
        return iout
    @ti.func
    def feq9(self, s,i,j,k): #计算平衡分布函数
        rho = self.rho[i,j,k]
        u = self.v[i,j,k]
        eps = 1-self.solid[i,j,k]
        eu = self.e9[s].dot(u)
        uv = u.dot(u)
        feqout = 1.0
        if self.EOS==FLUID_STATE_EQUATION.INCOMPRESSIBLE:
            feqout = self.w9[s]*(rho+3.0*eu+4.5*eu*eu/(eps+1e-12)-1.5*uv/(eps+1e-12))        
        if self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS:
            feqout = self.w9[s]*rho*(1.0+3.0*eu+4.5*eu*eu/(eps+1e-12)-1.5*uv/(eps+1e-12))
        return feqout
    @ti.func
    def feq5(self, k,s,i): #计算平衡分布函数 考虑多孔介质
        u = self.v[i]
        eu = self.e5[k].dot(u)
        feqout = self.w5[k]*s*(1.0+3.0*eu)
        return feqout
    
    @ti.func
    def viscosity(self,i):
        return 0.1
    @ti.func
    def kinetic_viscosity(self,i):
        nu = self.viscosity(i)
        if self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS:
            nu = nu/self.rho[i]
        return nu
    @ti.func
    def tau(self,i):
        return 3*self.kinetic_viscosity(i)+0.5
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
                    F += (-self.kinetic_viscosity(i)/perm*eps-beta*ti.math.length(self.v[i]))*self.v[i]
                elif ti.static(self.poro_model==PORO_MODEL.DARCY):
                    eps = 1.0-self.solid[i]
                    F +=  -eps*self.kinetic_viscosity(i)*self.coefDarcy[i]*self.v[i]
                elif ti.static(self.poro_model==PORO_MODEL.DARCYFORCHHEIMER):
                    eps = 1.0-self.solid[i]
                    F += (-eps*self.kinetic_viscosity(i)*self.coefDarcy[i]-eps*self.coefForchheimer[i]*ti.math.length(self.v[i]))*self.v[i]
                elif ti.static(self.poro_model==PORO_MODEL.DARCY_HIGH):
                    eps = 1.0-self.solid[i]
                    F +=  -eps*self.kinetic_viscosity(i)*self.coefDarcy[i]*self.v[i]
                    # F += (ti.exp(-eps*self.kinetic_viscosity(i)*self.coefDarcy[i])-1)*self.v[i]
        return F
    @ti.func
    def forceTermGuo(self,s,i): # 将力转化为分布函数源项 Guo Zhao 实际上是动量变化量
        rho = self.rho[i]
        u = self.v[i]
        F = self.force(i)
        tau = self.tau(i)
        eps = 1.0-self.solid[i]
        term = 0.0
        if ti.static(self.EOS==FLUID_STATE_EQUATION.INCOMPRESSIBLE):
            term = (1.0-1.0/2.0/tau)*self.w9[s]*(3.0*(self.e9[s]-u/eps).dot(F)\
              +9.0*self.e9[s].dot(u)*self.e9[s].dot(F)/eps)
        elif ti.static(self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS):
            term = (1.0-1.0/2.0/tau)*rho*self.w9[s]*(3.0*(self.e9[s]-u/(eps+1e-12)).dot(F)\
              +9.0*self.e9[s].dot(u)*self.e9[s].dot(F)/(eps+1e-12))
        return term
    @ti.func
    def scalarCorrectionTerm(self,k,duS,tau):
        return (1.0-1.0/2.0/tau)*3.0*self.w5[k]*self.e5[k].dot(duS)
    
