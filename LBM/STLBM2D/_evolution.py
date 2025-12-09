import taichi as ti
from ._core import STLBM2D_BASE
from ._chemical import Specie,Reaction
from ._thermal import TemperatureFluid,TemperatureSolid
from ..util.flag import *
@ti.data_oriented
class STLBM2D_EVOLUTION(STLBM2D_BASE):
    """
    evolution part of LBM
    The realisation of Boundary condition is written in _boundary.py because of its complexity
    """
    def step(self):
        self.step_kernel()
        self.updateBC(self.t)
        self.t+=self.dt
        
    @ti.kernel
    def step_kernel(self):
        self.predictor()
        self.corrector()
        self.Boundary_condition()
    @ti.func
    def predictor(self): 
        # predictor and corrector
        ## predictor 逆向迁移 （从四周吸收）
        ## corrector 在格子内部进行松弛（根据
        for i in ti.grouped(self.rho):
            if (self.solid[i] < 1): # 流体和多孔介质区域 流体 物种 流体温度
                rho1 = 0.0
                rho1v1 = ti.Vector([0.0,0.0,0.0])
                if ti.static(self.TEMPERATURE):
                    T1 = 0.0
                # pred1
                for k in ti.static(range(9)):
                    # pred1
                    j = self.periodic_index(i - self.e9[k]) # 逆向迁移位置 
                    if self.solid[j] < 1: # 逆向迁移之后的位置不是固体
                        rho1 += self.feq9(k,self.rho[j],self.v[j])
                        rho1v1 += self.feq9(k,self.rho[j],self.v[j])*self.e9[k]
                    else:
                        rho1 += self.feq9(self.LR[k],self.rho[i],self.v[i]) # 迁移之后是固体eq进行反弹
                        rho1v1 += self.feq9(self.LR[k],self.rho[i],self.v[i])*self.e9[k]
                v1 = rho1v1/rho1
                self.rho1[i] = rho1
                self.v1[i] = v1
    @ti.func
    def corrector(self):
        for i in ti.grouped(self.rho):
            if (self.solid[i] < 1): # 流体和多孔介质区域 流体 物种 流体温度
                # corrector
                rho2 = self.rho1[i]
                rhov_neq_tau = ti.Vector([0.0,0.0,0.0])
                for k in ti.static(range(9)):
                    j = self.periodic_index(i - self.e9[k]) # 逆向迁移位置
                    if self.solid[j] < 1: # 逆向迁移的位置不是固体
                        j2 = self.periodic_index(i - 2*self.e9[k]) 
                        if self.solid[j2] < 1: 
                            rhov_neq_tau -= (self.feq9(k,self.rho1[j],self.v1[j])-self.feq9(k,self.rho[j2],self.v[j2]))*self.e9[k]
                        else: # 格点在流固边界
                            rhov_neq_tau -= (self.feq9(k,self.rho1[j],self.v1[j])-self.feq9(self.LR[k],self.rho[j],self.v[j]))*self.e9[k] # 第一项j格点 第二项是来自固体格点，实际上是j格点的信息反弹之后的
                    else: # 从固体格点来的信息 neq进行反弹 应该是i格点反向的neq
                        j3 = self.periodic_index(i - self.e9[self.LR[k]]) # 反向位置
                        rhov_neq_tau -= (self.feq9(self.LR[k],self.rho1[i],self.v1[i])-self.feq9(self.LR[k],self.rho[j3],self.v[j3]))*self.e9[k]
                rho2v2 = self.rho1[i]*self.v1[i]+(self.tau(i)-1)*rhov_neq_tau
                v2 = rho2v2/rho2
                self.rho[i] = rho2
                self.v[i] = v2
    def updateBC(self,t):
        for func in self.UpdateBCfunc:
            func(self,t)
    
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
    def feq9(self, s, rho,u): #计算平衡分布函数
        eu = self.e9[s].dot(u)
        uv = u.dot(u)
        feqout = 1.0
        if self.EOS==FLUID_STATE_EQUATION.INCOMPRESSIBLE:
            feqout = self.w9[s]*(rho+3.0*eu+4.5*eu*eu-1.5*uv)        
        if self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS:
            feqout = self.w9[s]*rho*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
        return feqout
    @ti.func
    def fneq9(self,s,i):
        return 0
    @ti.func
    def feq5(self, s,S,u): #计算平衡分布函数 考虑多孔介质
        eu = self.e5[s].dot(u)
        feqout = self.w5[s]*S*(1.0+3.0*eu)
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
    
