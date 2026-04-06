import taichi as ti
from ._core import LBM3D_BASE
from ._chemical import Specie,Reaction
from ._thermal import TemperatureFluid,TemperatureSolid
from ..util.flag import *
@ti.data_oriented
class LBM3D_EVOLUTION(LBM3D_BASE):
    """
    evolution part of LBM3D solver
    """
    def step(self):
        self.updateBC(self.t)
        # 目前默认使用 AA 单数组步进，仅流体部分；
        # 如需回退到原双数组算法，可改为调用 step_kernel。
        self.step_AA_kernel()
        self.tLattice += 1
        ti.sync()
    @ti.kernel
    def step_AA_kernel(self):# 使用AA实现单数组步进（仅流体部分，周期/NEE/ES 边界）
        even = self.even_step[None]
        # ========== 1. AA 碰撞 + 迁移（单数组 self.f） ==========
        for idx in ti.grouped(self.solid):
            # 目前仅对流体 / 多孔介质区域更新，纯固体区域流场更新默认全程反弹边界（一阶精度）
            if self.solid[idx] < 1.0:
                # ----- 1.1 读取阶段：偶数步从邻居读，奇数步从本格读 -----
                f_local = ti.Vector([0.0] * 19)
                # read
                for q in ti.static(range(19)):
                    e = self.e19[q]
                    opp_q = self.LR[q]
                    if even == 0:  # 偶数步：从邻居读取
                        ip = self.periodic_index(idx - e)
                        f_local[q] = self.f[ip][q]
                    else:  # 奇数步：从本格读取
                        f_local[q] = self.f[idx][opp_q]
                # 宏观量
                rho = 0.0
                v = ti.Vector([0.0, 0.0, 0.0])
                for q in ti.static(range(19)):
                    fi = f_local[q]
                    rho += fi
                    v += self.e19[q] * fi
                self.rho[idx] = rho
                if self.EOS == FLUID_STATE_EQUATION.IDEAL_GAS:
                    self.v[idx] = v / (rho + 1e-12)
                else:
                    self.v[idx] = v

                if ti.static(self.PORO):
                    if self.solid[idx] > 0.0: # 多孔介质区域，计算渗透性修正
                        Dc,Fc = 0.0,0.0
                        eps = 1-self.solid[idx] # 孔隙率
                        if ti.static(self.poro_model==PORO_MODEL.SPHERICAL):
                            Dc = 1.0/self.perm(eps)
                            Fc = 1.75/ti.sqrt(150*ti.pow(eps,3))
                            c0 = 0.5*(1+eps*self.kinetic_viscosity(idx)/2*Dc)
                            c1 = 0.5*eps*Fc
                            v_mag = ti.math.length(v)
                            v/=(c0+ti.sqrt(c0**2+c1*v_mag))
                        elif ti.static(self.poro_model==PORO_MODEL.DARCY):
                            Dc = self.coefDarcy[idx]
                            c = 1+eps*self.kinetic_viscosity(idx)*Dc/2
                            v/=c
                        elif ti.static(self.poro_model==PORO_MODEL.DARCYFORCHHEIMER):
                            Dc = self.coefDarcy[idx]
                            Fc = self.coefForchheimer[idx]
                            c0 = 0.5*(1+eps*self.kinetic_viscosity(idx)/2*Dc)
                            c1 = 0.5*eps*Fc
                            v_mag = ti.math.length(v)
                            v/=(c0+ti.sqrt(c0**2+c1*v_mag))

                # ----- 1.2 碰撞（BGK）+ 体力源项（GUO） -----
                tau = self.tau(idx)
                f_collided = ti.Vector([0.0] * 19)
                F = self.force(idx) # 计算体积力
                drho = self.drho[idx] # 化学反应引起的密度变化率
                if rho+drho < self.tol:
                    print("Warning: Density is zero or negative at idx ", idx,rho,drho)
                    drho = 0.0
                self.drho[idx] = 0.0 # 重置密度变化率，下一步重新计算
                # 碰撞
                for q in ti.static(range(19)):
                    feq = self.feq19(q, rho,idx[0], idx[1], idx[2])
                    # 标准 BGK: f_new = f_old - (f_old - f_eq) / tau
                    fq = f_local[q] - (f_local[q] - feq) / tau # 碰撞项
                    fq += self.forceTermGuo(q, idx) # 源项
                    fq += self.feq19(q, drho,idx[0], idx[1], idx[2])# 化学反应引起的密度变化修正
                    f_collided[q] = fq  # 赋值碰撞结果

                # ----- 1.3 写入阶段（AA 迁移，写回 self.f） -----
                for q in ti.static(range(19)):
                    e = self.e19[q]
                    opp_q = self.LR[q]
                    if even == 0:  # 偶数步：写邻居
                        ip = self.periodic_index(idx + e)
                        self.f[ip][opp_q] = f_collided[q]
                    else:  # 奇数步：写本地
                        self.f[idx][q] = f_collided[q]
            #---标量场更新---#
            g_local = ti.Vector([0.0] * 7) # 定义临时变量 不同标量场复用
            g_collided = ti.Vector([0.0] * 7)
            S_local = 0.0
            dS_local = 0.0
            tau_local = 0.0
            # 温度场更新
            ## 流体温度场
            if ti.static(self.TEMPERATURE):
                if  self.t[None] > self.TF_delay:
                    if self.solid[idx]<1:
                        # 读取离散速度分量
                        for q in ti.static(range(7)):
                            e = self.e7[q]
                            opp_q = self.LR[q]
                            if even == 0:  # 偶数步：从邻居读取
                                ip = self.periodic_index(idx - e)
                                g_local[q] = self.TF.g[ip][q]
                            else:  # 奇数步：从本格读取
                                g_local[q]= self.TF.g[idx][opp_q]
                        # 计算宏观量
                        S_local = 0.0
                        for q in ti.static(range(7)):
                            S_local += g_local[q]
                        self.TF.S[idx] = S_local
                        # 碰撞
                        tau_local = self.TF.tau(idx)
                        dS_local = self.TF.dS[idx]
                        for q in ti.static(range(7)):
                            geq = self.TF.geq7(q, S_local, idx[0], idx[1], idx[2])
                            gq = g_local[q] - (g_local[q] - geq) / tau_local # 碰撞项
                            gq += self.TF.geq7(q, dS_local, idx[0], idx[1], idx[2]) # 微观源项
                            g_collided[q] = gq
                        # 更新场
                        for q in ti.static(range(7)):
                            e = self.e7[q]
                            opp_q = self.LR[q]
                            if even == 0:  # 偶数步：写邻居
                                ip = self.periodic_index(idx + e)
                                self.TF.g[ip][opp_q] = g_collided[q]
                            else:  # 奇数步：写本地
                                self.TF.g[idx][q] = g_collided[q]
            ## 固体温度场
            if ti.static(self.TEMPERATURE):
                if self.t[None] > self.TS_delay:
                    if self.solid[idx]>0:
                        # 读取离散速度分量
                        for q in ti.static(range(7)):
                            e = self.e7[q]
                            opp_q = self.LR[q]
                            if even == 0:  # 偶数步：从邻居读取
                                ip = self.periodic_index(idx - e)
                                g_local[q] = self.TS.g[ip][q]
                            else:  # 奇数步：从本格读取
                                g_local[q]= self.TS.g[idx][opp_q]
                        # 计算宏观量
                        S_local = 0.0
                        for q in ti.static(range(7)):
                            S_local += g_local[q]
                        self.TS.S[idx] = S_local
                        # 碰撞
                        tau_local = self.TS.tau(idx)
                        dS_local = self.TS.dS[idx]
                        for q in ti.static(range(7)):
                            geq = self.TS.geq7(q, S_local, idx[0], idx[1], idx[2])
                            gq = g_local[q] - (g_local[q] - geq) / tau_local # 碰撞项
                            gq += self.TS.geq7(q, dS_local, idx[0], idx[1], idx[2]) # 微观源项
                            g_collided[q] = gq
                        # 更新场
                        for q in ti.static(range(7)):
                            e = self.e7[q]
                            opp_q = self.LR[q]
                            if even == 0:  # 偶数步：写邻居
                                ip = self.periodic_index(idx + e)
                                self.TS.g[ip][opp_q] = g_collided[q]
                            else:  # 奇数步：写本地
                                self.TS.g[idx][q] = g_collided[q]
            # 浓度场更新
            if ti.static(self.CHEMISTRY):
                if self.t[None] > self.chemistry_field_delay:
                    self.rhos[idx] = 0.0 # 更新固相密度场
                    for specie in ti.static(list(self.species)):
                        if ti.static(not specie.FIX):
                            # 流体组分更新
                            if self.solid[idx]<1:
                                # 读取离散速度分量
                                for q in ti.static(range(7)):
                                    e = self.e7[q]
                                    opp_q = self.LR[q]
                                    if even == 0:  # 偶数步：从邻居读取
                                        ip = self.periodic_index(idx - e)
                                        g_local[q] = specie.g[ip][q]
                                    else:  # 奇数步：从本格读取
                                        g_local[q]= specie.g[idx][opp_q]
                                # 计算宏观量
                                S_local = 0.0
                                for q in ti.static(range(7)):
                                    S_local += g_local[q]
                                specie.S[idx] = S_local
                                # 碰撞
                                tau_local = specie.tau(idx)
                                dS_local = specie.dS[idx]
                                for q in ti.static(range(7)):
                                    geq = specie.geq7(q, S_local, idx[0], idx[1], idx[2])
                                    gq = g_local[q] - (g_local[q] - geq) / tau_local # 碰撞项
                                    gq += specie.geq7(q, dS_local, idx[0], idx[1], idx[2]) # 微观源项
                                    g_collided[q] = gq
                                # 更新场
                                for q in ti.static(range(7)):
                                    e = self.e7[q]
                                    opp_q = self.LR[q]
                                    if even == 0:  # 偶数步：写邻居
                                        ip = self.periodic_index(idx + e)
                                        specie.g[ip][opp_q] = g_collided[q]
                                    else:  # 奇数步：写本地
                                        specie.g[idx][q] = g_collided[q]
                        else:
                            # 固体组分更新（仅源项，假设固体物种不迁移）
                            dS_local = specie.dS[idx]
                            specie.S[idx] += dS_local 
                            self.rhos[idx] += specie.S[idx] # 更新固相密度场
            # 更新源项场
            ## 温度场源项
            if ti.static(self.TEMPERATURE):
                if  self.t[None] > self.TF_delay and self.t[None] > self.TS_delay:
                    self.TS.dS[idx] = 0.0
                    self.TF.dS[idx] = 0.0
                    if self.solid[idx] > 0: # 有固体
                        # 流固热交换
                        if self.solid[idx]<1: # 有流体
                            dH = self.TS.exchangeCoef[idx]*self.TS.exchangeSurface[idx]*(self.TF.physical_value(self.TF.S[idx])-self.TS.physical_value(self.TS.S[idx]))*self.dt
                            self.TS.dS[idx] += dH/self.TS.capacity_m(idx)/self.rhos[idx]/self.TS.v_scale # 归一化温度变化 
                            self.TF.dS[idx] += -dH/self.TF.capacity_m(idx)/self.rho[idx]/self.TF.v_scale # 归一化温度变化
                        # 辐射
                        if ti.static(self.RADIATION):
                            self.TS.dS[idx] += self.TS.radiation(idx)*self.dt/self.TS.capacity_m(idx)/self.rhos[idx]/self.TS.v_scale # 归一化温度变化
            ## 化学反应源项
            if ti.static(self.CHEMISTRY) :
                if self.t[None] > self.chemistry_field_delay:
                    self.reactions.update_dS(idx)
        if ti.static(self.boundary_condition_model == BC_MODEL.NEE):
            self.Boundary_condition_NEE_AA()
        if ti.static(self.boundary_condition_model == BC_MODEL.ES):
            self.Boundary_condition_ES_AA() # not implemented
        # ========== 4. 更新 ET 奇偶步标记 ==========
        self.even_step[None] = 1 - self.even_step[None]
        self.t[None] += self.dt 
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
        # collistion + source term + streaming : merged kernels
        for i in ti.grouped(self.rho):
            if (self.solid[i] < 1):
                for k in ti.static(range(19)):
                    # collision f
                    f =self.f[i][k]-1/self.tau(i)*(self.f[i][k]-self.feq19(k,i[0],i[1],i[2]))
                    self.f[i][k] =f
                    if ti.static(k<7):
                        if ti.static(self.TEMPERATURE): # 流体温度更新 有固体更新固体温度
                            # 有气体
                            g = self.TF.g[i][k]-1/self.TF.tau(i)*(self.TF.g[i][k]-self.TF.geq7(k,self.TF.S[i],i[0],i[1],i[2]))
                            self.TF.g[i][k] = g
                            if self.solid[i] > 0.0: # 有固体
                                g = self.TS.g[i][k]-1/self.TS.tau(i)*(self.TS.g[i][k]-self.TS.geq7(k,self.TS.S[i],i[0],i[1],i[2]))
                                self.TS.g[i][k] = g
                        if ti.static(self.CHEMISTRY):      
                            for specie in ti.static(list(self.species)):
                                if ti.static(not specie.FIX): # 气体物质更新
                                    g = specie.g[i][k]-1/specie.tau(i)*(specie.g[i][k]-specie.geq7(k,specie.S[i],i[0],i[1],i[2]))
                                    specie.g[i][k] = g 

                    # source term f # 将预先计算好的源项施加到各个分布函数分量当中 (化学反应 体积热源（辐射）)
                    ## force term for fluid flow: volume force like gravity, darcy drag force
                    self.f[i][k] += self.forceTermGuo(k,i)
                    ## source term microscopic of scalar field
                    if ti.static(self.TEMPERATURE):
                        if ti.static(k<7):
                            self.TF.g[i][k] += self.TF.geq7(k,self.TF.dS[i],i[0],i[1],i[2])
                            if self.solid[i] > 0.0:
                                self.TS.g[i][k] += self.TS.geq7(k,self.TS.dS[i],i[0],i[1],i[2])
                    if ti.static(self.CHEMISTRY):      
                        if ti.static(k<7):
                            for specie in ti.static(list(self.species)):
                                if ti.static(not specie.FIX): # 气体物质更新
                                    specie.g[i][k] += self.TF.geq7(k,specie.dS[i],i[0],i[1],i[2])
                                else:
                                    if ti.static(k==0):
                                        specie.S[i] += specie.dS[i]
                    # streaming f->F
                    ip = self.periodic_index(i+self.e19[k]) # 更新之后的位置索引，默认边界视为周期边界
                    if (self.solid[ip]<1): # 如果更新之后的位置是液体 更新流体 气相温度 浓度 
                        self.F[ip][k] = self.f[i][k] # 直接更新F
                        if ti.static(k<7):
                            if ti.static(self.TEMPERATURE):
                                self.TF.G[ip][k] = self.TF.g[i][k]
                            if ti.static(self.CHEMISTRY):
                                for specie in ti.static(list(self.species)):
                                    if ti.static(not specie.FIX):
                                        specie.G[ip][k] = specie.g[i][k]
                    else: 
                        self.F[i][self.LR[k]] = self.f[i][k] # 如果不是 反弹 流固表面处理
                        if ti.static(k<7):
                            if ti.static(self.TEMPERATURE):
                                self.TF.G[i][self.LR[k]] = self.TF.g[i][k]
                            if ti.static(self.CHEMISTRY):
                                for specie in ti.static(list(self.species)):
                                    if ti.static(not specie.FIX):
                                        specie.G[i][self.LR[k]] = specie.g[i][k]
                    if (self.solid[i]) > 0.0: # 多孔介质区域的固体温度迁移
                        if(self.solid[ip] > 0): # 迁移之后有固体
                            if ti.static(k<7):
                                if ti.static(self.TEMPERATURE):
                                    self.TS.G[ip][k] = self.TS.g[i][k]
                        else: # 迁移之后没有固体当作绝热反弹边界
                            if ti.static(k<7):
                                if ti.static(self.TEMPERATURE):
                                    self.TS.G[i][self.LR[k]] = self.TS.g[i][k]

                        
            else: # 纯固体区域 只更新固体温度
                for k in ti.static(range(19)):
                    # collision
                    if ti.static(k<7):
                        if ti.static(self.TEMPERATURE):
                            g = self.TS.g[i][k]-1/self.TS.tau(i)*(self.TS.g[i][k]-self.TS.geq7(k,self.TS.S[i],i[0],i[1],i[2]))
                            self.TS.g[i][k] =g
                    # source term
                    # streaming
                    ip = self.periodic_index(i+self.e19[k]) # 更新之后的位置索引，默认边界视为周期边界
                    if (self.solid[ip]>0): # 如果更新之后的位置有固体  
                        if ti.static(k<7):
                            if ti.static(self.TEMPERATURE):
                                self.TS.G[ip][k] = self.TS.g[i][k]
                    else: # 更新之后无固体 迁移步骤中认为是绝热壁 热交换在源项中
                        if ti.static(k<7):
                            if ti.static(self.TEMPERATURE):
                                self.TS.G[i][self.LR[k]] = self.TS.g[i][k]

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
                            for s in ti.static(range(7)):
                                specie.g[i][s] = specie.G[i][s] # 更新G
                                specie.S[i] += specie.G[i][s]
                            if specie.S[i]<-self.tol:
                                specie.S[i]=-self.tol
                            Yall += specie.S[i]
                        else:
                            if ti.static(self.PORO): # 计算当前固体物质总密度
                                self.rhos[i]+=specie.S[i]
                    for specie in ti.static(list(self.species)):
                        if ti.static(not specie.FIX): # 更新流体组分
                            specie.S[i] /= Yall
                    # 化学反应
                    self.reactions.update_dS(i)
                    
                # if ti.static(self.PORO):
                #     if self.rho1[i] != 0 and self.rhos[i] != 0:
                #         self.solid[i] = self.rhos[i]/self.rho1[i] # 更新孔隙结构
                # # 宏观恢复温度场 并计算源项
                if ti.static(self.TEMPERATURE):
                    Tf = 0.0
                    Ts = 0.0
                    for s in ti.static(range(7)):
                        self.TF.g[i][s] = self.TF.G[i][s] # 更新G
                        self.TS.g[i][s] = self.TS.G[i][s] # 更新G
                        Tf += self.TF.G[i][s] # 计算温度
                        Ts += self.TS.G[i][s]
                    self.TF.S[i] = Tf
                    self.TS.S[i] = Ts
                    if self.solid[i] > 0: # 有固体
                        dH = self.TS.exchangeCoef[i]*self.TS.exchangeSurface[i]*(self.TF.physical_value(Tf)-self.TS.physical_value(Ts))*self.dt
                        self.TS.dS[i] += dH/self.TS.capacity_m(i)/self.rhos[i]/self.TS.v_scale
                        self.TF.dS[i] += -dH/self.TF.capacity_m(i)/self.rho[i]/self.TF.v_scale
                        if ti.static(self.RADIATION):
                            self.TS.dS[i] += self.TS.radiation(i)*self.dt/self.TS.capacity_m(i)/self.rhos[i]/self.TS.v_scale

    
    
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
    def feq19(self, s,rho,i,j,k): #计算平衡分布函数 
        u = self.v[i,j,k]
        eps = 1-self.solid[i,j,k]
        eu = self.e19[s].dot(u)
        uv = u.dot(u)
        feqout = 1.0
        if self.EOS==FLUID_STATE_EQUATION.INCOMPRESSIBLE:
            feqout = self.w19[s]*(rho+3.0*eu+4.5*eu*eu/(eps+1e-12)-1.5*uv/(eps+1e-12))        
        if self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS:
            feqout = self.w19[s]*rho*(1.0+3.0*eu+4.5*eu*eu/(eps+1e-12)-1.5*uv/(eps+1e-12))
        return feqout
    
    @ti.func
    def feq19_no_poro(self, s,rho,i,j,k): #计算平衡分布函数 
        u = self.v[i,j,k]
        eu = self.e19[s].dot(u)
        uv = u.dot(u)
        feqout = 1.0
        if self.EOS==FLUID_STATE_EQUATION.INCOMPRESSIBLE:
            feqout = self.w19[s]*(rho+3.0*eu+4.5*eu*eu-1.5*uv)        
        if self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS:
            feqout = self.w19[s]*rho*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
        return feqout
    @ti.func
    def viscosity(self,i): # in LU
        visco = 0.1
        if ti.static(self.viscosity_model==VISCOSITY_MODEL.CONSTANT):
            visco = self.visco*self.dt/self.dx**2
        elif ti.static(self.viscosity_model==VISCOSITY_MODEL.SUTHERLAND):
            T = self.GetTF(i)
            visco = self.sutherland_coef[0]*T**1.5/(T+self.sutherland_coef[1])*self.dt/self.dx**2
        elif ti.static(self.viscosity_model == VISCOSITY_MODEL.MIXTURE):
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(self.species):
                    if ti.static(not specie.FIX):
                        visco += specie.S[i]*specie.viscosity(i)
        return visco
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
                    F += (-self.viscosity(i)/perm*eps-beta*ti.math.length(self.v[i]))*self.v[i]
                elif ti.static(self.poro_model==PORO_MODEL.DARCY):
                    eps = 1.0-self.solid[i]
                    F +=  -eps*self.viscosity(i)*self.coefDarcy[i]*self.v[i]
                elif ti.static(self.poro_model==PORO_MODEL.DARCYFORCHHEIMER):
                    eps = 1.0-self.solid[i]
                    F += (-eps*self.viscosity(i)*self.coefDarcy[i]-eps*self.coefForchheimer[i]*ti.math.length(self.v[i]))*self.v[i]
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
            term = (1.0-1.0/2.0/tau)*self.w19[s]*(3.0*ti.math.dot(self.e19[s]-u/(eps+1e-12),F)\
              +9.0*self.e19[s].dot(u)*self.e19[s].dot(F)/(eps+1e-12))
        elif ti.static(self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS):
            term = (1.0-1.0/2.0/tau)*rho*self.w19[s]*(3.0*ti.math.dot(self.e19[s]-u/(eps+1e-12),F)\
              +9.0*self.e19[s].dot(u)*self.e19[s].dot(F)/(eps+1e-12))
        return term    
    @ti.func
    def scalarCorrectionTerm(self,k,duS,tau):
        return (1.0-1.0/2.0/tau)*3.0*self.w7[k]*self.e7[k].dot(duS)
    
    @ti.func
    def GetTF(self,i):
        TF = 273.15
        if ti.static(self.TEMPERATURE):
            TF = self.TF.physical_value(self.TF.S[i])
        return TF
    
    @ti.func
    def GetTS(self,i):
        TS = 273.15
        if ti.static(self.TEMPERATURE):
            TS = self.TS.physical_value(self.TS.S[i])
        return TS