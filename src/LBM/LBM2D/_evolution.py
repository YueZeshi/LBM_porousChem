import taichi as ti
from ._core import LBM2D_BASE
from ._chemical import Specie, Reaction
from ._thermal import TemperatureFluid, TemperatureSolid
from ..util.flag import *
@ti.data_oriented
class LBM2D_EVOLUTION(LBM2D_BASE):
    """
    Evolution part of LBM2D
    """
    def step(self):
        self.updateBC(self.t)
        self.step_AA_kernel()
        self.tLattice += 1
        ti.sync()

    @ti.kernel
    def step_AA_kernel(self):
        for idx in ti.grouped(self.solid):
            if self.solid[idx] < 1.0:
                f_local = ti.Vector([0.0] * 9)
                for q in ti.static(range(9)):
                    e = self.e9[q]
                    opp_q = self.LR[q]
                    if self.even_step[None]:
                        ip = self.periodic_index(idx - e)
                        f_local[q] = self.f[ip][q]
                    else:
                        f_local[q] = self.f[idx][opp_q]

                rho = 0.0
                v = ti.Vector([0.0, 0.0, 0.0])
                for q in ti.static(range(9)):
                    fi = f_local[q]
                    rho += fi
                    v += self.e9[q] * fi
                
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
                self.v[idx] = v / (rho + 1e-12)
                self.rho[idx] = rho

                # ----- 1.2 碰撞（BGK/MRT）+ 体力源项（GUO） -----
                F = self.force(idx) # 计算体积力
                drho = self.drho[idx]
                self.drho[idx] = 0.0 # 化学反应引起的密度变化在碰撞后重置，下一步重新计算
                # 碰撞
                f_collided = ti.Vector([0.0] * 9)
                eps_local = 1.0 - self.solid[idx]
                if ti.static(self.collision_model == COLLISION_MODEL.MRT):
                    u_mrt = self.v[idx]
                    tau = self.tau(idx)
                    f_collided = self.collision_MRT_D2Q9(f_local, rho, u_mrt[0], u_mrt[1], tau, eps_local)
                else:
                    tau = self.tau(idx)
                    for q in ti.static(range(9)):
                        feq = self.feq9(q, rho, idx[0], idx[1], idx[2])
                        f_collided[q] = f_local[q] - (f_local[q] - feq) / tau
                # 体力源项 + 密度源项 (BGK/MRT 通用)
                for q in ti.static(range(9)):
                    f_collided[q] += self.forceTermGuo(q, idx, F)
                    f_collided[q] += self.feq9(q, drho, idx[0], idx[1], idx[2])

                # ----- 1.3 写入阶段（AA 迁移，写回 self.f） -----
                for q in ti.static(range(9)):
                    e = self.e9[q]
                    opp_q = self.LR[q]
                    if self.even_step[None]:
                        ip = self.periodic_index(idx + e)
                        self.f[ip][opp_q] = f_collided[q]
                    else:  # 奇数步：写本地
                        self.f[idx][q] = f_collided[q]
            #---标量场更新---#
            g_local = ti.Vector([0.0] * 5) # 定义临时变量 不同标量场复用
            g_collided = ti.Vector([0.0] * 5)
            S_local = 0.0
            dS_local = 0.0
            tau_local = 0.0
            # 温度场更新
            ## 流体温度场
            if ti.static(self.TEMPERATURE) :
                if self.t[None] > self.TF_delay[None]: # 温度场延迟更新:
                    if self.solid[idx]<1:
                        # 读取离散速度分量
                        for q in ti.static(range(5)):
                            e = self.e5[q]
                            opp_q = self.LR[q]
                            if self.even_step[None]:  # 偶数步：从邻居读取
                                ip = self.periodic_index(idx - e)
                                g_local[q] = self.TF.g[ip][q]
                            else:  # 奇数步：从本格读取
                                g_local[q]= self.TF.g[idx][opp_q]
                        # 计算宏观量
                        S_local = 0.0
                        for q in ti.static(range(5)):
                            S_local += g_local[q]
                        self.TF.S[idx] = S_local
                        # 碰撞 (BGK/MRT)
                        dS_local = self.TF.dS[idx]
                        if ti.static(self.collision_model == COLLISION_MODEL.MRT):
                            u_mrt = self.v[idx]
                            tau_local = self.TF.tau(idx)
                            g_collided = self.collision_MRT_D2Q5(g_local, S_local, u_mrt[0], u_mrt[1], tau_local)
                        else:
                            tau_local = self.TF.tau(idx)
                            for q in ti.static(range(5)):
                                geq = self.TF.geq5(q, S_local, idx[0], idx[1], idx[2])
                                g_collided[q] = g_local[q] - (g_local[q] - geq) / tau_local
                        # 微观源项 (BGK/MRT通用)
                        for q in ti.static(range(5)):
                            g_collided[q] += self.TF.geq5(q, dS_local, idx[0], idx[1], idx[2])
                        # 更新场
                        for q in ti.static(range(5)):
                            e = self.e5[q]
                            opp_q = self.LR[q]
                            if self.even_step[None]:  # 偶数步：写邻居
                                ip = self.periodic_index(idx + e)
                                self.TF.g[ip][opp_q] = g_collided[q]
                            else:  # 奇数步：写本地
                                self.TF.g[idx][q] = g_collided[q]
            ## 固体温度场
            if ti.static(self.TEMPERATURE):
                if self.t[None] > self.TS_delay[None]:
                    if self.solid[idx]>0:
                        # 读取离散速度分量
                        for q in ti.static(range(5)):
                            e = self.e5[q]
                            opp_q = self.LR[q]
                            if self.even_step[None]:  # 偶数步：从邻居读取
                                ip = self.periodic_index(idx - e)
                                g_local[q] = self.TS.g[ip][q]
                            else:  # 奇数步：从本格读取
                                g_local[q]= self.TS.g[idx][opp_q]
                        # 计算宏观量
                        S_local = 0.0
                        for q in ti.static(range(5)):
                            S_local += g_local[q]
                        self.TS.S[idx] = S_local
                        # 碰撞 (BGK/MRT)
                        dS_local = self.TS.dS[idx]
                        if ti.static(self.collision_model == COLLISION_MODEL.MRT):
                            u_mrt = self.v[idx]
                            tau_local = self.TS.tau(idx)
                            g_collided = self.collision_MRT_D2Q5(g_local, S_local, u_mrt[0], u_mrt[1], tau_local)
                        else:
                            tau_local = self.TS.tau(idx)
                            for q in ti.static(range(5)):
                                geq = self.TS.geq5(q, S_local, idx[0], idx[1], idx[2])
                                g_collided[q] = g_local[q] - (g_local[q] - geq) / tau_local
                        # 微观源项 (BGK/MRT通用)
                        for q in ti.static(range(5)):
                            g_collided[q] += self.TS.geq5(q, dS_local, idx[0], idx[1], idx[2])
                        # 更新场
                        for q in ti.static(range(5)):
                            e = self.e5[q]
                            opp_q = self.LR[q]
                            if self.even_step[None]:  # 偶数步：写邻居
                                ip = self.periodic_index(idx + e)
                                self.TS.g[ip][opp_q] = g_collided[q]
                            else:  # 奇数步：写本地
                                self.TS.g[idx][q] = g_collided[q]
            # 浓度场更新
            if ti.static(self.CHEMISTRY) :
                if self.t[None] > self.chemistry_field_delay[None]:
                    self.rhos[idx] = 0.0 # 更新固相密度场
                    self.inertSpecie.S[idx] = 1.0
                    for specie in ti.static(list(self.species)):
                        if ti.static(not specie.FIX):
                            if ti.static(not specie.isInert):
                                # 流体组分更新
                                if self.solid[idx]<1:
                                    # 读取离散速度分量
                                    for q in ti.static(range(5)):
                                        e = self.e5[q]
                                        opp_q = self.LR[q]
                                        if self.even_step[None]:  # 偶数步：从邻居读取
                                            ip = self.periodic_index(idx - e)
                                            g_local[q] = specie.g[ip][q]
                                        else:  # 奇数步：从本格读取
                                            g_local[q]= specie.g[idx][opp_q]
                                    # 计算宏观量
                                    S_local = 0.0
                                    for q in ti.static(range(5)):
                                        S_local += g_local[q]
                                    specie.S[idx] = S_local
                                    self.inertSpecie.S[idx] -= S_local # 惰性物质浓度由其他物质浓度反推
                                    # 碰撞 (BGK/MRT)
                                    dS_local = specie.dS[idx]/self.rho[idx] # 浓度源项归一化
                                    if ti.static(self.collision_model == COLLISION_MODEL.MRT):
                                        u_mrt = self.v[idx]
                                        tau_local = specie.tau(idx)
                                        g_collided = self.collision_MRT_D2Q5(g_local, S_local, u_mrt[0], u_mrt[1], tau_local)
                                    else:
                                        tau_local = specie.tau(idx)
                                        for q in ti.static(range(5)):
                                            geq = specie.geq5(q, S_local, idx[0], idx[1], idx[2])
                                            g_collided[q] = g_local[q] - (g_local[q] - geq) / tau_local
                                    # 微观源项 (BGK/MRT通用)
                                    for q in ti.static(range(5)):
                                        g_collided[q] += specie.geq5(q, dS_local, idx[0], idx[1], idx[2])
                                    # 更新场
                                    for q in ti.static(range(5)):
                                        e = self.e5[q]
                                        opp_q = self.LR[q]
                                        if self.even_step[None]:  # 偶数步：写邻居
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
                if self.t[None] > self.TF_delay[None] and self.t[None]>self.TS_delay[None]:
                    self.TS.dS[idx] = 0.0
                    self.TF.dS[idx] = 0.0
                    if self.solid[idx] > 0: # 有固体
                        # 流固热交换
                        if self.solid[idx] < 1: # 有流体
                            dH = self.TS.exchangeCoef[idx]*self.TS.exchangeSurface[idx]*(self.TF.physical_value(self.TF.S[idx])-self.TS.physical_value(self.TS.S[idx]))*self.dt # 热交换量
                            self.TS.dS[idx] += dH/self.TS.capacity_m(idx)/self.rhos[idx]/self.TS.v_scale # 归一化温度变化 
                            self.TF.dS[idx] += -dH/self.TF.capacity_m(idx)/self.rho[idx]/self.TF.v_scale # 归一化温度变化
                        # 辐射
                        if ti.static(self.RADIATION):
                            self.TS.dS[idx] += self.TS.radiation(idx)*self.dt/self.TS.capacity_m(idx)/self.rhos[idx]/self.TS.v_scale # 归一化温度变化
                ## 化学反应源项
            if ti.static(self.CHEMISTRY):
                if  self.t[None] > self.chemistry_field_delay[None]:
                    self.reactions.update_dS(idx)
        # ========== 3. 边界条件（仅 NEE / ES，基于 rho, v, f） ==========
        if ti.static(self.boundary_condition_model == BC_MODEL.NEE):
            self.Boundary_condition_NEE_AA()
        if ti.static(self.boundary_condition_model == BC_MODEL.ES):
            self.Boundary_condition_ES_AA() # not implemented

        # ========== 4. 更新 ET 奇偶步标记 ==========
        self.even_step[None] = not self.even_step[None]
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
        # collistion + source term + streaming : merge kernels
        for i in ti.grouped(self.rho):
            if self.solid[i] < 1:  # 流体和多孔介质区域
                F = self.force(i)
                for k in ti.static(range(9)):
                    # 碰撞 f
                    f = self.f[i][k] - 1 / self.tau(i) * (self.f[i][k] - self.feq9(k, i[0], i[1], i[2]))
                    self.f[i][k] = f

                    if ti.static(k < 5):
                        if ti.static(self.TEMPERATURE):  # 流体温度更新
                            g = self.TF.g[i][k] - 1 / self.TF.tau(i) * (self.TF.g[i][k] - self.TF.geq5(k, self.TF.S[i], i[0], i[1], i[2]))
                            self.TF.g[i][k] = g
                            if self.solid[i] > 0.0:  # 有固体
                                g = self.TS.g[i][k] - 1 / self.TS.tau(i) * (self.TS.g[i][k] - self.TS.geq5(k, self.TS.S[i], i[0], i[1], i[2]))
                                self.TS.g[i][k] = g

                        if ti.static(self.CHEMISTRY):
                            for specie in ti.static(list(self.species)):
                                if ti.static(not specie.FIX):  # 气体物质更新
                                    g = specie.g[i][k] - 1 / specie.tau(i) * (specie.g[i][k] - specie.geq5(k, specie.S[i], i[0], i[1], i[2]))
                                    specie.g[i][k] = g

                    # 源项 f
                    self.f[i][k] += self.forceTermGuo(k, i, F)

                    # 微观源项标量场
                    if ti.static(self.TEMPERATURE):
                        if ti.static(k < 5):
                            self.TF.g[i][k] += self.TF.geq5(k, self.TF.dS[i], i[0], i[1], i[2])
                            if self.solid[i] > 0.0:
                                self.TS.g[i][k] += self.TS.geq5(k, self.TS.dS[i], i[0], i[1], i[2])

                    if ti.static(self.CHEMISTRY):
                        if ti.static(k < 5):
                            for specie in ti.static(list(self.species)):
                                if ti.static(not specie.FIX):
                                    specie.g[i][k] += specie.geq5(k, specie.dS[i], i[0], i[1], i[2])
                                else:
                                    if ti.static(k == 0):
                                        specie.S[i] += specie.dS[i]

                    # 迁移 f -> F
                    ip = self.periodic_index(i + self.e9[k])
                    if self.solid[ip] < 1:  # 如果更新后的位置是液体
                        self.F[ip][k] = self.f[i][k]
                        if ti.static(k < 5):
                            if ti.static(self.TEMPERATURE):
                                self.TF.G[ip][k] = self.TF.g[i][k]
                            if ti.static(self.CHEMISTRY):
                                for specie in ti.static(list(self.species)):
                                    if ti.static(not specie.FIX):
                                        specie.G[ip][k] = specie.g[i][k]
                    else:  # 如果不是，反弹处理
                        self.F[i][self.LR[k]] = self.f[i][k]
                        if ti.static(k < 5):
                            if ti.static(self.TEMPERATURE):
                                self.TF.G[i][self.LR[k]] = self.TF.g[i][k]
                            if ti.static(self.CHEMISTRY):
                                for specie in ti.static(list(self.species)):
                                    if ti.static(not specie.FIX):
                                        specie.G[i][self.LR[k]] = specie.g[i][k]

                    if self.solid[i] > 0.0:  # 多孔介质区域的固体温度迁移
                        if self.solid[ip] > 0:  # 迁移后有固体
                            if ti.static(k < 5):
                                if ti.static(self.TEMPERATURE):
                                    self.TS.G[ip][k] = self.TS.g[i][k]
                        else:  # 迁移后无固体，绝热反弹边界
                            if ti.static(k < 5):
                                if ti.static(self.TEMPERATURE):
                                    self.TS.G[i][self.LR[k]] = self.TS.g[i][k]

            else:  # 纯固体区域，只更新固体温度
                for k in ti.static(range(9)):
                    if ti.static(k < 5):
                        if ti.static(self.TEMPERATURE):
                            g = self.TS.g[i][k] - 1 / self.TS.tau(i) * (self.TS.g[i][k] - self.TS.geq5(k, self.TS.S[i], i[0], i[1], i[2]))
                            self.TS.g[i][k] = g

                    ip = self.periodic_index(i + self.e9[k])
                    if self.solid[ip] > 0:  # 更新后有固体
                        if ti.static(k < 5):
                            if ti.static(self.TEMPERATURE):
                                self.TS.G[ip][k] = self.TS.g[i][k]
                    else:  # 更新后无固体，绝热壁
                        if ti.static(k < 5):
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
                self.v[i] /= (self.rho[i] + 1e-12)
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
                            if specie.S[i]<-self.tol:
                                specie.S[i]=-self.tol
                            Yall += specie.S[i]
                        else:
                            if ti.static(self.PORO): # 计算当前固体物质总密度
                                self.rhos[i] += specie.S[i]
                    for specie in ti.static(list(self.species)):
                        if ti.static(not specie.FIX): # 更新流体组分
                            specie.S[i] /= Yall # 归一化处理
                    # 化学反应
                    # 计算化学反应源项
                    self.reactions.update_dS(i)
                    
                # if ti.static(self.PORO):
                #     if self.rho1[i] != 0 and self.rhos[i] != 0:
                #         self.solid[i] = self.rhos[i]/self.rho1[i] # 更新孔隙结构
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

                    # 计算流固热交换和辐射换热
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
        return iout
    @ti.func
    def feq9(self, s,rho,i,j,k): #计算平衡分布函数
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
    def feq9_no_poro(self,s,rho,i,j,k):# 计算平衡分布函数 无多孔介质修正 用于初始化固体格点的分布函数
        u = self.v[i,j,k]
        eu = self.e9[s].dot(u)
        uv = u.dot(u)
        feqout = 1.0
        if self.EOS==FLUID_STATE_EQUATION.INCOMPRESSIBLE:
            feqout = self.w9[s]*(rho+3.0*eu+4.5*eu*eu-1.5*uv)        
        if self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS:
            feqout = self.w9[s]*rho*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)
        return feqout
    @ti.func
    def viscosity(self,i): # in LU
        visco = 0.1
        if ti.static(self.viscosity_model==VISCOSITY_MODEL.CONSTANT):
            visco = self.visco[None]*self.dt/self.dx**2
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
    def forceTermGuo(self,s,i,F): # 将力转化为分布函数源项 Guo Zhao 实际上是动量变化量
        rho = self.rho[i]
        u = self.v[i]
        tau = self.tau(i)
        eps = 1.0-self.solid[i]
        term = 0.0
        if ti.static(self.EOS==FLUID_STATE_EQUATION.INCOMPRESSIBLE):
            term = (1.0-1.0/2.0/tau)*self.w9[s]*(3.0*ti.math.dot(self.e9[s]-u/(eps+1e-12),F)\
              +9.0*self.e9[s].dot(u)*self.e9[s].dot(F)/(eps+1e-12))
        elif ti.static(self.EOS==FLUID_STATE_EQUATION.IDEAL_GAS):
            term = (1.0-1.0/2.0/tau)*rho*self.w9[s]*(3.0*ti.math.dot(self.e9[s]-u/(eps+1e-12),F)\
              +9.0*self.e9[s].dot(u)*self.e9[s].dot(F)/(eps+1e-12))
        return term
    @ti.func
    def scalarCorrectionTerm(self,k,duS,tau):
        return (1.0-1.0/2.0/tau)*3.0*self.w5[k]*self.e5[k].dot(duS)
    
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
    # ========= MRT Collision Operators =========
    @ti.func
    def mrt_transform_D2Q9(self, f):
        """D2Q9 分布函数 → 矩空间变换 m = M @ f
        使用已验证的M矩阵：m0=ρ, m1=e, m2=ε, m3=jx, m4=qx, m5=jy, m6=qy, m7=pxx, m8=pxy
        ti.field手动循环代替ti.Matrix.rows @ 避免9×9矩阵编译展开
        """
        m = ti.Vector([0.0] * 9)
        for i in ti.static(range(9)):
            val = 0.0
            for j in ti.static(range(9)):
                val += self.M9[i, j] * f[j]
            m[i] = val
        return m
    
    @ti.func
    def invmrt_transform_D2Q9(self, m):
        """矩空间 → D2Q9 分布函数逆变换 f = invM @ m"""
        f = ti.Vector([0.0] * 9)
        for i in ti.static(range(9)):
            val = 0.0
            for j in ti.static(range(9)):
                val += self.invM9[i, j] * m[j]
            f[i] = val
        return f
    
    @ti.func
    def mrt_eq_moments_D2Q9(self, rho, u, v, eps=1.0):
        """计算 D2Q9 矩空间平衡态 m_eq
        通过先计算分布空间平衡态 feq 再投影到矩空间，确保与 BGK 完全一致。
        由于 M 矩阵方向排序 ≠ e9 方向排序，硬编码矩公式会产生系统性偏差。
        u: x方向速度分量, v: y方向速度分量
        eps: 孔隙率，用于多孔介质 feq 二次项修正（与 feq9 一致）
        """
        usqr = u*u + v*v
        # 1) 先计算分布空间平衡态 feq（含孔隙率修正）
        feq = ti.Vector([0.0] * 9)
        for q in ti.static(range(9)):
            eu = self.e9[q].x * u + self.e9[q].y * v
            if self.EOS == FLUID_STATE_EQUATION.INCOMPRESSIBLE:
                feq[q] = self.w9[q] * (rho + 3.0*eu + 4.5*eu*eu/(eps+1e-12) - 1.5*usqr/(eps+1e-12))
            else:  # IDEAL_GAS
                feq[q] = self.w9[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu/(eps+1e-12) - 1.5*usqr/(eps+1e-12))
        # 2) 投影到矩空间: m_eq = M @ feq (确保方向排序一致)
        m_eq = ti.Vector([0.0] * 9)
        for i in ti.static(range(9)):
            val = 0.0
            for j in ti.static(range(9)):
                val += self.M9[i, j] * feq[j]
            m_eq[i] = val
        return m_eq
    @ti.func
    def mrt_relaxation_D2Q9(self, tau):
        """根据局部松弛时间 tau 计算 D2Q9 MRT 松弛率数组 s_mrt
        守恒矩(s=0): m0(ρ), m3(jx), m5(jy)
        剪切矩(s=ω=1/τ): m7(pxx), m8(pxy) — 决定运动粘度
        非守恒非剪切矩: 使用 setup_mrt_rates 预设值 (e, ε, qx, qy)
        """
        s_mrt = ti.Vector([0.0] * 9)
        # 守恒矩: 松弛率=0
        s_mrt[0] = 0.0         # m0 (ρ)
        s_mrt[3] = 0.0         # m3 (jx)
        s_mrt[5] = 0.0         # m5 (jy)
        # 非守恒矩: 使用预设值 (含体粘性和高阶模)
        s_mrt[1] = self.s_mrt[1]  # m1 (能量e)
        s_mrt[2] = self.s_mrt[2]  # m2 (能量平方ε)
        s_mrt[4] = self.s_mrt[4]  # m4 (qx)
        s_mrt[6] = self.s_mrt[6]  # m6 (qy)
        # 剪切矩: 实时关联局部τ (变粘度核心)
        omega = 1.0 / tau
        s_mrt[7] = omega       # m7 (pxx): 决定运动粘度 ν = cs^2*(τ-0.5)*dt
        s_mrt[8] = omega       # m8 (pxy): 剪切应力
        return s_mrt

    @ti.func
    def collision_MRT_D2Q9(self, f, rho, u, v, tau, eps=1.0):
        """D2Q9 MRT 碰撞算子：f* = invM @ (I - S) @ M @ f + invM @ S @ m_eq
        tau: 局部松弛时间，支持变粘度；s7(pxx)/s8(pxy) = 1/tau 实时计算
        eps: 孔隙率，用于多孔介质 feq 修正
        """
        m = self.mrt_transform_D2Q9(f)
        m_eq = self.mrt_eq_moments_D2Q9(rho, u, v, eps)
        s_mrt = self.mrt_relaxation_D2Q9(tau) # 获取当前松弛率数组
        # 矩空间松弛: s7,s8实时关联tau(变粘度核心)，其余读预存值
        # note: 避免局部变量 s 在 ti.static 编译时作用域问题，直接内联
        for q in ti.static(range(9)):
            m[q] = m[q] - s_mrt[q] * (m[q] - m_eq[q])
        
        return self.invmrt_transform_D2Q9(m)
    
    @ti.func
    def mrt_transform_D2Q5(self, g):
        """D2Q5 分布函数 → 矩空间变换 (用于标量输运)
        ti.field手动循环代替ti.Matrix.rows @
        """
        m = ti.Vector([0.0] * 5)
        for i in ti.static(range(5)):
            val = 0.0
            for j in ti.static(range(5)):
                val += self.M5[i, j] * g[j]
            m[i] = val
        return m
    
    @ti.func
    def invmrt_transform_D2Q5(self, m):
        """矩空间 → D2Q5 分布函数逆变换"""
        f = ti.Vector([0.0] * 5)
        for i in ti.static(range(5)):
            val = 0.0
            for j in ti.static(range(5)):
                val += self.invM5[i, j] * m[j]
            f[i] = val
        return f

    @ti.func
    def invmrt_transform_D2Q5(self, m):
        """矩空间 → D2Q5 分布函数逆变换"""
        inv5 = 0.2
        inv4 = 0.25
        f = ti.Vector([0.0] * 5)
        f[0] = inv5*m[0] - inv5*m[3]
        f[1] = inv5*m[0] + inv4*m[1] + 0.2*m[3] + inv4*m[4]
        f[2] = inv5*m[0] + inv4*m[2] + 0.2*m[3] - inv4*m[4]
        f[3] = inv5*m[0] - inv4*m[1] + 0.2*m[3] + inv4*m[4]
        f[4] = inv5*m[0] - inv4*m[2] + 0.2*m[3] - inv4*m[4]
        return f
    
    @ti.func
    def mrt_eq_moments_D2Q5(self, scalar, ux, uy):
        """计算 D2Q5 矩空间平衡态"""
        m_eq = ti.Vector([0.0] * 5)
        m_eq[0] = scalar                                   # m0 = φ
        m_eq[1] = scalar * ux * (1.0/3.0)                  # m1 = jx_eq
        m_eq[2] = scalar * uy * (1.0/3.0)                  # m2 = jy_eq
        m_eq[3] = scalar * (-2.0/3.0)                      # m3 = e_eq
        m_eq[4] = 0.0                                       # m4 = ε_eq
        return m_eq
    @ti.func
    def collision_MRT_D2Q5(self, g, scalar, ux, uy, tau):
        """D2Q5 MRT 碰撞算子 (用于标量输运如温度/组分)
        tau: 局部松弛时间，支持变扩散系数；s1,s2 = 1/tau 实时计算
        """
        m = self.mrt_transform_D2Q5(g)
        m_eq = self.mrt_eq_moments_D2Q5(scalar, ux, uy)
        s = self.mrt_relaxation_D2Q5(tau) # 获取当前松弛率数组
        for q in ti.static(range(5)):
            m[q] = m[q] - s[q] * (m[q] - m_eq[q])
        
        return self.invmrt_transform_D2Q5(m)
    @ti.func
    def mrt_relaxation_D2Q5(self, tau):
        """根据局部松弛时间 tau 计算 D2Q5 MRT 松弛率数组 s
        s1, s2 实时关联 tau，其他使用预设值
        """
        s = ti.Vector([0.0] * 5)
        s[0] = 0.0  # m0 (标量密度): 守恒矩
        omega = 1.0 / tau
        s[1] = omega  # m1 (jx): 通量, 决定扩散系数 D = cs^2*(τ-0.5)*dt
        s[2] = omega  # m2 (jy): 通量
        s[3] = self.s_mrt_q5[3]  # m3 (e): 高阶矩
        s[4] = self.s_mrt_q5[4]  # m4 (ε): 高阶矩
        return s
    