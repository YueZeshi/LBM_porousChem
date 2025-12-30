import taichi as ti
from util.flag import *
from ._scalarField import ScalarField

@ti.data_oriented
class TemperatureFluid(ScalarField):
    def __init__(self,name,nx,ny,nz,lb3d,FIX = False):
        super().__init__(name,nx,ny,nz,lb3d,FIX)
            # 热
    @ti.func
    def capacity_v(self,i): # 流体的体积热容 SI
        cv = 0.0
        for specie in ti.static(list(self.LBM.species.values())):
            if ti.static(not specie.FIX):
                cv += specie.capacity_m(i)*specie.S[i]*self.LBM.rho[i]
        return cv
    @ti.func
    def coefDiff(self, i):
        D = 0.0
        if ti.static(self.LBM.CHEMISTRY):
            rhoc = 0.0
            k = 0.0
            for specie in ti.static(list(self.LBM.species.values())):
                # rho+=specie.S[i]
                if ti.static(not specie.FIX):
                    k+=specie.S[i]*specie.conductivity(i)
                    rhoc += specie.S[i]*self.LBM.rho[i]*specie.capacity_m(i)
            D += k*self.LBM.dt/rhoc/self.LBM.dx**2
        else:
            D += 0.1
        return D    
    @ti.func
    def geq7(self,k,T,x,y,z):
        u = self.LBM.v[x,y,z]
        eu = self.LBM.e7[k].dot(u)
        uv = u.dot(u)
        geqout=0.0
        geqout += self.LBM.w7[k]*T*(1.0 + (3.0*eu+4.5*eu**2-1.5*uv))
        return geqout

@ti.data_oriented
class TemperatureSolid(ScalarField):
    def __init__(self,name,nx,ny,nz,lb3d,FIX = False):
        super().__init__(name,nx,ny,nz,lb3d,FIX)
    @ti.func
    def coefDiff(self, i):
        D = 0.0
        if self.LBM.solid==0:
            D+=10000
        elif ti.static(self.LBM.CHEMISTRY):
            rhoc = 0.0
            rho = 0.0
            k = 0.0
            for specie in ti.static(list(self.LBM.species.values())):
                # rho+=specie.S[i]
                if ti.static(specie.FIX):
                    k+=specie.S[i]*specie.conductivity(i)
                    rho += specie.S[i]
                    rhoc += specie.S[i]*specie.capacity_m(i)
            D += k*self.LBM.dt/rhoc/rho/self.LBM.dx**2
        else:
            D += 0.1
        return D    

    @ti.func
    def capacity_v(self,i): # 所有物质的体积热容 按照孔隙率加权。流体密度为纯流体的密度 而固体密度为考虑孔隙率的密度 因此固体密度不需要再用孔隙率修正
        c = 0.0
        for specie in ti.static(list(self.LBM.species.values())):
            if ti.static(specie.FIX):
                c += specie.S[i]*specie.capacity_m(i)
        return  c # kgm-3Jkg-1K-1=Jm-3K-1 SI
    # @ti.func
    # def correction_SF(self,i):
    #     return self.capacity_v_all(i)/self.capacity_v_fluid(i)
    @ti.func
    def geq7(self,k,T,x,y,z):
        geqout=0.0
        geqout += self.LBM.w7[k]*T
        return geqout
    
@ti.data_oriented
class Reaction:
    R = 8.31
    def __init__(self,name,reactant,product,param,lb2d,unit=SPECIE_UNIT.MASS):
        self.reactionType = REACTION_TYPE.ARREHNIUS
        self.name = name
        self.A = float(param[0])
        self.b = float(param[1])
        self.Ea = float(param[2])
        self.Tmin = float(param[3])
        self.deltaH = float(param[4])
        self.LBM = lb2d
        self.unit = unit
        self.coefProduct = ti.field(ti.f32,shape=(len(self.LBM.species)))
        self.coefReactant = ti.field(ti.f32,shape=(len(self.LBM.species)))
        self.coefRate = ti.field(ti.f32,shape=(len(self.LBM.species)))
        for i in range(len(self.LBM.species)):
            for r in reactant:
                if r[0]==list(self.LBM.species.keys())[i]:
                    self.coefReactant[i] = r[1]
                    if len(r)==3:
                        self.coefRate[i]= r[2] # 可明确哪些物质浓度参与反应速率计算
                    else:
                        self.coefRate[i] =r[1] # 默认按照化学计量数参与计算
            for p in product:
                if p[0]==list(self.LBM.species.keys())[i]:
                    self.coefProduct[i]=p[1]
    @ti.func
    def Arrehnius(self,i):  # 温度从0开始演化的时候数值震荡可能会出现负温度，最好定义截止温度
        k = 0.0
        T = 0.0
        if ti.static(self.LBM.TEMPERATURE):
            # 有固相参与使用固相温度
            if self.LBM.solid[i]>0:
                T = self.LBM.TS.S[i]
            else:
                T = self.LBM.TF.S[i]
            if (T>self.Tmin):
                k = self.A*(T+1e-6)**self.b*ti.math.exp(-self.Ea/(T+1e-6)/Reaction.R)
        else:
            k = self.A
        return k
    @ti.func
    def reaction(self,i):  # mole修正还没有写好
        kr = self.Arrehnius(i)
        j = 0
        # 计算化学反应速率
        for specie in ti.static(list(self.LBM.species.values())):
            if self.coefReactant[j]>0 : # 该物质参与反应
                if specie.S[i]>1e-6:
                    if self.coefRate[j]!=0:#浓度参与化学反应速率计算
                        if ti.static(self.unit==SPECIE_UNIT.MASS):
                            if ti.static(not specie.FIX):
                                kr *= (specie.S[i]*self.LBM.rho[i])**self.coefRate[j] # 气态反应物 该物质对反应的贡献 可以不贡献
                            else:
                                kr *= specie.S[i]**self.coefRate[j] # 固态反应物 该物质对反应的贡献 可以不贡献
                        # elif ti.static(self.unit==SPECIE_UNIT.MOLE): # mole表示的化学反应需要按照摩尔质量修正因为物种信息存储的是密度信息
                        #     kr *= (specie.S[i]/specie.molemass)**self.coefRate[j] # 该物质对反应的贡献 可以不贡献
                else: # 某反应物不存在
                    kr = 0
            j += 1
        j = 0
        # 更新物种浓度和物种更新带来的焓变
        dh = 0.0 # 物种变化带来的焓变以及反应焓变
        for specie in ti.static(list(self.LBM.species.values())):
            coef = -self.coefReactant[j]+self.coefProduct[j] 
            if coef!=0.:
                ds = kr*coef*self.LBM.dt # 密度变化
                if ti.static(self.unit==SPECIE_UNIT.MOLE):
                    ds *= specie.molemass # 摩尔质量修正到密度
                if ti.static(specie.FIX):# 固体直接更新密度
                    specie.S[i] += ds
                else: # 流体更新分布函数
                    for k in ti.static(range(19)):
                        self.LBM.f[i][k] += self.LBM.feq19(k,ds,self.LBM.v[i]) # 更新密度分布函数
                        if ti.static(k<7):
                            specie.g[i][k] += specie.geq7(k,ds/self.LBM.rho[i],i[0],i[1],i[2]) # 更新物种质量分数分布函数，用气体密度用当前密度近似
                        # self.LBM.Temperature.g[i][k]+=self.LBM.w7[k]*ds*self.LBM.Temperature.S[i]*specie.capacity_m(i) # 流体的焓随麦克斯韦分布展开
                # 物质的生成和消失会影响焓变                    
                # dh += ds*self.LBM.Temperature.S[i]*specie.capacity_m(i) # 物种生成和消失带来的焓变
            j+=1
        # 反应热效应
        if ti.static(self.LBM.TEMPERATURE):
            dh += -kr*self.deltaH*self.LBM.dt # 注意保证kr deltaH的单位匹配。是质量都是质量，是摩尔数都是摩尔数。
            """
            热源项疑似会带来问题，热源项先还原到温度再按照平衡态分配进去。如果直接放在格点中心会导致热量在格点中心堆积，出现温度奇点
            """    
            for k in ti.static(range(7)):
                self.LBM.TS.g[i][k] += self.LBM.TS.geq7(k,dh/self.LBM.TS.capacity_v(i),i[0],i[1],i[2])
                
    @ti.func
    def reaction_macro(self): 
        for i in ti.grouped(self.LBM.rho):
            kr = self.Arrehnius(self.LBM.Temperature.S[i])
            j = 0
            for specie in ti.static(list(self.LBM.species.values())): 
                if self.coefReactant[j]>0: # 该物质参与反应
                    if specie.S[i]>1e-6:
                        if self.coefRate[j]!=0:#浓度参与化学反应速率计算
                            if ti.static(self.unit==SPECIE_UNIT.MASS):
                                kr *= specie.S[i]**self.coefRate[j] # 该物质对反应的贡献 可以不贡献
                            elif ti.static(self.unit==SPECIE_UNIT.MOLE): # mole表示的化学反应需要按照摩尔质量修正因为物种信息存储的是密度信息
                                kr *= (specie.S[i]/specie.molemass)**self.coefRate[j] # 该物质对反应的贡献 可以不贡献
                    else: # 某反应物不存在
                        kr = 0
                j += 1
            j = 0
            for specie in ti.static(list(self.LBM.species.values())):
                coef = -self.coefReactant[j]+self.coefProduct[j] 
                if coef!=0.:
                    ds = kr*coef*self.LBM.dt
                    if ti.static(self.unit==SPECIE_UNIT.MOLE):
                        ds *= specie.molemass # 摩尔质量修正到密度
                    specie.S[i] += ds
                    if not ti.static(specie.FIX):
                        self.LBM.rho[i] += ds # 更新密度分布函数
                    self.LBM.Temperature.S[i] += ds*self.LBM.Temperature.S[i]*specie.capacity_m(i)/self.LBM.Temperature.capacity_v_all(i)
                j+=1
            # 反应热效应
            de = -kr*self.deltaH*self.LBM.dt
            self.LBM.Temperature.S[i]+=de/self.LBM.Temperature.capacity_v_all(i) # 反应热直接更新温度场