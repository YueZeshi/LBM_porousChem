import taichi as ti
# from LBM.LBM2D import LBM2DSolver
from ._scalarField import ScalarField
from ..util.flag import *
@ti.data_oriented
class Specie(ScalarField): # 物种质量分数场
    def __init__(self,name,nx,ny,nz,lb3d,FIX = False,Mmass = 1.0):
        super().__init__(name,nx,ny,nz,lb3d,FIX)
        self.molemass = Mmass
    @ti.func
    def capacity_mole(self,i): #UDF
        return 2.5*8.314
    @ti.func
    def capacity_m(self,i): #UDF
        cm = self.capacity_mole(i)/self.molemass
        return cm
    @ti.func
    def conductivity(self,i): #UDF
        return 0.2
    
    def set_s_BC_Y(self,index,s):
        self.Y_BC[index]=s
    def set_s_BCs_Y(self,s):
        for i in range(4):
            self.Y_BC[i]=s[i]
    @ti.func
    def geq5(self,k,S,x,y,z):
        eu = self.LBM.e5[k].dot(self.LBM.v[x,y,z])
        return self.LBM.w5[k]*S*(1+3*eu)


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
        self.reactionResult = ti.field(ti.f32,shape=(len(self.LBM.species)+1)) # specie concentration and enthalpy change
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
                else: # no reaction including absence of catalyst
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
                self.reactionResult[j]=ds
                # # 物质的生成和消失会影响焓变                    
                # # dh += ds*self.LBM.Temperature.S[i]*specie.capacity_m(i) # 物种生成和消失带来的焓变
            j+=1
        # 反应热效应
        if ti.static(self.LBM.TEMPERATURE):
            dh += -kr*self.deltaH*self.LBM.dt # 注意保证kr deltaH的单位匹配。是质量都是质量，是摩尔数都是摩尔数。
            self.reactionResult[j]=dh
            
            
@ti.data_oriented
class Reactions:
    def __init__(self,lbm):
        self.LBM = lbm
        self.dS = [ti.field(ti.f32,shape = lbm.rho.shape)]*len(lbm.species)
        self.dH = ti.field(ti.f32,shape = lbm.rho.shape)
        self.reactions:list[Reaction] = []
        pass
    def add_reaction(self,reaction):
        self.reactions.append(reaction)
    @ti.func
    def update_dS(self,i): # 计算所有化学反应带来的物质源项和能量源项
        dS[i] = 0
        for r in self.reactions:
            r.reaction()
            j = 0
            for dS in ti.static(self.dS):
                dS[i] += r.reactionResult[j]
                j+=1
            self.dH[i] += r.reactionResult[j]