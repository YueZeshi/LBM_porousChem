import taichi as ti

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
    def geq7(self,k,S,x,y,z):
        u = self.LBM.v[x,y,z]
        eu = self.LBM.e7[k].dot(u)
        uv = u.dot(u)
        geqout = 0.0
        geqout += self.LBM.w7[k]*S*(1+4.0*eu+8.0*eu*eu-2*uv)
        return geqout

@ti.data_oriented
class Reaction:
    R = 8.31
    def __init__(self,name,reactant,product,param,lb3d,unit=SPECIE_UNIT.MASS):
        self.reactionType = REACTION_TYPE.ARREHNIUS
        self.name = name
        self.A = float(param[0])
        self.b = float(param[1])
        self.Ea = float(param[2])
        self.Tmin = float(param[3])
        self.deltaH = float(param[4])
        self.LBM= lb3d
        self.unit = unit
        self.coefProduct = ti.field(float,shape=(len(self.LBM.species)))
        self.coefReactant = ti.field(float,shape=(len(self.LBM.species)))
        self.coefRate = ti.field(float,shape=(len(self.LBM.species)))
        self.reactionResult = ti.Vector.field(len(self.LBM.species),dtype=float,shape=self.LBM.rho.shape) # specie concentration and enthalpy change
        self.dH = ti.field(float,shape=self.LBM.rho.shape)
        for i in range(len(self.LBM.species)):
            for r in reactant:
                if r[0]==self.LBM.specieName[i]:
                    self.coefReactant[i] = r[1]
                    if len(r)==3:
                        self.coefRate[i]= r[2] # 可明确哪些物质浓度参与反应速率计算
                    else:
                        self.coefRate[i] =r[1] # 默认按照化学计量数参与计算
            for p in product:
                if p[0]==self.LBM.specieName[i]:
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
        # 计算化学反应速率
        for j in ti.static(range(len(self.LBM.species))):
            if self.coefReactant[j]>0 : # 该物质参与反应
                if self.LBM.species[j].S[i]>1e-6: # 存在该物质
                    if self.coefRate[j]!=0:#浓度参与化学反应速率计算
                        if ti.static(self.unit==SPECIE_UNIT.MASS):
                            if ti.static(not self.LBM.species[j].FIX):
                                kr *= (self.LBM.species[j].S[i]*self.LBM.rho[i])**self.coefRate[j] # 气态反应物 该物质对反应的贡献 可以不贡献
                            else:
                                kr *= self.LBM.species[j].S[i]**self.coefRate[j] # 固态反应物 该物质对反应的贡献 可以不贡献
                        # elif ti.static(self.unit==SPECIE_UNIT.MOLE): # mole表示的化学反应需要按照摩尔质量修正因为物种信息存储的是密度信息
                        #     kr *= (specie.S[i]/specie.molemass)**self.coefRate[j] # 该物质对反应的贡献 可以不贡献
                else: # no reaction including absence of catalyst
                    kr = 0
        # 计算物种浓度变化和焓变
        dh = 0.0 # 物种变化带来的焓变以及反应焓变
        for j in ti.static(range(len(self.LBM.species))):
            coef = -self.coefReactant[j]+self.coefProduct[j] # 物质生成或者消耗
            if coef != 0.:
                ds = kr*coef*self.LBM.dt # 密度变化
                if ti.static(self.unit==SPECIE_UNIT.MOLE):
                    ds *= self.LBM.species[j].molemass # 摩尔质量修正到密度
                self.reactionResult[i][j]=ds
                # # 物质的生成和消失会影响焓变                    
                # # dh += ds*self.LBM.Temperature.S[i]*specie.capacity_m(i) # 物种生成和消失带来的焓变
        # 反应热效应
        if ti.static(self.LBM.TEMPERATURE):
            dh += -kr*self.deltaH*self.LBM.dt # 注意保证kr deltaH的单位匹配。是质量都是质量，是摩尔数都是摩尔数。
            self.dH[i]=dh

@ti.data_oriented
class Reactions:
    def __init__(self,lbm):
        self.LBM = lbm
        self.dS = None # initialize during init_python
        self.dH = ti.field(float,shape=self.LBM.rho.shape)
        self.reactions:list[Reaction] = []
        self.specieNum = 0

    def add_reaction(self,reaction):
        self.reactions.append(reaction)

    @ti.func
    def update_dS(self,i): # 计算所有化学反应带来的物质源项和能量源项
        for j in ti.static(range(self.specieNum)):
            self.dS[i][j] = 0
        self.dH[i] = 0.0
        for r in ti.static(self.reactions):
            r.reaction(i)
            for j in ti.static(range(self.specieNum)):
                self.dS[i][j] += r.reactionResult[i][j]
            self.dH[i] += r.dH[i]
        for j in ti.static(range(self.specieNum)):
            self.LBM.species[j].dS[i] = self.dS[i][j]
        if ti.static(self.LBM.TEMPERATURE):
            if self.LBM.solid[i] > 0:
                self.LBM.TS.dS[i] += self.dH[i]/self.LBM.TS.capacity_v(i)
            else:
                self.LBM.TF.dS[i] += self.dH[i]/self.LBM.TF.capacity_v(i)