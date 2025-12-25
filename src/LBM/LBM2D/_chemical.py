import taichi as ti

from ._scalarField import ScalarField
from ..util.flag import *
from ..util import constant
@ti.data_oriented
class Specie(ScalarField): # 物种质量分数场
    def __init__(self,name,lb3d,FIX = False,Mmass = 1.0):
        super().__init__(name,lb3d,FIX)
        self.molemass = Mmass
        self.Trange = ti.Vector.field(3,float,shape=())
        self.NASAcoef1 = ti.Vector.field(7,float,shape=())
        self.NASAcoef2 = ti.Vector.field(7,float,shape=())
        self.coefSutherland = ti.Vector.field(2,float,shape = ())
        self.default_init_NASA()
    def default_init_NASA(self):
        for i in range(3):
            self.Trange[i]=0.0
        for i in range(7):
            self.NASAcoef1[i]=0.0
            self.NASAcoef2[i]=0.0
    def __str__(self):
        return f"Specie: {self.name}, Molar Mass: {self.molemass} kg/mol, FIX: {self.FIX}"
    def __repr__(self):
        return self.__str__()
    
    @ti.func
    def capacity_mole(self,T): # T in Kelvin
        c_R = 1
        if T < self.Trange[1]:
            c_R = (((self.NASAcoef1[4]*T+self.NASAcoef1[3])*T+self.NASAcoef1[2])*T+self.NASAcoef1[1])*T+self.NASAcoef1[0]
        else:
            c_R = (((self.NASAcoef2[4]*T+self.NASAcoef2[3])*T+self.NASAcoef2[2])*T+self.NASAcoef2[1])*T+self.NASAcoef2[0]
        return c_R*constant.R
    @ti.func
    def enthalpy_mole(self,T): 
        H_RT = 1.0
        if T < self.Trange[1]:
            H_RT = ((((self.NASAcoef1[4]*T/5+self.NASAcoef1[3]/4)*T+self.NASAcoef1[2]/3)*T+self.NASAcoef1[1]/2)*T+self.NASAcoef1[0])*T+self.NASAcoef1[5]        
        else:
            H_RT = ((((self.NASAcoef2[4]*T/5+self.NASAcoef2[3]/4)*T+self.NASAcoef2[2]/3)*T+self.NASAcoef2[1]/2)*T+self.NASAcoef2[0])*T+self.NASAcoef2[5]
        return H_RT*constant.R*T
    @ti.func
    def entropy_mole(self,T):
        S_R = 1.0
        if T < self.Trange[1]:
            S_R = (((self.NASAcoef1[4]*T/4+self.NASAcoef1[3]/3)*T+self.NASAcoef1[2]/2)*T+self.NASAcoef1[1])*T+self.NASAcoef1[0]*ti.log(T)+self.NASAcoef1[6]        
        else:
            S_R = (((self.NASAcoef2[4]*T/4+self.NASAcoef2[3]/3)*T+self.NASAcoef2[2]/2)*T+self.NASAcoef2[1])*T+self.NASAcoef2[0]*ti.log(T)+self.NASAcoef2[6]  
        return S_R*constant.R
    @ti.func
    def capacity_m(self,T):
        # cm = self.capacity_mole(T)/self.molemass
        cm = 1.0
        return cm
    
    @ti.func
    def conductivity(self,i): #UDF
        return 0.2
    
    @ti.func
    def geq5(self,k,S,x,y,z):
        u = self.LBM.v[x,y,z]
        eu = self.LBM.e5[k].dot(u)
        geqout = 0.0
        geqout += self.LBM.w5[k]*S*(1+3*eu)
        return geqout

@ti.data_oriented
class Reaction:
    R = 8.31
    def __init__(self,formula,A,Ea,b = 0,Tmin = 0,deltaH = 0,lb2d=None,name="unnamed reaction",unit = SPECIE_UNIT.MASS,fixDH = True):
        self.reactionType = REACTION_TYPE.ARREHNIUS
        self.formula = formula
        self.name = name
        self.A = A
        self.b = b
        self.Ea = Ea
        self.Tmin = Tmin
        self.deltaH = deltaH
        self.LBM = lb2d
        self.unit = unit
        self.isFixDH = fixDH
        self.coefProduct = ti.field(float,shape=(len(self.LBM.species)))
        self.coefReactant = ti.field(float,shape=(len(self.LBM.species)))
        self.coefRate = ti.field(float,shape=(len(self.LBM.species)))

        self.reactionResult = ti.Vector.field(len(self.LBM.species),dtype=float,shape=self.LBM.rho.shape) # specie concentration and enthalpy change
        self.dH = ti.field(float,shape=self.LBM.rho.shape)
        self.parse_formula(formula)
    def parse_formula(self,formula:str):
        reactant,product = formula.split("=>")
        reactant = [specie.strip() for specie in reactant.split('+')]
        product = [specie.strip() for specie in product.split('+')]
        for term in reactant:
            terms = term.split()
            coef = 1
            specie = ""
            if len(terms)>1:
                coef = float(terms[0])
                specie = terms[1]
            else:
                specie = terms[0]
            self.coefReactant[self.LBM.specieName.index(specie)] = coef # chemkin 中默认基元反应，没有反应阶数
        for term in product:
            terms = term.split()
            coef = 1
            specie = ""
            if len(terms)>1:
                coef = float(terms[0])
                specie = terms[1]
            else:
                specie = terms[0]
            self.coefProduct[self.LBM.specieName.index(specie)] = coef

        # for i in range(len(self.LBM.species)):
        #     for r in reactant:
        #         if r[0]==self.LBM.specieName[i]:
        #             self.coefReactant[i] = r[1]
        #             if len(r)==3:
        #                 self.coefRate[i] = r[2] # 可明确哪些物质浓度参与反应速率计算
        #             else:
        #                 self.coefRate[i] = r[1] # 默认按照化学计量数参与计算
        #     for p in product:
        #         if p[0]==self.LBM.specieName[i]:
        #             self.coefProduct[i]=p[1]
    def __str__(self):
        return self.formula
    def __repr__(self):
        return self.__str__()
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
        dS = ti.Vector([0.0]*self.specieNum)
        # # 计算化学反应速率
        # for j in ti.static(range(len(self.LBM.species))):
        #     if self.coefReactant[j]>0 : # 该物质参与反应
        #         if self.LBM.species[j].S[i]>1e-6: # 存在该物质
        #             if self.coefRate[j] != 0:#浓度参与化学反应速率计算
        #                 if ti.static(self.unit==SPECIE_UNIT.MASS):
        #                     if ti.static(not self.LBM.species[j].FIX):
        #                         kr *= (self.LBM.species[j].S[i]*self.LBM.rho[i])**self.coefRate[j] # 气态反应物 该物质对反应的贡献 可以不贡献
        #                     else:
        #                         kr *= self.LBM.species[j].S[i]**self.coefRate[j] # 固态反应物 该物质对反应的贡献 可以不贡献
        #                 # elif ti.static(self.unit==SPECIE_UNIT.MOLE): # mole表示的化学反应需要按照摩尔质量修正因为物种信息存储的是密度信息
        #                 #     kr *= (specie.S[i]/specie.molemass)**self.coefRate[j] # 该物质对反应的贡献 可以不贡献
        #         else: # no reaction including absence of catalyst
        #             kr = 0
        # # 计算物种浓度变化和焓变
        # dh = 0.0 # 物种变化带来的焓变以及反应焓变
        # for j in ti.static(range(len(self.LBM.species))):
        #     coef = -self.coefReactant[j]+self.coefProduct[j] # 物质生成或者消耗
        #     if coef != 0.:
        #         ds = kr*coef*self.LBM.dt # 密度变化
        #         if ti.static(self.unit==SPECIE_UNIT.MOLE):
        #             ds *= self.LBM.species[j].molemass # 摩尔质量修正到密度
        #         self.reactionResult[i][j]=ds
        #         # # 物质的生成和消失会影响焓变                    
        #         # # dh += ds*self.LBM.Temperature.S[i]*specie.capacity_m(i) # 物种生成和消失带来的焓变
        # # 反应热效应
        # if ti.static(self.LBM.TEMPERATURE):
        #     dh += -kr*self.deltaH*self.LBM.dt # 注意保证kr deltaH的单位匹配。是质量都是质量，是摩尔数都是摩尔数。
        #     self.dH[i]=dh
        return dS
@ti.data_oriented
class Reactions:
    def __init__(self,lbm):
        self.LBM = lbm
        self.dS = None # initialize during init_python
        self.dH = ti.field(float,shape=self.LBM.rho.shape)
        self.reactions:list[Reaction] = []
        self.specieNum = 0
    def __str__(self):
        description = "Reactions:\n"
        for r in self.reactions:
            description += f" - {r.name}\n"
            description += "   "+r.__str__() + "\n"
        description += f"{len(self.reactions)} reactions in total.\n"
        return description 
    def __repr__(self):
        return self.__str__()
    def add_reaction(self,reaction):
        self.reactions.append(reaction)

    @ti.func
    def update_dS(self,i): # 计算所有化学反应带来的物质源项和能量源项
        for j in ti.static(range(self.specieNum)): 
            self.dS[i][j] = 0
        self.dH[i] = 0.0 # Set to 0
        for r in ti.static(self.reactions): # reaction update
            self.dS[i]+=r.reaction(i)
            # r.reaction(i)
            # for j in ti.static(range(self.specieNum)):
            #     self.dS[i][j] += r.reactionResult[i][j]
            # self.dH[i] += r.dH[i]
        for j in ti.static(range(self.specieNum)):
            self.LBM.species[j].dS[i] = self.dS[i][j]
        # for specie in ti.static(list(self.LBM.species)):
        #     specie.dS[i] = self.dS[i][j]
            # j += 1
        if ti.static(self.LBM.TEMPERATURE):
            if self.LBM.solid[i] > 0:
                self.LBM.TS.dS[i] += self.dS[i][self.specieNum]/self.LBM.TS.capacity_v(i)
            else:
                self.LBM.TF.dS[i] += self.dS[i][self.specieNum]/self.LBM.TF.capacity_v(i)