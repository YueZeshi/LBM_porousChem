import taichi as ti

from ._scalarField import ScalarField
from ..util.flag import *
from ..util import constant
@ti.data_oriented
class Specie(ScalarField): # 物种质量分数场
    def __init__(self,name,lb,FIX = False,Mmass = 1.0):
        super().__init__(name,lb,FIX)
        self.molemass = Mmass/1000
        if not FIX:
            self.viscosity_type = VISCOSITY_MODEL.NONE
            self.visco = 1e-5
            self.coefSutherland = [0,0]
            self.diff_model = DIFF_MODEL.CONSANT
            self.diff = 1e-5
        self.thermo_model = THERMO_MODEL.CONSTANT
        self.enthaply = 100
        self.capa = 100
        self.Trange = [0,0,0]
        self.NASAcoef = [[0]*7]*2
        self.Sc = 1.0
        self.cond_model = CONDUCTIVITY_MODEL.CONSTANT
        self.cond = 1.0
        self.cond_poly = [0,0,0,0,0]
    def __str__(self):
        des = f"{self.name} : \n" 
        des +=f"        - Molar Mass : {self.molemass} kg/mol\n"
        des +="        - State : "
        if self.FIX:
            des += "SOLID\n"
        else: 
            des +="FLUID\n"
        if not self.FIX:
            des += "        - Viscosity model : "
            if self.viscosity_type==VISCOSITY_MODEL.CONSTANT:
                des += f"constant {self.visco}\n"
            elif self.viscosity_type==VISCOSITY_MODEL.SUTHERLAND:
                des += f"sutherland {self.coefSutherland}\n"
            else:
                des += "not valid\n"
        
        return des
    def __repr__(self):
        return self.__str__()
    @ti.func
    def coefDiff(self,i):
        diff = 0.0
        if ti.static(self.diff_model==DIFF_MODEL.CONSANT):
            diff += self.diff*self.LBM.dt/self.LBM.dx**2
        elif ti.static(self.diff_model==DIFF_MODEL.SCHMIDT):
            diff += self.Sc*self.viscosity(i)*self.LBM.dt/self.LBM.dx**2
        return diff
    @ti.func
    def viscosity(self,i): # in LU
        visco = 1e-5
        if ti.static(self.viscosity_type==VISCOSITY_MODEL.CONSTANT):
            visco = self.visco*self.LBM.dt/self.LBM.dx**2
        elif ti.static(self.viscosity_type==VISCOSITY_MODEL.SUTHERLAND):
            T = self.LBM.GetTF(i)
            visco = self.coefSutherland[0]*T**1.5/(T+self.coefSutherland[1])
        return visco
    @ti.func
    def capacity_mole(self,T): # T in Kelvin
        capa = 0.0
        if ti.static(self.thermo_model==THERMO_MODEL.CONSTANT):
            capa += self.capa
        elif ti.static(self.thermo_model==THERMO_MODEL.NASA7):
            if T < self.Trange[1]:
                capa = (((self.NASAcoef1[4]*T+self.NASAcoef1[3])*T+self.NASAcoef1[2])*T+self.NASAcoef1[1])*T+self.NASAcoef1[0]
            else:
                capa = (((self.NASAcoef2[4]*T+self.NASAcoef2[3])*T+self.NASAcoef2[2])*T+self.NASAcoef2[1])*T+self.NASAcoef2[0]
            capa *=constant.R
        return capa
    @ti.func
    def enthalpy_mole(self,T): 
        H = 0.0
        if ti.static(self.thermo_model==THERMO_MODEL.CONSTANT):
            H += self.enthaply
        elif ti.static(self.thermo_model==THERMO_MODEL.NASA7):
            if T < self.Trange[1]:
                H = ((((self.NASAcoef1[4]*T/5+self.NASAcoef1[3]/4)*T+self.NASAcoef1[2]/3)*T+self.NASAcoef1[1]/2)*T+self.NASAcoef1[0])*T+self.NASAcoef1[5]        
            else:
                H = ((((self.NASAcoef2[4]*T/5+self.NASAcoef2[3]/4)*T+self.NASAcoef2[2]/3)*T+self.NASAcoef2[1]/2)*T+self.NASAcoef2[0])*T+self.NASAcoef2[5]
            H *=constant.R*H*T
        return H
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
        cm = self.capacity_mole(T)/self.molemass
        return cm
    @ti.func
    def enthalpy_m(self,T):
        Hm = self.enthalpy_mole(T)/self.molemass
        return Hm
    
    @ti.func
    def conductivity(self,i): #UDF
        cond = 0.0
        if ti.static(self.cond_model==CONDUCTIVITY_MODEL.CONSTANT):
            cond += self.cond
        elif ti.static(self.cond_model==CONDUCTIVITY_MODEL.POLYNOMIAL):
            T = self.LBM.GetTF(i)
            cond+=(((self.cond_poly[4]*T+self.cond_poly[3])*T+self.cond_poly[2])*T+self.cond_poly[1])*T+self.cond_poly[0]
        return cond
    
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
        if ti.static(self.LBM.TEMPERATURE):
            if self.LBM.solid[i] > 0:
                self.LBM.TS.dS[i] += self.dS[i][self.specieNum]/self.LBM.TS.capacity_m(i)/self.LBM.rhos[i]
            else:
                self.LBM.TF.dS[i] += self.dS[i][self.specieNum]/self.LBM.TF.capacity_m(i)/self.LBM.rho[i]