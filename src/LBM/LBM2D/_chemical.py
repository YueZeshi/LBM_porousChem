from sphinx.addnodes import desc
import taichi as ti

from ._scalarField import ScalarField
from ..util.flag import *
from ..util import constant
@ti.data_oriented
class Specie(ScalarField): # 物种质量分数场
    def __init__(self,name,lb,FIX = False,Mmass = 1.0,unit = UNIT.MOLE):
        super().__init__(name,lb,FIX)
        self.molemass = Mmass / 1000
        self.unit = unit

        if not FIX:
            self.viscosity_type = VISCOSITY_MODEL.CONSTANT
            self.visco = ti.field(float,shape=())
            self.visco[None] = 1e-5
            self.coefSutherland = [0,0]
            self.diff_model = DIFF_MODEL.CONSTANT
            self.diff = 1e-5

        self.thermo_model = THERMO_MODEL.CONSTANT
        self.enthalpy = 100.0
        self.capa = 100.0
        self.Trange = ti.field(float,shape=(3))
        # self.Trange = [0,0,0]
        self.NASAcoef = ti.field(float,shape=(2,7))
        # self.NASAcoef = [[0]*7]*2
        self.capa_poly = [0]*5
        self.capa_unit = UNIT.MOLE
        self.Sc = 1.0
        self.cond_model = CONDUCTIVITY_MODEL.CONSTANT
        self.cond = 1.0
        self.cond_poly = [0.0]*5

    def description(self):
        des = f"{self.name} : \n"
        des += f"        - Molar Mass : {self.molemass} kg/mol\n"
        des += "        - State : "
        des += "SOLID\n" if self.FIX else "FLUID\n"

        if not self.FIX:
            des += "        - Viscosity model : "
            if self.viscosity_type == VISCOSITY_MODEL.CONSTANT:
                des += f"constant {self.visco[None]}\n"
            elif self.viscosity_type == VISCOSITY_MODEL.SUTHERLAND:
                des += f"sutherland {self.coefSutherland}\n"
            else:
                des += "not valid\n"

        des += "        - Thermodynamic model : "
        if self.thermo_model==THERMO_MODEL.CONSTANT:
            des += f"constant : enthalpy {self.enthalpy}, capacity {self.capa}, unit {self.capa_unit}\n"
        elif self.thermo_model==THERMO_MODEL.NASA7:
            des += f"NASA7 : Trange {self.Trange}, coef {self.NASAcoef}\n"
        elif self.thermo_model==THERMO_MODEL.POLYNOMIAL:
            des += f"polynomial : capa_poly {self.capa_poly}, unit {self.capa_unit}\n"
        else:
            des += "not valid\n"
        
        des += "        - Conductivity model : "
        if self.cond_model==CONDUCTIVITY_MODEL.CONSTANT:
            des += f"constant : conductivity {self.cond}\n"
        elif self.cond_model==CONDUCTIVITY_MODEL.POLYNOMIAL:
            des += f"polynomial : coef {self.cond_poly}\n"
        else:
            des += "not valid\n"
        return des
    def __repr__(self):
        return self.__str__()
    @ti.func
    def coefDiff(self,i):
        diff = 0.0
        if ti.static(self.diff_model==DIFF_MODEL.CONSTANT):
            diff += self.diff*self.LBM.dt[None]/self.LBM.dx**2
        elif ti.static(self.diff_model==DIFF_MODEL.SCHMIDT):
            diff += self.Sc*self.viscosity(i)*self.LBM.dt[None]/self.LBM.dx**2
        return diff
    @ti.func
    def viscosity(self,i): # in LU
        visco = 1e-5
        if ti.static(self.viscosity_type==VISCOSITY_MODEL.CONSTANT):
            visco = self.visco[None]*self.LBM.dt[None]/self.LBM.dx**2
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
                capa = (((self.NASAcoef[0,4]*T+self.NASAcoef[0,3])*T+self.NASAcoef[0,2])*T+self.NASAcoef[0,1])*T+self.NASAcoef[0,0]
            else:
                capa = (((self.NASAcoef[1,4]*T+self.NASAcoef[1,3])*T+self.NASAcoef[1,2])*T+self.NASAcoef[1,1])*T+self.NASAcoef[1,0]
            capa *=constant.R
        elif ti.static(self.thermo_model==THERMO_MODEL.POLYNOMIAL):
            capa = (((self.capa_poly[4]*T+self.capa_poly[3])*T+self.capa_poly[2])*T+self.capa_poly[1])*T+self.capa_poly[0]
        return capa
    @ti.func
    def enthalpy_mole(self,T): 
        H = 0.0
        if ti.static(self.thermo_model==THERMO_MODEL.CONSTANT):
            H += self.enthalpy
        elif ti.static(self.thermo_model==THERMO_MODEL.NASA7):
            if T < self.Trange[1]:
                H = ((((self.NASAcoef[0,4]*T/5+self.NASAcoef[0,3]/4)*T+self.NASAcoef[0,2]/3)*T+self.NASAcoef[0,1]/2)*T+self.NASAcoef[0,0])*T+self.NASAcoef[0,5]        
            else:
                H = ((((self.NASAcoef[1,4]*T/5+self.NASAcoef[1,3]/4)*T+self.NASAcoef[1,2]/3)*T+self.NASAcoef[1,1]/2)*T+self.NASAcoef[1,0])*T+self.NASAcoef[1,5]
            H *=constant.R*H*T
        return H
    @ti.func
    def entropy_mole(self,T):
        S_R = 1.0
        if T < self.Trange[1]:
            S_R = (((self.NASAcoef[0,4]*T/4+self.NASAcoef[0,3]/3)*T+self.NASAcoef[0,2]/2)*T+self.NASAcoef[0,1])*T+self.NASAcoef[0,0]*ti.log(T)+self.NASAcoef[0,6]        
        else:
            S_R = (((self.NASAcoef[1,4]*T/4+self.NASAcoef[1,3]/3)*T+self.NASAcoef[1,2]/2)*T+self.NASAcoef[1,1])*T+self.NASAcoef[1,0]*ti.log(T)+self.NASAcoef[1,6]  
        return S_R*constant.R
    @ti.func
    def capacity_m(self,T):
        cm = 0.0
        if ti.static(self.capa_unit==UNIT.MOLE):
            cm = self.capacity_mole(T)/self.molemass
        elif ti.static(self.capa_unit==UNIT.MASS):
            if ti.static(self.thermo_model==THERMO_MODEL.CONSTANT):
                cm = self.capa
            elif ti.static(self.thermo_model==THERMO_MODEL.POLYNOMIAL):
                cm = (((self.capa_poly[4]*T+self.capa_poly[3])*T+self.capa_poly[2])*T+self.capa_poly[1])*T+self.capa_poly[0]
        return cm
    @ti.func
    def enthalpy_m(self,T):
        Hm = 0.0
        if ti.static(self.unit==UNIT.MOLE):
            Hm = self.enthalpy_mole(T)/self.molemass
        else:
            if self.thermo_model==THERMO_MODEL.CONSTANT:
                Hm = self.enthalpy
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
    def __init__(self,lb2d,formula,A,Ea,b = 0,Tmin = 0,deltaH = 0,name="unnamed reaction",unit = UNIT.MASS,fixDH = True):
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
        self.specieNum = len(self.LBM.species)
        self.coefProduct = [0.0] * self.specieNum
        self.coefReactant = [0.0] * self.specieNum
        self.coefRate = [0.0] * self.specieNum
        self.parse_formula(formula)
    def parse_formula(self,formula:str):
        formula = formula.replace("<=>","|").replace("=>","|")
        reactant,product = formula.split("|")
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
            self.coefRate[self.LBM.specieName.index(specie)] = coef # 默认按照化学计量数进行化学反应
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
    def description(self):
        description = f" - {self.name} : \n"
        description += f"    formula : {self.formula}\n"
        if self.isFixDH:
            description += f"    deltaH : {self.deltaH} J\n"
        else:
            description += f"    deltaH : not fixed, calculated based on species enthalpy\n"
        description += f"    kinetic model : "
        if self.reactionType == REACTION_TYPE.ARREHNIUS:
            description += f"Arrhenius : A {self.A}, b {self.b}, Ea {self.Ea} J\n"
        if self.unit==UNIT.MASS:
            description += f"    unit : mass-based\n"
        elif self.unit==UNIT.MOLE:
            description += f"    unit : mole-based\n"
        else:
            description += f"    unit : not valid\n"
        
        return description
    def __repr__(self):
        return self.__str__()
    @ti.func
    def Arrehnius(self,i):  # 温度从0开始演化的时候数值震荡可能会出现负温度，最好定义截止温度
        k = 0.0
        T = 0.0
        if ti.static(self.LBM.TEMPERATURE):
            # 有固相参与使用固相温度 
            if self.LBM.solid[i] > 0: # 有点问题，有固体不代表固相反应
                T = self.LBM.GetTS(i)
            else:
                T = self.LBM.GetTF(i)
            if (T > self.Tmin):
                k = self.A*(T+1e-6)**self.b*ti.math.exp(-self.Ea/(T+1e-6)/constant.R)
        else:
            k = self.A
        return k
    @ti.func
    def reaction(self,i):  # mole修正还没有写好
        kr = self.Arrehnius(i)
        dS = ti.Vector([0.0]*(self.specieNum+1)) # concentration change (specieNum) and enthalpy change (1)
        # 计算化学反应速率 mole 
        for j in ti.static(range(len(self.LBM.species))):
            if self.coefReactant[j] > 0.0 : # 该物质参与反应
                if self.LBM.species[j].S[i] > self.LBM.tol: # 存在该物质
                    if self.coefRate[j] != 0.0:#浓度参与化学反应速率计算
                        if ti.static(not self.LBM.species[j].FIX): # 流体反应物 该物质对反应的贡献 可以不贡献
                            if ti.static(self.unit==UNIT.MOLE):
                                kr *= (self.LBM.species[j].S[i]*self.LBM.rho[i]/self.LBM.species[j].molemass)**self.coefRate[j] # 气态反应物 该物质对反应的贡献 可以不贡献
                            elif ti.static(self.unit==UNIT.MASS):
                                kr *= (self.LBM.species[j].S[i]*self.LBM.rho[i])**self.coefRate[j] # 气态反应物 该物质对反应的贡献 可以不贡献
                        else: # 固体反应物 该物质对反应的贡献 可以不贡献
                            if ti.static(self.unit==UNIT.MOLE):
                                kr *= (self.LBM.species[j].S[i]/self.LBM.species[j].molemass)**self.coefRate[j] # 固态反应物 该物质对反应的贡献 可以不贡献
                            elif ti.static(self.unit==UNIT.MASS):
                                kr *= (self.LBM.species[j].S[i])**self.coefRate[j] # 固态反应物 该物质对反应的贡献 可以不贡献
                        # elif ti.static(self.unit==UNIT.MOLE): # mole表示的化学反应需要按照摩尔质量修正因为物种信息存储的是密度信息
                        #     kr *= (specie.S[i]/specie.molemass)**self.coefRate[j] # 该物质对反应的贡献 可以不贡献
                else: # no reaction including absence of catalyst
                    kr = 0
        # if kr>0:
        # print(kr)
        # 计算物种浓度变化和焓变
        # kr mol/m3/s
        dH = 0.0 # 物种变化带来的焓变以及反应焓变
        for j in ti.static(range(len(self.LBM.species))):
            coef = -self.coefReactant[j]+self.coefProduct[j] # 物质生成或者消耗
            if coef != 0.:
                if ti.static(self.unit==UNIT.MASS):
                    ds = kr*coef*self.LBM.dt[None] # 物种的生成和消失
                    dS[j] = ds # d\rho
                    if ti.static(not self.isFixDH and self.LBM.TEMPERATURE):
                        dH += coef*self.LBM.species[j].enthalpy_m(i)
                else:
                    ds = kr*coef*self.LBM.dt[None]*self.LBM.species[j].molemass # 摩尔质量修正到密度
                    dS[j] = ds # d\rho
                    if ti.static(not self.isFixDH and self.LBM.TEMPERATURE):
                        dH += coef*self.LBM.species[j].enthalpy_mole(i)
        if ti.static(self.LBM.TEMPERATURE):
            if ti.static(self.isFixDH):
                dH += self.deltaH # 反应热效应
                        # # 物质的生成和消失会影响焓变                    
                        # # dh += ds*self.LBM.Temperature.S[i]*specie.capacity_m(i) # 物种生成和消失带来的焓变
            # 反应热效应    
            dH *= -kr*self.LBM.dt[None] # 注意保证kr deltaH的单位匹配。是质量都是质量，是摩尔数都是摩尔数。
            dS[self.specieNum]= dH # J
        return dS
@ti.data_oriented
class Reactions:
    def __init__(self,lbm):
        self.LBM = lbm
        self.dS:ti.MatrixField # initialize during init_python (specieNum + 1)
        self.reactions:list[Reaction] = []
        self.specieNum = 0
    def description(self):
        description = "Reactions:\n"
        for r in self.reactions:
            description += "   "+r.description() + "\n"
        description += f"{len(self.reactions)} reaction(s) in total.\n"
        return description 
    def add_reaction(self,reaction):
        self.reactions.append(reaction)
    @ti.func
    def update_dS(self,i): # 计算所有化学反应带来的物质源项和能量源项
        for j in ti.static(range(self.specieNum+1)): 
            self.dS[i][j] = 0
        for r in ti.static(self.reactions): # reaction update
            self.dS[i] += r.reaction(i)
        for j in ti.static(range(self.specieNum)):
            self.LBM.species[j].dS[i] = self.dS[i][j]
        if ti.static(self.LBM.TEMPERATURE):
            if self.LBM.solid[i] > 0:
                self.LBM.TS.dS[i] += self.dS[i][self.specieNum]/self.LBM.TS.capacity_m(i)/self.LBM.rhos[i]/self.LBM.TS.v_scale # 化学反应带来的能量变化转换为温度变化 注意温度变化量归一化
            else:
                self.LBM.TF.dS[i] += self.dS[i][self.specieNum]/self.LBM.TF.capacity_m(i)/self.LBM.rho[i]/self.LBM.TF.v_scale # 化学反应带来的能量变化转换为温度变化 注意温度变化量归一化