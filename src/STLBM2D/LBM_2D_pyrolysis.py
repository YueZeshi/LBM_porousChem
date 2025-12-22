#==========
# 描述
## 按照porousGasificationFoam中的控制方程编写的热解LBM 2D仿真程序
## 气体温度和固体温度视为不同但是只考虑流体的温度输运
## 标量场边界条件支持质量分数
## 按照WANG的论文实现标量场的修正，修正平衡函数和额外源项
## 添加辐射传热 类似于光照烘培 如何烘培光照
## 详见论文
#==========


import os
import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK
from enum import Enum
import pickle
import json
class SPECIE_UNIT(Enum):
    MASS = 1
    MOLE = 2
class REACTION_TYPE(Enum):
    ARREHNIUS = 0
    LMH = 1
class RADIATION_MODEL(Enum):
    NONE = 0
    SURFACE_UNIFORM = 1
    REAL_RADIATION = 2
class PORO_MODEL(Enum):
    SPHERICAL = 0
    DARCY = 1
    DARCYFORCHHEIMER=2
class SOURCE_TERM(Enum):
    NONE = 0
    MICRO = 1 # 修改分布函数
    MACRO = 2 # 修改宏观量
class FORCE_TERM(Enum):
    NONE = 0
    GUO = 1 # 外力项
    SHAN = 2 # 改变平衡速度
    MACRO = 3 # 处理外力项太大的刚性问题
class BC(Enum):
    PERIODIC = 0
    OPEN = 1
    SYMMETRIC = 2
    FIXVALUE = 3
class BC_V(Enum):
    PERIODIC = 0
    OPEN = 1
    SYMMETRIC = 2
    FIXVALUE = 3
    NOSLIP = 4
    FIXPRESSURE = 5
class BC_S(Enum):
    PERIODIC = 0
    OPEN = 1
    SYMMETRIC = 2
    FIXVALUE = 3
    FIXFLUX = 5
    WALL = 4
    FIXPERCENTAGE = 6

@ti.data_oriented    
class ScalarField:
    '''
    标量场 D2Q5
    '''
    def __init__(self,name,nx,ny,nz,lb2d,FIX = False):
        self.name = name
        self.nx,self.ny,self.nz = nx,ny,nz
        self.LBM = lb2d
        self.FIX = FIX
        self.S = ti.field(ti.f32,shape=(nx,ny,nz))
        if not self.FIX:
            self.g = ti.Vector.field(5,ti.f32,shape=(nx,ny,nz))
            self.G = ti.Vector.field(5,ti.f32,shape=(nx,ny,nz))
            self.BC = [BC.PERIODIC,BC.PERIODIC,BC.PERIODIC,BC.PERIODIC]
            self.flux_BC =ti.field(ti.f32,shape = (4))
            self.s_BC = ti.field(ti.f32,shape = (4))
            for i in range(4):
                self.flux_BC[i]=0.0
                self.s_BC[i]=0.0
    @ti.func
    def coefDiff(self,i):
        return 0.1
    # boundary condition
    def set_BC(self,index_boundary,BC):
        self.BC[index_boundary]=BC
    def set_BCs(self,BCs):
        self.BC = BCs
    def set_s_BC_value(self,index,s):
        self.s_BC[index]=s
    def set_s_BCs_value(self,s):
        for i in range(4):
            self.s_BC[i]=s[i]
    def set_s_BC_flux(self,index,f):
        self.flux_BC[index]=f
    def set_s_BCs_flux(self,f):
        for i in range(4):
            self.flux_BC[i]=f[i]
    @ti.func
    def geq5(self,k,S,x,y,z):
        return 0
    @ti.func
    def Boundary_condition_scalar_0(self,x,y,z):
        if ti.static(self.BC[0]==BC_S.WALL):# no slip
            self.S[0,y,z] = self.S[1,y,z]  
        if ti.static(self.BC[0]==BC_S.FIXVALUE): # fix v
            self.S[0,y,z] = self.s_BC[0]
        if ti.static(self.BC[0]==BC_S.OPEN): # open
            self.S[0,y,z] = self.S[1,y,z]  
        # todo
        for s in ti.static(range(5)):
            self.G[0,y,z][s] = self.geq5(s,self.S[0,y,z],0,y,z)+(self.G[1,y,z][s]-self.geq5(s,self.S[1,y,z],1,y,z))
    @ti.func
    def Boundary_condition_scalar_1(self,x,y,z):
        if ti.static(self.BC[1]==BC_S.WALL):# no slip
            self.S[self.nx-1,y,z] = self.S[self.nx-2,y,z]  
        if ti.static(self.BC[1]==BC_S.FIXVALUE): # fix v
            self.S[self.nx-1,y,z] = self.s_BC[1]
        if ti.static(self.BC[1]==BC_S.OPEN): # open
            self.S[self.nx-1,y,z] = self.S[self.nx-2,y,z]  
        # todo
        for s in ti.static(range(5)):
            self.G[self.nx-1,y,z][s] = self.geq5(s,self.S[self.nx-1,y,z],self.nx-1,y,z)+(self.G[self.nx-2,y,z][s]-self.geq5(s,self.S[self.nx-2,y,z],self.nx-2,y,z))
    @ti.func
    def Boundary_condition_scalar_2(self,x,y,z):
        if ti.static(self.BC[2]==BC_S.WALL):# no slip
            self.S[x,0,z] = self.S[x,1,z]  
        if ti.static(self.BC[2]==BC_S.FIXVALUE): # fix v
            self.S[x,0,z] = self.s_BC[2]
        if ti.static(self.BC[2]==BC_S.OPEN): # open
            self.S[x,0,z] = self.S[x,1,z]  
        # todo
        for s in ti.static(range(5)):
            self.G[x,0,z][s] = self.geq5(s,self.S[x,0,z],x,0,z)+(self.G[x,1,z][s]-self.geq5(s,self.S[x,1,z],x,1,z))
    @ti.func
    def Boundary_condition_scalar_3(self,x,y,z):
        if ti.static(self.BC[3]==BC_S.WALL):# no slip
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        if ti.static(self.BC[3]==BC_S.FIXVALUE): # fix v
            self.S[x,self.ny-1,z] = self.s_BC[3]
        if ti.static(self.BC[3]==BC_S.OPEN): # open
            self.S[x,self.ny-1,z] = self.S[x,self.ny-2,z]  
        # todo
        for s in ti.static(range(5)):
            self.G[x,self.ny-1,z][s] = self.geq5(s,self.S[x,self.ny-1,z],x,self.ny-1,z)+(self.G[x,self.ny-2,z][s]-self.geq5(s,self.S[x,self.ny-2,z],x,self.ny-2,z))
        
        
@ti.data_oriented
class Specie(ScalarField): # 物种质量分数场
    def __init__(self,name,nx,ny,nz,lb2d,FIX = False,Mmass = 1.0):
        super().__init__(name,nx,ny,nz,lb2d,FIX)
        if not self.FIX:
            self.Y = ti.field(ti.f32,shape=(nx,ny,1))
            self.Y_BC = ti.field(ti.f32,shape = (4))
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
class TemperatureFluid(ScalarField):
    def __init__(self,name,nx,ny,nz,lb2d,FIX = False):
        super().__init__(name,nx,ny,nz,lb2d,FIX)
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
    def geq5(self,k,T,x,y,z):
        u = self.LBM.v[x,y,z]
        eu = self.LBM.e5[k].dot(u)
        uv = u.dot(u)
        geqout=0.0
        geqout += self.LBM.w5[k]*T*(1.0 + (3.0*eu+4.5*eu**2-1.5*uv))
        return geqout

@ti.data_oriented
class TemperatureSolid(ScalarField):
    def __init__(self,name,nx,ny,nz,lb2d,FIX = False):
        super().__init__(name,nx,ny,nz,lb2d,FIX)
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
    def geq5(self,k,T,x,y,z):
        geqout=0.0
        geqout += self.LBM.w5[k]*T
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
    def reaction(self):  # mole修正还没有写好
        for i in ti.grouped(self.LBM.rho):
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
                        for k in ti.static(range(9)):
                            self.LBM.f[i][k] += self.LBM.feq9(k,ds,self.LBM.v[i]) # 更新密度分布函数
                            if ti.static(k<5):
                                specie.g[i][k] += specie.geq5(k,ds/self.LBM.rho[i],i[0],i[1],i[2]) # 更新物种质量分数分布函数，用气体密度用当前密度近似
                            # self.LBM.Temperature.g[i][k]+=self.LBM.w5[k]*ds*self.LBM.Temperature.S[i]*specie.capacity_m(i) # 流体的焓随麦克斯韦分布展开
                    # 物质的生成和消失会影响焓变                    
                    # dh += ds*self.LBM.Temperature.S[i]*specie.capacity_m(i) # 物种生成和消失带来的焓变
                j+=1
            # 反应热效应
            if ti.static(self.LBM.TEMPERATURE):
                dh += -kr*self.deltaH*self.LBM.dt # 注意保证kr deltaH的单位匹配。是质量都是质量，是摩尔数都是摩尔数。
                """
                热源项疑似会带来问题，热源项先还原到温度再按照平衡态分配进去。如果直接放在格点中心会导致热量在格点中心堆积，出现温度奇点
                """    
                for k in ti.static(range(5)):
                    self.LBM.TS.g[i][k] += self.LBM.TS.geq5(k,dh/self.LBM.TS.capacity_v(i),i[0],i[1],i[2])
                
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

@ti.data_oriented
class LB2D_PYRO:
    SIGMA = 5.67e-8
    def __init__(self, nx, ny,nz = 1,dx = 0.001,dt = 0.001,isThermal = False,isChemical = False,isPoro = False,isRadiation = False):
        self.t = 0.0
        # 模型参数
        self.enable_projection = True
        self.nx,self.ny,self.nz= nx,ny,nz # 模型大小
        self.dx,self.dt = dx,dt #格子尺度 步进时间
        self.max_v=ti.field(ti.f32,shape=())
        self.source_term_model = SOURCE_TERM.NONE
        self.force_term_model = FORCE_TERM.NONE
        self.poro_model = PORO_MODEL.SPHERICAL # 使用的多孔介质模型 如球孔介质模型 Darcy Darcy-Forhheimer
        self.radiation_model = RADIATION_MODEL.NONE # 辐射模型
        self.coefDarcy,self.coefForchheimer = 0.0,0.0
        #Boundary condition mode: 0= periodic, 1= no slip, 2= fix velocity, 3= open; boundary pressure value (rho); boundary velocity value for vx,vy
        # LBM使用常量
        self.e = ti.Vector.field(3,ti.i32, shape=(9)) # e 方向向量
        self.w = ti.field(ti.f32, shape=(9)) # 权重
        self.LR = [0,3,4,1,2,7,8,5,6] #对称索引
        self.e5 = ti.Vector.field(3,ti.i32, shape=(5)) # e 方向向量
        self.w5 = ti.field(ti.f32, shape=(5)) # 权重
        self.normal = ti.Vector.field(3,dtype=ti.i32,shape = (4))
        self.large_step = 1
        # self.LR5 = [0,3,4,1,2] #对称索引
        self.x = np.linspace(0, nx, nx)
        self.y = np.linspace(0, ny, ny)
        self.z = np.linspace(0, nz, nz)
        self.ext_f = ti.Vector.field(3,ti.f32,shape=()) # 外部力
        
        #X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # 声明物理场
        self.rho = ti.field(ti.f32, shape=(nx,ny,nz))
        self.v = ti.Vector.field(3,ti.f32, shape=(nx,ny,nz))
        self.solid = ti.field(ti.f32,shape = (nx,ny,nz))
        self.f = ti.Vector.field(9,ti.f32,shape=(nx,ny,nz))
        self.F = ti.Vector.field(9,ti.f32,shape=(nx,ny,nz))

        # 定义边界条件
        self.BC = [BC.PERIODIC,BC.PERIODIC,BC.PERIODIC,BC.PERIODIC]
        self.v_BC = ti.Vector.field(3,ti.f32,shape = (4))
        self.rho_BC = ti.field(ti.f32,shape = (4))
        self.UpdateBCfunc = [] # 可变边界条件
        self.GetVariableFunc = [] # 获取观测量的函数
        
        # 定义标量场
        self.TEMPERATURE = isThermal
        self.CHEMISTRY = isChemical
        self.PORO = isPoro
        self.RADIATION = isRadiation and isThermal
        if self.TEMPERATURE:
            self.TF = TemperatureFluid("Temperature of Fluid",nx,ny,nz,self)
            self.TS = TemperatureSolid("Temperature of Solid",nx,ny,nz,self)
            self.heat_trasnfer_surface = ti.field(ti.f32,shape = (nx,ny,nz))
            self.min_T = ti.field(ti.f32,shape=())
            if self.RADIATION:
                # self.radiation = ti.field(ti.f32,shape=(nx,ny,nz))
                self.radiation_surface = ti.field(ti.f32,shape = (nx,ny,nz)) # S/V L-1
        if self.CHEMISTRY:
            self.species = {}
            self.reactions = dict()
        if self.PORO:
            self.rhos = ti.field(ti.f32,shape=(nx,ny,1))
            self.rho1 = ti.field(ti.f32,shape=(nx,ny,1))
        # 默认初始化所有场
        self.default_init()
    # 内置函数
    def __repr__(self):
        return self.__str__()
    def __call__(self):
        print(self)
    def __str__(self):
        return "多孔生物质热解LBM2D求解器。包含多孔介质达西定律，温度场，温度流场耦合单松弛因子LBM，可根据Darcy定律模拟多孔介质行为。可定义不同区域物性"
    # 初始化
    @ti.kernel
    def default_init(self): # 创建时初始化
        for i in ti.ndrange(4):
            self.rho_BC[i] = 0.0
            self.v_BC[i] = ti.Vector([0,0,0])
        for i in ti.grouped(self.solid):
            self.solid[i] = 0.0
            self.v[i] = ti.Vector([0,0,0])
            self.rho[i] = 1.0
    @ti.kernel
    def init_simulation(self):# 用户设置完数值之后进行手动初始化
        # self.init_python()
        self.static_init_kernel()
        # self.check()
        self.init_kernel()
        # self.check()
        self.macro()
        # self.check()
        self.print_information()
    


    @ti.func
    def static_init_kernel(self): # 初始化静态变量
        if ti.static(self.enable_projection): # No runtime overhead
            self.e[0] = ti.Vector([0,0,0])
            self.e[1] = ti.Vector([1,0,0]); self.e[2] = ti.Vector([0,1,0]); self.e[3] = ti.Vector([-1,0,0]); self.e[4] = ti.Vector([0,-1,0])
            self.e[5] = ti.Vector([1,1,0]); self.e[6] = ti.Vector([-1,1,0]); self.e[7] = ti.Vector([-1,-1,0]); self.e[8] = ti.Vector([1,-1,0])
            self.w[0] = 4.0/9.0
            self.w[1] = 1.0/9.0; self.w[2] = 1.0/9.0; self.w[3] = 1.0/9.0; self.w[4] = 1.0/9.0
            self.w[5] = 1.0/36.0; self.w[6] = 1.0/36.0; self.w[7] = 1.0/36.0; self.w[8] = 1.0/36.0
            self.e5[0] = ti.Vector([0,0,0])
            self.e5[1] = ti.Vector([1,0,0]); self.e5[2] = ti.Vector([0,1,0]); self.e5[3] = ti.Vector([-1,0,0]); self.e5[4] = ti.Vector([0,-1,0])
            self.w5[0] = 1.0/3.0
            self.w5[1] = 1.0/6.0; self.w5[2] = 1.0/6.0; self.w5[3] = 1.0/6.0; self.w5[4] = 1.0/6.0
            
    @ti.func
    def init_kernel(self): # 初始化所有分布函数
        for i in ti.grouped(self.solid):
            eps = 1-self.solid[i]
            if ti.static(self.PORO):
                if ti.static(self.CHEMISTRY): # 计算固体物质密度
                    self.rhos[i]=0.0
                    for specie in ti.static(list(self.species.values())):
                        if ti.static(specie.FIX):
                            self.rhos[i] += specie.S[i]
                if self.solid[i]!=0: # 计算孔隙率为1的密度作为参考
                    self.rho1[i] = self.rhos[i]/self.solid[i]
                    if self.rho1[i]==0:
                        self.rho1[i]=1.0
            for s in ti.static(range(9)):
                self.f[i][s] = self.feq9(s,self.rho[i],self.v[i],eps)
                self.F[i][s] = self.f[i][s]
            # if ti.static(self.TEMPERATURE):
            #     self.IE.uS[i] = self.IE.S[i]*self.v[i]
            for s in ti.static(range(5)):
                if ti.static(self.CHEMISTRY):
                    for specie in ti.static(list(self.species.values())):
                        if ti.static(not specie.FIX):
                            # specie.init()
                            specie.g[i][s] = specie.geq5(s,specie.S[i],i[0],i[1],i[2])
                            specie.G[i][s] = specie.g[i][s]
                if ti.static(self.TEMPERATURE):
                # self.IE.init()
                    self.TF.g[i][s] = self.TF.geq5(s,self.TF.S[i],i[0],i[1],i[2])
                    self.TF.G[i][s] = self.TF.g[i][s]
                    self.TS.g[i][s] = self.TS.geq5(s,self.TS.S[i],i[0],i[1],i[2])
                    self.TS.G[i][s] = self.TS.g[i][s]
        
    # 步进
    @ti.kernel
    def step(self):
        for i in ti.static(range(self.large_step)):
            # self.check()
            self.collision() # F->f
            # self.check()
            self.source_term_micro() # f 
            # self.check()
            self.streaming() #f->F
            # self.check()
            self.macro()  # F->value
            self.source_term_macro() # value 
            # self.check()
            self.Boundary_condition() # value changed for the boundary condition
            # self.check()
            # print("")
            # self.updateBC(self.t)
            # self.t=self.t+self.dt
        # self.check()
    @ti.func
    def print_information(self):
        print(f"The {self.nx}*{self.ny}*{self.nz} LBM has been successfully initialized.")
        print("It contains :")
        if self.PORO:
            print(" -porous medium")
        if self.TEMPERATURE:
            print(" -thermal transfer")
        if self.RADIATION:
            print(" -radiation")
        if ti.static(self.CHEMISTRY):
            print(" -chemical reaction")
            print("The species concerned: ",end="")
            for specie in ti.static(list(self.species.keys())):
                print(specie,end=" ")
            print("")
    @ti.func
    def collision(self):# F->f H->h G->g reaction
        for i in ti.grouped(self.rho):
            if (self.solid[i] < 1):
                for k in ti.static(range(9)):
                    eps = 1-self.solid[i]
                    self.f[i][k] =self.F[i][k]-1/(3*self.viscosity(i)+0.5)*(self.F[i][k]-self.feq9(k,self.rho[i],self.v[i],eps))
                    if ti.static(k<5):
                        if ti.static(self.CHEMISTRY):
                            for specie in ti.static(list(self.species.values())):
                                if ti.static(not specie.FIX):
                                    specie.g[i][k] =specie.G[i][k]-1/(3*specie.coefDiff(i)+0.5)*(specie.G[i][k]-specie.geq5(k,specie.S[i],i[0],i[1],i[2]))
                        if ti.static(self.TEMPERATURE):
                            self.TF.g[i][k] = self.TF.G[i][k]-1/(3*self.TF.coefDiff(i)+0.5)*(self.TF.G[i][k]-self.TF.geq5(k,self.TF.S[i],i[0],i[1],i[2]))
                            if eps>0.0:
                                self.TS.g[i][k] = self.TS.G[i][k]-1/(3*self.TS.coefDiff(i)+0.5)*(self.TS.G[i][k]-self.TS.geq5(k,self.TS.S[i],i[0],i[1],i[2]))
                                

    @ti.func
    def source_term_micro(self): # f           
            # body force
        if ti.static(self.force_term_model==FORCE_TERM.GUO):
            if ti.static(self.PORO):
                for i in ti.grouped(self.rho):
                    if (self.solid[i] < 1 and self.solid[i]>0):
                        for k in ti.ndrange(5):
                            self.f[i][k] +=self.forceTermGuo(k,self.rho[i],self.v[i],self.force(i),3*self.viscosity(i)+0.5) # 如果力太大，刚性太强->需要使用宏观指数衰减
        
        
        if ti.static(self.source_term_model==SOURCE_TERM.MICRO):
            for i in ti.grouped(self.rho):
                if self.solid[i]>1e-6: # 有固体的区域
                    if ti.static(self.TEMPERATURE):
                        q = 100.0*self.heat_trasnfer_surface[i]*(self.TF.S[i]-self.TS.S[i]) # from f to s
                        for k in ti.ndrange(5):
                            self.TS.g[i][k] += self.TS.geq5(k,q/self.TS.capacity_v(i),i[0],i[1],i[2])
                            self.TF.g[i][k] += self.TF.geq5(k,-q/self.TF.capacity_v(i),i[0],i[1],i[2])
                        
            
                if ti.static(self.RADIATION): # radiation
                    dT =self.dt*self.radiation(i)/self.TS.capacity_v(i) # 计算温度变化 # s*{rad}/(Jm-3K-1)=K; {rad}=Wm-3
                    for k in ti.ndrange(5):
                        self.TS.g[i][k] += self.TS.geq5(k,dT,i[0],i[1],i[2]) # 辐射吸热 # 直接添加到格子中央可能会带来问题
            if ti.static(self.CHEMISTRY):
                for r in ti.static(list(self.reactions.values())):
                    r.reaction() # update species and internal energy

    @ti.func
    def streaming(self): # f->F h->H
        for i in ti.grouped(self.rho):
            if (self.solid[i]<1):# 流体更新
                for s in ti.static(range(9)):
                    ip = self.periodic_index(i+self.e[s]) # 更新之后的位置索引，默认边界视为周期边界
                    if (self.solid[ip]!=1): # 如果更新之后的位置是液体    
                        self.F[ip][s] = self.f[i][s] # 直接更新F
                        if s<5:
                            if ti.static(self.TEMPERATURE):
                                self.TF.G[ip][s] = self.TF.g[i][s]
                            if ti.static(self.CHEMISTRY):
                                for specie in ti.static(list(self.species.values())):
                                    if ti.static(not specie.FIX):
                                        specie.G[ip][s] = specie.g[i][s]
                    else:
                        self.F[i][self.LR[s]] = self.f[i][s] # 如果不是 反弹 流固表面处理
                        if ti.static(s<5):
                            if ti.static(self.TEMPERATURE):
                                self.TF.G[i][self.LR[s]] = self.TF.g[i][s]
                            if ti.static(self.CHEMISTRY):
                                for specie in ti.static(list(self.species.values())):
                                    if ti.static(not specie.FIX):
                                        specie.G[i][self.LR[s]] = specie.g[i][s]
            if (self.solid[i] > 0):# 固体更新
                for s in ti.static(range(5)):
                    ip = self.periodic_index(i+self.e[s]) # 更新之后的位置索引，默认边界视为周期边界
                    if (self.solid[ip] >0): # 如果更新之后的位置有固体  
                        if ti.static(self.TEMPERATURE):
                            self.TS.G[ip][s] = self.TS.g[i][s]
                    else: # 更新之后无固体 迁移步骤中认为是绝热壁 热交换在源项中
                        if ti.static(self.TEMPERATURE):
                            self.TS.G[i][self.LR[s]] = self.TS.g[i][s]
    @ti.func
    def Boundary_condition(self):
        for j in ti.ndrange((0,self.ny)):
            self.Boundary_condition_flow_0(0,j,0)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_0(0,j,0)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_0(0,j,0) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_0(0,j,0)
            self.Boundary_condition_flow_1(self.nx-1,j,0)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_1(self.nx-1,j,0)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_1(self.nx-1,j,0) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_1(self.nx-1,j,0)
        for i in ti.ndrange((0,self.nx)):

            self.Boundary_condition_flow_2(i,0,0)
            if ti.static(self.CHEMISTRY):
               for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_2(i,0,0)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_2(i,0,0) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_2(i,0,0)
            self.Boundary_condition_flow_3(i,self.ny-1,0)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species.values())):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_3(i,self.ny-1,0)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_3(i,self.ny-1,0) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_3(i,self.ny-1,0)
    @ti.func
    def Boundary_condition_flow_0(self,x,y,z):
        if ti.static(self.BC[0]==BC_V.NOSLIP):
            self.v[0,y,z] = ti.Vector([0,0,0])
            self.rho[0,y,z] = self.rho[1,y,z]        
        elif ti.static(self.BC[0]==BC_V.FIXVALUE):
            self.v[0,y,z] = self.v_BC[0]
            self.rho[0,y,z] = self.rho[1,y,z]
        elif ti.static(self.BC[0]==BC_V.OPEN):
            self.v[0,y,z] = self.v[1,y,z]
            self.rho[0,y,z] = self.rho[1,y,z]
        elif ti.static(self.BC[0]==BC_V.FIXPRESSURE): # fix pressure
            self.v[0,y,z] = self.v[1,y,z]
            self.rho[0,y,z] = self.rho_BC[0]
        for s in ti.static(range(9)):
            self.F[0,y,z][s] = self.feq9(s,self.rho[0,y,z],self.v[0,y,z])+(self.F[1,y,z][s]-self.feq9(s,self.rho[1,y,z],self.v[1,y,z]))
    @ti.func
    def Boundary_condition_flow_1(self,x,y,z):
        if ti.static(self.BC[1]==BC_V.NOSLIP):
            self.v[self.nx-1,y,z] = ti.Vector([0,0,0])
            self.rho[self.nx-1,y,z] = self.rho[self.nx-2,y,z]        
        elif ti.static(self.BC[1]==BC_V.FIXVALUE):
            self.v[self.nx-1,y,z] = self.v_BC[1]
            self.rho[self.nx-1,y,z] = self.rho[self.nx-2,y,z]
        elif ti.static(self.BC[1]==BC_V.OPEN):
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
            self.rho[self.nx-1,y,z] = self.rho[self.nx-2,y,z]
        elif ti.static(self.BC[1]==BC_V.FIXPRESSURE): # fix pressure
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
            self.rho[self.nx-1,y,z] = self.rho_BC[1]
        for s in ti.static(range(9)):
            self.F[self.nx-1,y,z][s] = self.feq9(s,self.rho[self.nx-1,y,z],self.v[self.nx-1,y,z])+(self.F[self.nx-2,y,z][s]-self.feq9(s,self.rho[self.nx-2,y,z],self.v[self.nx-2,y,z]))
    @ti.func
    def Boundary_condition_flow_2(self,x,y,z):
        if ti.static(self.BC[2]==BC_V.NOSLIP):
            self.v[x,0,z] = ti.Vector([0,0,0])
            self.rho[x,0,z] = self.rho[x,1,z]        
        elif ti.static(self.BC[2]==BC_V.FIXVALUE):
            self.v[x,0,z] = self.v_BC[2]
            self.rho[x,0,z] = self.rho[x,1,z]
        elif ti.static(self.BC[2]==BC_V.OPEN):
            self.v[x,0,z] = self.v[x,1,z]
            self.rho[x,0,z] = self.rho[x,1,z]
        elif ti.static(self.BC[2]==BC_V.FIXPRESSURE): # fix pressure
            self.v[x,0,z] = self.v[x,1,z]
            self.rho[x,0,z] = self.rho_BC[2]
        for s in ti.static(range(9)):
            self.F[x,0,z][s] = self.feq9(s,self.rho[x,0,z],self.v[x,0,z])+(self.F[x,1,z][s]-self.feq9(s,self.rho[x,1,z],self.v[x,1,z]))
    @ti.func
    def Boundary_condition_flow_3(self,x,y,z):
        if ti.static(self.BC[3]==BC_V.NOSLIP):
            self.v[x,self.ny-1,z] = ti.Vector([0,0,0])
            self.rho[x,self.ny-1,z] = self.rho[x,self.ny-2,z]        
        elif ti.static(self.BC[3]==BC_V.FIXVALUE):
            self.v[x,self.ny-1,z] = self.v_BC[3]
            self.rho[x,self.ny-1,z] = self.rho[x,self.ny-2,z]
        elif ti.static(self.BC[3]==BC_V.OPEN):
            self.v[x,self.ny-1,z] = self.v[x,self.ny-2,z]
            self.rho[x,self.ny-1,z] = self.rho[x,self.ny-2,z]
        elif ti.static(self.BC[3]==BC_V.FIXPRESSURE): # fix pressure
            self.v[x,self.ny-1,z] = self.v[x,self.ny-2,z]
            self.rho[x,self.ny-1,z] = self.rho_BC[3]
        for s in ti.static(range(9)):
            self.F[x,self.ny-1,z][s] = self.feq9(s,self.rho[x,self.ny-1,z],self.v[x,self.ny-1,z])+(self.F[x,self.ny-2,z][s]-self.feq9(s,self.rho[x,self.ny-2,z],self.v[x,self.ny-2,z]))
    
    # @ti.func
    # def Boundary_condition_extrapolation(self): 
    #     # 非平衡态外推格式 
    #     # 使用此外推格式的时候最好不可以使用默认的周期边界条件，因为边界条件在宏观量之后恢复，会导致在另一个边界出现虚假流。
    #     # 修正方法是强行在此边界条件中重新定义边界格点宏观物理量，以防止错误的宏观物理量传递到下一次碰撞中。
    #     ti.loop_config(parallelize=8,block_dim=64)
    #     for j in ti.ndrange((0,self.ny)):
    #         # x left
    #         if ti.static(self.BC[0]==BC_V.NOSLIP):# no slip
    #             self.v[0,j,0] = ti.Vector([0,0,0])
    #             self.rho[0,j,0] = self.rho[1,j,0]    
    #         if ti.static(self.BC[0]==BC_V.FIXVALUE): # fix v
    #             self.v[0,j,0] = self.v_BC[0]
    #             self.rho[0,j,0] = self.rho[1,j,0]
    #         if ti.static(self.BC[0]==BC_V.OPEN): # open
    #             self.v[0,j,0] = self.v[1,j,0]
    #             self.rho[0,j,0] = self.rho[1,j,0]
    #         if ti.static(self.BC[0]==BC_V.FIXPRESSURE): # fix pressure
    #             self.v[0,j,0] = self.v[1,j,0]
    #             self.rho[0,j,0] = self.rho_BC[0]
            
    #         # x right
    #         if ti.static(self.BC[1]==BC_V.NOSLIP):
    #             self.v[self.nx-1,j,0] = ti.Vector([0,0,0])
    #             self.rho[self.nx-1,j,0] = self.rho[self.nx-2,j,0]        
    #         if ti.static(self.BC[1]==BC_V.FIXVALUE):
    #             self.v[self.nx-1,j,0] = self.v_BC[1]
    #             self.rho[self.nx-1,j,0] = self.rho[self.nx-2,j,0]
    #         if ti.static(self.BC[1]==BC_V.OPEN):
    #             self.v[self.nx-1,j,0] = self.v[self.nx-2,j,0]
    #             self.rho[self.nx-1,j,0] = self.rho[self.nx-2,j,0]

    #         if ti.static(self.BC[1]==BC_V.FIXPRESSURE): # fix pressure
    #             self.v[self.nx-1,j,0] = self.v[self.nx-2,j,0]
    #             self.rho[self.nx-1,j,0] = self.rho_BC[1]

    #         for s in ti.static(range(9)):
    #             self.F[self.nx-1,j,0][s] = self.feq9(s,self.rho[self.nx-1,j,0],self.v[self.nx-1,j,0])+(self.F[self.nx-2,j,0][s]-self.feq9(s,self.rho[self.nx-2,j,0],self.v[self.nx-2,j,0]))
    #             self.F[0,j,0][s] = self.feq9(s,self.rho[0,j,0],self.v[0,j,0])+(self.F[1,j,0][s]-self.feq9(s,self.rho[1,j,0],self.v[1,j,0]))
    #     ti.loop_config(parallelize=8,block_dim=64)
    #     for i in ti.ndrange((0,self.nx)):
    #         # y left
    #         if ti.static(self.BC[2]==BC_V.NOSLIP):
    #             self.v[i,0,0]=ti.Vector([0,0,0])
    #             self.rho[i,0,0] = self.rho[i,1,0]
    #         if ti.static(self.BC[2]==BC_V.FIXVALUE):
    #             self.v[i,0,0]=self.v_BC[2]
    #             self.rho[i,0,0] = self.rho[i,1,0]
    #         if ti.static(self.BC[2]==BC_V.OPEN):
    #             self.v[i,0,0] = self.v[i,1,0]
    #             self.rho[i,0,0] = self.rho[i,1,0]
    #         if ti.static(self.BC[2]==BC_V.FIXPRESSURE):
    #             self.v[i,0,0] = self.v[i,1,0]
    #             self.rho[i,0,0] = self.rho_BC[2]
    #         # y right
    #         if ti.static(self.BC[3]==BC_V.NOSLIP):
    #             self.v[i,self.ny-1,0] = ti.Vector([0,0,0])
    #             self.rho[i,self.ny-1,0] = self.rho[i,self.ny-2,0]
    #         if ti.static(self.BC[3]==BC_V.FIXVALUE):
    #             self.v[i,self.ny-1,0] = self.v_BC[3]
    #             self.rho[i,self.ny-1,0] = self.rho[i,self.ny-2,0]
    #         if ti.static(self.BC[3]==BC_V.OPEN):
    #             self.v[i,self.ny-1,0] = self.v[i,self.ny-2,0]
    #             self.rho[i,self.ny-1,0] = self.rho[i,self.ny-2,0]
    #         if ti.static(self.BC[3]==BC_V.FIXPRESSURE):
    #             self.v[i,self.ny-1,0] = self.v[i,self.ny-2,0]
    #             self.rho[i,self.ny-1,0] = self.rho_BC[3]
    #         for s in ti.static(range(9)):
    #             self.F[i,0,0][s] = self.feq9(s,self.rho[i,0,0],self.v[i,0,0])+(self.F[i,1,0][s]-self.feq9(s,self.rho[i,1,0],self.v[i,1,0]))
    #             self.F[i,self.ny-1,0][s] = self.feq9(s,self.rho[i,self.ny-1,0],self.v[i,self.ny-1,0])+(self.F[i,self.ny-2,0][s]-self.feq9(s,self.rho[i,self.ny-2,0],self.v[i,self.ny-2,0]))
    #         # 其他标量场的边界条件   
        
    #     if ti.static(self.CHEMISTRY):
    #         for specie in ti.static(list(self.species.values())):
    #             if ti.static(not specie.FIX):
    #                 specie.Boundary_condition_extrapolation()
    #     if ti.static(self.TEMPERATURE):
    #         # print("boundary IE")
    #         self.TF.Boundary_condition_extrapolation() # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
    #         self.TS.Boundary_condition_extrapolation()
    def updateBC(self,t):
        for func in self.UpdateBCfunc:
            func(self,t)

    @ti.func
    def macro(self):# F->f 计算宏观量
        for i in ti.grouped(self.rho):
            if self.solid[i]<1: # 流体/多孔介质区域
                eps = 1-self.solid[i]
                self.v[i] = ti.Vector([0,0,0])
                self.rho[i] = self.F[i].sum() # 宏观量重建 计算密度
                for s in ti.static(range(9)):
                    self.v[i] += self.e[s]*self.F[i][s]
                self.v[i] /= self.rho[i]
                # if self.v[i].norm()>0.5:
                #     print("warning: high velocity",self.v[i],self.t)
                if ti.static(self.PORO):
                    if eps!=0 and eps!=1: # 涉及多孔介质的速度修正
                        Dc,Fc = 0.0,0.0
                        if ti.static(self.poro_model==PORO_MODEL.SPHERICAL):
                            Dc = 1.0/self.perm(eps)
                            Fc = 1.75/ti.sqrt(150*ti.pow(eps,3))
                        elif ti.static(self.poro_model==PORO_MODEL.DARCY):
                            Dc = self.coefDarcy
                        elif ti.static(self.poro_model==PORO_MODEL.DARCYFORCHHEIMER):
                            Dc = self.coefDarcy
                            Fc = self.coefForchheimer
                        c0 = 0.5*(1+eps*self.viscosity(i)/2*Dc)
                        c1 = 0.5*eps*Fc
                        v = ti.math.length(self.v[i])
                        self.v[i]/=(c0+ti.sqrt(c0**2+c1*v))
                if ti.static(self.CHEMISTRY): 
                    if ti.static(self.PORO):
                        self.rhos[i]=0.0
                    Yall = 0.0
                    for specie in ti.static(list(self.species.values())):
                        if ti.static(not specie.FIX): # 更新流体组分
                            specie.S[i] = specie.G[i].sum()
                            Yall += specie.S[i]
                        else:
                            if ti.static(self.PORO): # 计算当前固体物质总密度
                                self.rhos[i]+=specie.S[i]
                    for specie in ti.static(list(self.species.values())):
                        if ti.static(not specie.FIX): # 更新流体组分
                            specie.S[i] /= Yall
                    
                if ti.static(self.PORO):
                    if self.rho1[i]!=0 and self.rhos[i] != 0:
                        self.solid[i] = self.rhos[i]/self.rho1[i] # 更新孔隙结构
                
                if ti.static(self.TEMPERATURE):
                    self.TF.S[i] = self.TF.G[i].sum() # 计算温度
                    self.TS.S[i] = self.TS.G[i].sum()   
    @ti.func
    def source_term_macro(self):
        if ti.static(self.force_term_model==FORCE_TERM.MACRO):
            for i in ti.grouped(self.rho):
                if ti.static(self.PORO): 
                    # v  : 直接隐式求解速度 当Darcy阻力过大时，减速太快。显式更新破坏稳定性。最好使用隐式更新。
                    if (self.solid[i] < 1 and self.solid[i]>0): # 只有Darcy力 多孔介质区域
                        if self.poro_model == PORO_MODEL.DARCY:
                            self.v[i] *= ti.math.exp(-self.viscosity(i)*self.coefDarcy*self.dt)
        # chemical reaction 更新化学物质
        if ti.static(self.source_term_model==SOURCE_TERM.MACRO):
            if ti.static(self.RADIATION): # radiation
                for i in ti.grouped(self.rho):          
                        self.TS.S[i] += self.dt*self.radiation(i)/self.TS.capacity_v(i) # 辐射吸热 # 直接添加到格子中央可能会带来问题
            if ti.static(self.CHEMISTRY):
                for r in ti.static(list(self.reactions.values())):
                    r.reaction_macro() # update species and internal energy
    
    
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
    def feq9(self, k,rho, u,eps = 1.0): #计算平衡分布函数 
        eu = self.e[k].dot(u)
        uv = u.dot(u)
        feqout = self.w[k]*rho*(1.0+3.0*eu+4.5*eu*eu/(eps+1e-12)-1.5*uv/(eps+1e-12))
        return feqout
    @ti.func
    def feq5(self, k,rho_local, u,eps = 1.0): #计算平衡分布函数 考虑多孔介质
        eu = self.e5[k].dot(u)
        uv = u.dot(u)
        feqout = self.w5[k]*rho_local*(1.0+3.0*eu+4.5*eu*eu/(eps+1e-12)-1.5*uv/(eps+1e-12))
        return feqout
    
    @ti.func
    def viscosity(self,i):
        return 0.1
    
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
                    F +=  -self.viscosity(i)*self.coefDarcy*self.rho[i]*self.v[i]
                elif ti.static(self.poro_model==PORO_MODEL.DARCYFORCHHEIMER):
                    F += (-self.viscosity(i)*self.coefDarcy-self.coefForchheimer*ti.math.length(self.v[i]))*self.rho[i]*self.v[i]
        return F
    @ti.func
    def forceTermGuo(self,k,rho,u,F,tau): # 将力转化为分布函数源项 Guo Zhao
        return (1.0-1.0/2.0/tau)*rho*self.w[k]*(3.0*(self.e[k]-u).dot(F)\
              +9.0*self.e[k].dot(u)*self.e[k].dot(F))
    @ti.func
    def scalarCorrectionTerm(self,k,duS,tau):
        return (1.0-1.0/2.0/tau)*3.0*self.w5[k]*self.e5[k].dot(duS)
    @ti.func
    def radiation(self,i):# Wm-2K-4*m-1K4*{m2}=Wm-3 SI # 单位体积辐射 SI
        q = 0.0
        if ti.static(self.radiation_model==RADIATION_MODEL.SURFACE_UNIFORM):
            q += LB2D_PYRO.SIGMA*self.radiation_surface[i]/self.dx*(ti.pow(self.Tambient,4)-ti.pow(self.TS.S[i],4))
        elif ti.static(self.radiation_model==RADIATION_MODEL.REAL_RADIATION):
            q += self.real_radiation[i]-LB2D_PYRO.SIGMA*self.radiation_surface[i]/self.dx*ti.pow(self.TS.S[i],4)

        return q
    #----------
    # 用户使用函数
    #----------
    # 初始化场
    # 设置场初值
    def init_field(self,field,param):
        if(type(param) in [float,int]):
            data = param*np.ones(shape=(self.nx,self.ny,1),dtype=np.float32) 
            field.from_numpy(data)
        if(type(param) is str):
            
            in_dat = np.loadtxt(param,dtype=np.float32)
            in_dat = np.reshape(in_dat, (self.nx,self.ny,self.nz),order='F')
            field.from_numpy(in_dat)
        
    def init_field2(self,field,param1,param2): 
        if(type(param1) in [float,int]):
            dat1 = param1*np.ones(shape=(self.nx,self.ny,1))   
        if(type(param1) is str):
            dat1 = np.loadtxt(param1)
            dat1 = np.reshape(dat1, (self.nx,self.ny,1),order='F')
        if(type(param2) in [float,int]):
            dat2 = param2*np.ones(shape=(self.nx,self.ny,1))   
        if(type(param2) is str):
            dat2 = np.loadtxt(param2)
            dat2 = np.reshape(dat2, (self.nx,self.ny,1),order='F')
        data = np.concatenate((dat1,dat2),axis = 3)
        field.from_numpy(data)   
    def init_field3(self,field,param1,param2,param3): 
        if(type(param1) in [float,int]):
            dat1 = param1*np.ones(shape=(self.nx,self.ny,1))   
        if(type(param1) is str):
            dat1 = np.loadtxt(param1)
            dat1 = np.reshape(dat1, (self.nx,self.ny,1),order='F')

        if(type(param2) in [float,int]):
            dat2 = param2*np.ones(shape=(self.nx,self.ny,1))   
        if(type(param2) is str):
            dat2 = np.loadtxt(param2)
            dat2 = np.reshape(dat2, (self.nx,self.ny,1),order='F')

        if(type(param3) in [float,int]):
            dat3 = param3*np.ones(shape=(self.nx,self.ny,1))   
        if(type(param3) is str):
            dat3 = np.loadtxt(param3)
            dat3 = np.reshape(dat3, (self.nx,self.ny,1),order='F')
        dat1 = np.expand_dims(dat1,3)
        dat2 = np.expand_dims(dat2,3)
        dat3 = np.expand_dims(dat3,3)
        data = np.concatenate((dat1,dat2,dat3),axis = 3)
        field.from_numpy(data)
    # 设置物理场属性
    ## 密度场
    def set_viscosity(self,niu,unit = "lattice"):#定义常黏度
        if unit=="SI":
            niu *=self.dt/self.dx**2
        @ti.func
        def new_viscosity(self,i):
            return niu
        self.viscosity = new_viscosity.__get__(self,LB2D_PYRO)
    def set_viscosity_func(self,func):# 根据方程定义黏度
        self.viscosity = func.__get__(self,LB2D_PYRO)
    ## 多孔介质
    def set_poro_Darcy(self,coefDarcy,unit = "lattice"): # L-2
        self.poro_model = PORO_MODEL.DARCY
        if unit=="SI":
            coefDarcy *= self.dx**2
        self.coefDarcy = coefDarcy
    def set_poro_Darcy_Forchheimer(self,coefDarcy, coefForchheimer,unit = "lattice"):
        self.poro_model = PORO_MODEL.DARCYFORCHHEIMER
        if unit == "SI":
            coefDarcy*=self.dx**2 # L-2
            coefForchheimer*=self.dx #L-1
        self.coefDarcy = coefDarcy
        self.coefForchheimer = coefForchheimer
    ## 温度场
    def set_TS_diff(self,diff,unit = "lattice"):
        if unit=="SI":
            diff *=self.dt/self.dx**2
        @ti.func
        def new_diff(self,i):
            return diff
        self.set_TS_diff_func(new_diff)
    def set_TS_diff_func(self,func):
        self.TS.coefDiff = func.__get__(self.TS,ScalarField)
        
    def set_TF_diff(self,diff,unit = "lattice"):
        if unit=="SI":
            diff *=self.dt/self.dx**2
        @ti.func
        def new_diff(self,i):
            return diff
        self.set_TF_diff_func(new_diff)
    def set_TF_diff_func(self,func):
        self.TF.coefDiff = func.__get__(self.TF,ScalarField)
    ### 辐射相关
    def set_radiation_model(self,model,param):
        if model == RADIATION_MODEL.SURFACE_UNIFORM:
            self.radiation_model = model
            self.Tambient = float(param)
        if model == RADIATION_MODEL.REAL_RADIATION:
            self.radiation_model = model
            self.real_radiation = ti.field(ti.f32,shape=(self.nx,self.ny,self.nz))
            self.init_field(self.real_radiation,param)
    ## 浓度场 (物种密度)
    def set_specie(self,specie,FIX = False):
        self.species[specie] = Specie(specie,self.nx,self.ny,self.nz,self,FIX)
    def set_specie_mole(self,specie,isFix = False,molemass = 1.0):
        self.species[specie] = Specie(specie,self.nx,self.ny,self.nz,self,Mmass = molemass,FIX=isFix)
        
    def set_species(self,species,FIX=None):# 登记所有物质
        if FIX == None:
            FIX = [False]*len(species)
        i = 0
        for name in species:
            self.set_specie(name,FIX[i])
            i += 1
    def set_species_mole(self,species,FIX=None,molemass=None):
        if FIX == None:
            FIX = [False]*len(species)
        if molemass==None:
            molemass=[0.002]*len(species)
        i = 0
        for name in species:
            self.set_specie_mole(name,FIX[i],molemass[i])
            i += 1
    def init_specie(self,name,param):
        self.init_field(self.species[name].S,param)
    def set_specie_diff(self,name,diff,unit="lattice"):
        if unit=="SI":
            diff *=self.dt/self.dx**2
        @ti.func
        def new_diff(self,i):
            return diff
        self.set_specie_diff_func(name,new_diff)
    def set_specie_diff_func(self,name,func):
        self.species[name].coefDiff = func.__get__(self.species[name],ScalarField)
    def set_specie_capacity(self,name:str,cm:float): # 质量热容
        @ti.func
        def new_cm(self,i):
            return cm
        self.set_specie_capacity_func(name,new_cm)
    def set_specie_capacity_func(self,name,func):
        '''
        定义质量热容
        '''
        self.species[name].capacity_m = func.__get__(self.species[name],ScalarField)

    def set_specie_conductivity(self,name,lamb): # 传热系数
        @ti.func
        def new_lamb(self,i):
            return lamb
        self.set_specie_conductivity_func(name,new_lamb)
    def set_specie_conductivity_func(self,name,func):
        self.species[name].conductivity = func.__get__(self.species[name],ScalarField)
    
    # 定义化学反应
    def add_reaction(self,name,reactant,product, param,unit=SPECIE_UNIT.MASS):
        self.reactions[name]=(Reaction(name,reactant,product,param,self,unit=unit))
    
    # 设置边界条件
    def set_BC(self,i,BC):
        self.BC[i]=BC
    def set_BCs(self,BCs):
        self.BC = BCs
    def set_v_BC_value(self,i,v):
        self.v_BC[i]=v
    def set_v_BCs_value(self,vs,unit = "lattice"):
        for i in range(4):
            if unit=="SI":
                vs[i] = np.array(vs[i])*self.dt/self.dx
            self.set_v_BC_value(i,vs[i])
    def set_rho_BC_value(self,i,r):
        self.rho_BC[i]=r
    def set_rho_BCs_value(self,rhos):
        for i in range(4):        
            self.set_rho_BC_value(i,rhos[i])
    def set_TS_BC(self,i,BC):
        self.TS.set_BC(i,BC)
    def set_TS_BCs(self,BCs):
        self.TS.set_BCs(BCs)
    def set_TS_BC_value(self,i,T):
        self.TS.set_s_BC_value(i,T)
    def set_TS_BCs_value(self,Ts):
        self.TS.set_s_BCs_value(Ts)
    def set_TF_BC(self,i,BC):
        self.TF.set_BC(i,BC)
    def set_TF_BCs(self,BCs):
        self.TF.set_BCs(BCs)
    def set_TF_BC_value(self,i,T):
        self.TF.set_s_BC_value(i,T)
    def set_TF_BCs_value(self,Ts):
        self.TF.set_s_BCs_value(Ts)
    
    def set_specie_BC(self,specie,i,BC):
        self.species[specie].set_BC(i,BC)
    def set_specie_BCs(self,specie,BCs):
        self.species[specie].set_BCs(BCs)
    def set_species_BC(self,i,BC):
        for specie in self.species.values():
            specie.set_BCs(i,BC)
    def set_species_BCs(self,BCs):# 定义所有物种的边界条件
        for specie in self.species.values():
            specie.set_BCs(BCs)
    def set_specie_BC_value(self,specie,i,S):
        self.species[specie].set_s_BC_value(i,S)
    def set_specie_BCs_value(self,specie,Ss):
        self.species[specie].set_s_BCs_value(Ss)
    def set_specie_BC_flux(self,specie, i,f):
        self.species[specie].set_s_BC_flux(i,f)
    def set_specie_BCs_flux(self,specie, fs):
        self.species[specie].set_s_BCs_flux(fs)
    def set_specie_BC_percentage(self,specie, i,Y):
        self.species[specie].set_s_BC_Y(i,Y)
    def set_specie_BCs_percentage(self,specie, Ys):
        self.species[specie].set_s_BCs_Y(Ys)
    # 输出 可视化
    def get_max_v(self): # 获得最大速度，用于判断模型是否发散
        self.max_v[None] = -1e10        
        self.cal_max_v()
        return self.max_v[None]
    def get_min_T(self):
        if self.TEMPERATURE:
            self.min_T[None]= 1e10
            self.cal_min_T()
            return self.min_T[None]
        else:
            return -1
    @ti.kernel
    def cal_max_v(self):
        for I in ti.grouped(self.rho):
            ti.atomic_max(self.max_v[None], self.v[I].norm())
    @ti.kernel
    def cal_min_T(self):
        for I in ti.grouped(self.rho):
            ti.atomic_min(self.min_T[None], self.TF.S[I])
    
    

    def export_LBM(self,path):
        os.makedirs(path,exist_ok=True)
        self.export_data(os.path.join(path,"data"))
        self.export_config(os.path.join(path,"config"))

    @classmethod
    def load_LBM(cls,path):
        lbm = LB2D_PYRO.load_config_and_create_LBM(os.path.join(path,"config"))
        with open(os.path.join(path,"data"),"rb") as f:
            data:dict = pickle.load(f)
        for key in data.keys():
            if key == "solid":
                lbm.solid.from_numpy(data[key])
            elif key == "rho":
                lbm.rho.from_numpy(data[key])
            elif key == "velocity":
                lbm.v.from_numpy(np.array(data["velocity"]).transpose(1,2,3,0))
            elif key == "Fluid Temperature":
                lbm.TF.S.from_numpy(data[key])
            elif key == "Solid Temperature":
                lbm.TS.S.from_numpy(data[key])
        for specie in lbm.species.keys():
            lbm.species[specie].S.from_numpy(data[specie])
        return lbm
    @classmethod
    def load_config_and_create_LBM(cls,file): 
        config = {}
        with open(file,"r") as f:
            config = json.load(f)        
        LBM = LB2D_PYRO(config["size"][0],config["size"][1],config["size"][2],config["dx"],config["dt"],isThermal=config["isThermal"],isPoro=config["isPoro"],isChemical=config["isChemical"])
        LBM.set_species(config["species"],config["speciesState"])
        return LBM
    def export_config(self,file):
        config = {
            "size":[self.nx,self.ny,self.nz],
            "dx" : self.dx,
            "dt" : self.dt,
            "isPoro":self.PORO,
            "isChemical":self.CHEMISTRY,
            "isThermal":self.TEMPERATURE,
            "species" : list(self.species.keys()),
            "speciesState" : [specie.FIX for specie in self.species.values()]
        }
        with open(file,'w') as f:
            json.dump(config,f)
        # load 
    def export_data(self, file):
        with open(file,'wb') as f:
            pickle.dump(self.get_data(),f)
    def import_data(self,file):
        pass
    def export_VTK(self, name,n): # 导出为vtk 到 result文件夹中
        path = os.path.join("result",name)
        os.makedirs(path,exist_ok=True)
        gridToVTK(
                os.path.join(path,name+"_"+str(n)),
                self.x,
                self.y,
                self.z,
                #cellData={"pressure": pressure},
                pointData=self.get_data()
            )
    def get_data(self): # 获取所有数据（字典）
        data = {    "solid":self.solid.to_numpy(),
                    "rho": self.rho.to_numpy(),
                    "velocity": (
                        np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:1,0]), 
                        np.ascontiguousarray(self.v.to_numpy()[0:self.nx,0:self.ny,0:1,1]),
                        np.ascontiguousarray(np.zeros_like(self.v.to_numpy())[0:self.nx,0:self.ny,0:1,1]),
                        ),
                }
        if self.TEMPERATURE:
            data["Tf"]  = self.TF.S.to_numpy()
            data["Ts"]  = self.TS.S.to_numpy()
        if self.RADIATION:
            data["Radiation_surface"] = self.radiation_surface.to_numpy()
            if self.radiation_model == RADIATION_MODEL.REAL_RADIATION:
                data["Real Radiation"] = self.real_radiation.to_numpy()
        if self.CHEMISTRY:
            for specie in self.species.values():
                if specie.FIX and not specie.name.endswith("(S)"):
                    data[specie.name+"(S)"]=specie.S.to_numpy()
                else:
                    data[specie.name]=specie.S.to_numpy()
        return data    

    def export_variable(self, name,iter):
        path = os.path.join("result",name)
        os.makedirs(path,exist_ok=True)
        with open(os.path.join(path,name+"_"+str(iter)+".json"),'w') as f:
            json.dump(self.get_variable(iter*self.dt),f)
    def get_variable(self,t):
        res = {}
        res["time"] = t
        for func in self.GetVariableFunc:
            name,value = func(self)
            res[name]=value
        return res
    @ti.func
    def check(self):
        pass
        # rad = 0.0
        # rad_surface = 0.0
        # for i in ti.grouped(self.rho):
        #     rad_surface +=self.radiation_surface[i]*self.dx
        # print("check radiation:",rad_surface)
        # j = [int(self.nx/2),int(self.ny/2),0]
        # print(self.species["wood(S)"].S[j])
        # print(self.IE.S[j]/self.IE.capacity_v_fluid(j), self.IE.S[j],self.species["N2"].S[j],self.IE.capacity_v_fluid(j),self.IE.correction_SF(j))
        # for specie in ti.static(list(self.species.values())):
        #     print(specie.capacity_m(j)*specie.S[j])
        # j1 = [10,int(self.ny/5),0]
        # print(self.TS.coefDiff(j),self.TF.coefDiff(j1))

        # print(self.IE.T[j1],self.IE.coefDiff(j1),self.IE.correction_SF(j1))
        # print(self.IE.T[j1], self.IE.g[j1].sum()/self.IE.capacity_v_fluid(j1))
        # j2 = [2,int(self.ny/2),0]
        # print(self.IE.T[j2], self.IE.g[j2].sum()/self.IE.capacity_v_fluid(j2))
