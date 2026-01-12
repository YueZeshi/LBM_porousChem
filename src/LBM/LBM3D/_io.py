import ast
import taichi as ti
import numpy as np
import os
import json
from ruamel.yaml import YAML
from ._core import LBM3D_BASE
from ..util.flag import *
from ..util import constant
from ._scalarField import ScalarField
from ._chemical import Specie,Reaction
@ti.data_oriented
class LBM3D_INPUT(LBM3D_BASE):
    """用户侧 3D LBM 输入/配置接口。

    主要职责：
    - 配置输出路径、初始场、物性参数与边界条件。
    - 为后续求解阶段提供可复用的设置函数。

    约定：所有场形状默认为 ``(nx, ny, nz)``，存储顺序为 Fortran-order。
    """

    #----------
    # 用户使用函数
    #----------
    def set_vtk_path(self,path):
        """设置 VTK 导出目录。

        Parameters
        ----------
        path : str or PathLike
            目录路径，将在其中写入 .vts/.pvd 等结果文件。
        """
        self.exportPath = path
        self.PVD.exportPath = path
    # 初始化场
    # 设置场初值
    def init_field(self,field,param):
        """用标量或文件初始化单通道场。

        Parameters
        ----------
        field : taichi.field
            目标场（形状 ``(nx, ny, nz)``）。
        param : float | int | str | numpy.ndarray
            - 数值：填充为常数。
            - str：文本文件路径，按 Fortran-order reshape。
            - ndarray：直接写入（需要形状匹配）。
        """
        if(type(param) in [float,int]):
            data = param*np.ones(shape=(self.nx,self.ny,self.nz),dtype=np.float32).astype(np.float32)
            field.from_numpy(data)
        if(type(param) is str):
            in_dat = np.loadtxt(param,dtype=np.float32)
            in_dat = np.reshape(in_dat, (self.nx,self.ny,self.nz),order='F').astype(np.float32)
            field.from_numpy(in_dat)
        if(type(param)==np.ndarray):
            param = np.array(param).astype(np.float32)
            field.from_numpy(param)        
    def init_field2(self,field,param1,param2): 
        """用两组数据初始化双通道场，最后 axis=3 拼接。

        Parameters
        ----------
        field : taichi.field
            目标场，预期形状 ``(nx, ny, nz, 2)``。
        param1, param2 : float | int | str | numpy.ndarray
            支持常数、文件路径或 ndarray，均按 Fortran-order 对齐。
        """
        if(type(param1) in [float,int]):
            dat1 = param1*np.ones(shape=(self.nx,self.ny,self.nz)).astype(np.float32) 
        if(type(param1) is str):
            dat1 = np.loadtxt(param1)
            dat1 = np.reshape(dat1, (self.nx,self.ny,self.nz),order='F').astype(np.float32)
        if(type(param1)==np.ndarray):
            dat1 = np.array(param1).astype(np.float32)
        if(type(param2) in [float,int]):
            dat2 = param2*np.ones(shape=(self.nx,self.ny,self.nz)).astype(np.float32)  
        if(type(param2) is str):
            dat2 = np.loadtxt(param2)
            dat2 = np.reshape(dat2, (self.nx,self.ny,self.nz),order='F').astype(np.float32)
        if(type(param2)==np.ndarray):
            dat2 = np.array(param2).astype(np.float32)
        data = np.concatenate((dat1,dat2),axis = 3)
        field.from_numpy(data)   
    def init_field3(self,field,param1,param2,param3): 
        """用三组数据初始化三通道场，按 axis=3 堆叠。

        Parameters
        ----------
        field : taichi.field
            目标场，预期形状 ``(nx, ny, nz, 3)``。
        param1, param2, param3 : float | int | str | numpy.ndarray
            支持常数、文件路径或 ndarray，均按 Fortran-order 对齐。
        """
        if(type(param1) in [float,int]):
            dat1 = param1*np.ones(shape=(self.nx,self.ny,self.nz)).astype(np.float32)
        if(type(param1) is str):
            dat1 = np.loadtxt(param1)
            dat1 = np.reshape(dat1, (self.nx,self.ny,self.nz),order='F').astype(np.float32)
        if(type(param1)==np.ndarray):
            dat1 = np.array(param1).astype(np.float32)
        if(type(param2) in [float,int]):
            dat2 = param2*np.ones(shape=(self.nx,self.ny,self.nz)).astype(np.float32)  
        if(type(param2) is str):
            dat2 = np.loadtxt(param2)
            dat2 = np.reshape(dat2, (self.nx,self.ny,self.nz),order='F')
        if(type(param2)==np.ndarray):
            dat2 = np.array(param2).astype(np.float32)
        if(type(param3) in [float,int]):
            dat3 = param3*np.ones(shape=(self.nx,self.ny,self.nz)).astype(np.float32)   
        if(type(param3) is str):
            dat3 = np.loadtxt(param3)
            dat3 = np.reshape(dat3, (self.nx,self.ny,self.nz),order='F').astype(np.float32)
        if(type(param3)==np.ndarray):
            dat3 = np.array(param3).astype(np.float32)
        dat1 = np.expand_dims(dat1,3)
        dat2 = np.expand_dims(dat2,3)
        dat3 = np.expand_dims(dat3,3)
        data = np.concatenate((dat1,dat2,dat3),axis = 3)
        field.from_numpy(data)
    # 设置物理场属性   
    def add_solid(self,solid):
        """向固相掩膜累加布尔/数值场，用于逐步添加固体区域。"""
        s = self.solid.to_numpy()
        s += solid
        self.init_field(self.solid,s)
    def add_rho_solid(self,rhos):
        """向固相密度场累加数值，用于叠加固体密度。"""
        s = self.rhos.to_numpy()
        s += rhos
        self.init_field(self.rhos,s)
    def set_heat_exchange_surface(self,surface):
        """设置固/流换热接触面积分布，单位与网格尺度一致。"""
        self.init_field(self.TS.exchangeSurface,surface)
    def set_heat_exchange_coef(self,coef):
        """设置固/流换热系数分布。"""
        self.init_field(self.TS.exchangeCoef,coef)

    ## 密度场
    def set_viscosity(self,niu):#定义常黏度
        """设定常黏度模型，``niu`` 为格子单位黏度。"""
        self.visco = niu
        self.viscosity_model = VISCOSITY_MODEL.CONSTANT
    def set_viscosity_sutherland(self,As,Ts):
        """启用 Sutherland 黏度模型，参数 ``As``、``Ts`` 为模型系数。"""
        self.sutherland_coef = [As,Ts]
        self.viscosity_model = VISCOSITY_MODEL.SUTHERLAND
    def set_viscosity_mixture(self):
        """启用混合物黏度模型（多物种时）。"""
        self.viscosity_model = VISCOSITY_MODEL.MIXTURE

    ## 多孔介质
    def set_poro_Darcy(self,s,coefDarcy): # L-2
        """设置 Darcy 多孔介质阻力。

        Parameters
        ----------
        s : ndarray or scalar
            孔隙分布（0/1 或实数）。
        coefDarcy : float
            Darcy 系数，内部自动乘以 ``dx**2``。
        """
        self.poro_model = PORO_MODEL.DARCY
        coefDarcy *=self.dx**2
        self.init_field(self.coefDarcy,coefDarcy*s)
    def set_poro_Darcy_Forchheimer(self,s,coefDarcy, coefForchheimer):
        """设置 Darcy-Forchheimer 多孔介质阻力，含线性与二次项。"""
        self.poro_model = PORO_MODEL.DARCYFORCHHEIMER
        coefDarcy *=self.dx**2
        coefForchheimer *=self.dx
        self.init_field(self.coefDarcy,coefDarcy*s)
        self.init_field(self.coefForchheimer,coefForchheimer*s)
    
    ## 温度场
    def set_fluid_thermal_diff(self,diff):
        """流体：设定常数热扩散系数。"""
        self.TF.thermal_diff_model = THERMAL_DIFF_MODEL.CONSTANT
        self.TF.thermal_diff= diff
    def set_fluid_Prandtl(self,Pr):
        """流体：通过 Pr 数确定热扩散系数。"""
        self.TF.thermal_diff_model = THERMAL_DIFF_MODEL.PRANDTL
        self.TF.Pr = Pr
    def set_fluid_thermal_diff_derived(self):
        """流体：使用模型自适应热扩散。"""
        self.TF.thermal_diff_model = THERMAL_DIFF_MODEL.DERIVED
    def set_fluid_conductivity(self,value):
        """流体：设定常数导热系数。"""
        self.TF.conductivity_model = CONDUCTIVITY_MODEL.CONSTANT
        self.TF.cond = value
    def set_fluid_conductivity_poly(self,poly):
        """流体：设定多项式导热系数（按温度）。"""
        self.TF.conductivity_model = CONDUCTIVITY_MODEL.POLYNOMIAL
        self.TF.cond_poly = poly 
    def set_fluid_conductivity_mixture(self):
        """流体：启用混合物导热模型。"""
        self.TF.conductivity_model = CONDUCTIVITY_MODEL.MIXTURE
    def set_fluid_capacity(self,value):
        """流体：设定常数比热容。"""
        self.TF.capacity_model = THERMO_MODEL.CONSTANT
        self.TF.cm = value
        print(value)
    def set_fluid_capacity_poly(self,poly):
        """流体：设定多项式比热容。"""
        self.TF.capacity_model = THERMO_MODEL.POLYNOMIAL
        self.TF.cm_poly = poly 
    def set_fluid_capacity_NASA7(self,Trange,data):
        """流体：启用 NASA7 多项式比热模型。"""
        self.TF.capacity_model = THERMO_MODEL.NASA7
        self.TF.Trange = Trange
        self.TF.NASA_coef = data
    def set_fluid_capacity_mixture(self):
        """流体：混合物比热模型。"""
        self.TF.capacity_model = THERMO_MODEL.MIXTURE
    def set_fluid_Trange(self,Trange):
        """流体：设置参考温度区间 ``[T_ref, T_max]`` 以归一化。"""
        self.TF.v_ref = Trange[0]
        self.TF.v_scale = Trange[1]-Trange[0]
        
    def set_solid_thermal_diff(self,diff):
        """固体：设定常数热扩散系数。"""
        self.TS.thermal_diff_model = THERMAL_DIFF_MODEL.CONSTANT
        self.TS.thermal_diff= diff
    def set_solid_thermal_diff_derived(self):
        """固体：使用模型自适应热扩散。"""
        self.TS.thermal_diff_model = THERMAL_DIFF_MODEL.DERIVED
    def set_solid_conductivity(self,value):
        """固体：设定常数导热系数。"""
        self.TS.conductivity_model = CONDUCTIVITY_MODEL.CONSTANT
        self.TS.cond = value
    def set_solid_conductivity_poly(self,poly):
        """固体：设定多项式导热系数。"""
        self.TS.conductivity_model = CONDUCTIVITY_MODEL.POLYNOMIAL
        self.TS.cond_poly = poly 
    def set_solid_conductivity_mixture(self):
        """固体：启用混合物导热模型。"""
        self.TS.conductivity_model = CONDUCTIVITY_MODEL.MIXTURE
    def set_solid_capacity(self,value):
        """固体：设定常数比热容。"""
        self.TS.capacity_model = THERMO_MODEL.CONSTANT
        self.TS.cm = value
    def set_solid_capacity_poly(self,poly):
        """固体：设定多项式比热容。"""
        self.TS.capacity_model = THERMO_MODEL.POLYNOMIAL
        self.TS.cm_poly = poly 
    def set_solid_capacity_NASA7(self,Trange,data):
        """固体：启用 NASA7 多项式比热模型。"""
        self.TS.capacity_model = THERMO_MODEL.NASA7
        self.TS.Trange = Trange
        self.TS.NASA_coef = data

    def set_solid_capacity_mixture(self):
        self.TS.capacity_model = THERMO_MODEL.MIXTURE

    def set_solid_Trange(self,Trange):
        """固体：设置参考温度区间 ``[T_ref, T_max]`` 以归一化。"""
        self.TS.v_ref = Trange[0]
        self.TS.v_scale = Trange[1]-Trange[0]

    ### 辐射相关
    def set_radiation(self,model,param):
        """设置辐射模型。

        Parameters
        ----------
        model : RADIATION_MODEL
            当前支持 ``SURFACE_UNIFORM``。
        param : float
            当均匀辐射时为环境温度。未来扩展其他模型可重用此接口。
        """
        if model == RADIATION_MODEL.SURFACE_UNIFORM:
            self.TS.radiation_model = model
            self.TS.Tambient = float(param)
        # if model == RADIATION_MODEL.REAL_RADIATION:
        #     self.TS.radiation_model = model
        #     self.TS.real_radiation = ti.field(float,shape=(self.nx,self.ny,self.nz))
        #     self.init_field(self.TS.real_radiation,param)
    
    ## 浓度场 (物种密度)

    def set_specie(self,specie,Fix = False,molemass = 1.0):
        """注册单个物种。

        Parameters
        ----------
        specie : str
            物种名称。
        Fix : bool, default False
            是否视为固定场（不参与对流扩散）。
        molemass : float, default 1.0
            摩尔质量，用于混合物性质计算。
        """
        self.species.append(Specie(specie,self,Mmass = molemass,FIX=Fix))
        self.specieName.append(specie)
    def set_specie_viscosity(self,specieName,value):
        """为指定物种设置常数黏度。"""
        specie = self.species[self.specieName.index(specieName)]
        specie.viscosity_type = VISCOSITY_MODEL.CONSTANT
        specie.visco = value       
    def set_specie_viscosity_sutherland(self,specieName,coef):
        """为指定物种启用 Sutherland 黏度。"""
        specie = self.species[self.specieName.index(specieName)]
        specie.viscosity_type = VISCOSITY_MODEL.SUTHERLAND
        specie.coefSutherland = coef 
    def set_specie_NASA7(self,specieName:str,TRange:list[float],coef:list):
        """为指定物种设置 NASA7 热容模型。"""
        specie = self.species[self.specieName.index(specieName)]
        specie.Trange = TRange
        specie.NASAcoef = coef
    def set_species(self,species,FIX=None,molemass = None):# 登记所有物质
        """批量注册物种。

        Parameters
        ----------
        species : list[str]
            物种名称列表。
        FIX : list[bool] | None
            是否固定的布尔列表；为空时默认全 False。
        molemass : list[float] | None
            摩尔质量列表；为空时默认 0.028 kg/mol。"""
        if FIX == None:
            FIX = [False]*len(species)
        if molemass==None:
            molemass = [0.028]*len(species)
        i = 0
        for name in species:
            self.set_specie(name,FIX[i])
            i += 1
    def init_specie(self,name,param):
        """初始化指定物种场，同 `init_field` 语义。"""
        self.init_field(self.species[self.specieName.index(name)].S,param)
    def set_specie_diff(self,specieName,diff):
        """指定物种：常数扩散系数。"""
        specie = self.species[self.specieName.index(specieName)]
        specie.diff_model = DIFF_MODEL.CONSANT
        specie.diff = diff
    def set_specie_diff_Schmidt(self,specieName,Sc):
        """指定物种：通过 Schmidt 数设置扩散。"""
        specie = self.species[self.specieName.index(specieName)]
        specie.diff_model=DIFF_MODEL.SCHMIDT
        specie.Sc = Sc
    def set_specie_enthalpy(self,specieName,enthalpy):
        """指定物种：常数焓。"""
        specie = self.species[self.specieName.index(specieName)]
        specie.thermo_model = THERMO_MODEL.CONSTANT
        specie.enthalpy = enthalpy
    def set_specie_capacity(self,specieName,capacity): # 质量热容
        """指定物种：常数质量比热容。"""
        specie = self.species[self.specieName.index(specieName)]
        specie.thermo_model = THERMO_MODEL.CONSTANT
        specie.capa = capacity

    def set_specie_conductivity(self,name,lamb): # 传热系数
        """指定物种：常数导热系数。"""
        specie = self.species[self.specieName.index(name)]
        specie.cond_model=CONDUCTIVITY_MODEL.CONSTANT
        specie.cond = lamb
    
    def set_specie_conductivity_poly(self,name,poly): # 传热系数
        """指定物种：温度多项式导热系数。"""
        specie = self.species[self.specieName.index(name)]
        specie.cond_model=CONDUCTIVITY_MODEL.POLYNOMIAL
        specie.cond_poly = list(poly)

    # 定义化学反应
    def add_reaction(self,formula,A,Ea,b = 0,Tmin = 0,deltaH = 0,name="unnamed",unit=SPECIE_UNIT.MASS,fixDH = True):
        """登记单条化学反应。

        Parameters
        ----------
        formula : str
            反应式，遵循 cantera 写法。
        A, Ea, b : float
            Arrhenius 系数。
        Tmin : float, default 0
            触发温度阈值。
        deltaH : float, default 0
            反应焓变。
        name : str
            反应名称。
        unit : SPECIE_UNIT
            质量或摩尔基准。
        fixDH : bool
            是否固定焓项。
        """
        self.reactions.add_reaction(Reaction(self,formula,A,Ea,b,Tmin,deltaH,name,unit,fixDH))
    
    # 设置边界条件
    def set_BC(self,i,bc):
        """设置流场单边界类型（3D 共 6 面：x-/x+/y-/y+/z-/z+）。"""
        self.bc[i]=bc
        if bc==BC_FLOW.inlet:
            self.set_v_BC(i,BC.fixedValue)
            self.set_rho_BC(i,BC.zeroGradient)
        if bc==BC_FLOW.outlet:
            self.set_v_BC(i,BC.zeroGradient)
            self.set_rho_BC(i,BC.fixedValue)
        if bc==BC_FLOW.symmetric:
            self.set_v_BC(i,BC.zeroGradient)
            self.set_rho_BC(i,BC.zeroGradient)
        if bc==BC_FLOW.wall:
            self.set_v_BC(i,BC.fixedValue)
            self.set_rho_BC(i,BC.zeroGradient)
        if bc==BC_FLOW.inlet_flow:
            self.set_v_BC(i,BC.fixedValue)
            self.set_rho_BC(i,BC.zeroGradient)
    def set_BCs(self,BCs):
        """批量设置六个流场边界类型。"""
        for i in range(6):
            self.set_BC(i,BCs[i])
    def set_v_BC(self,i,bc):
        """设置速度边界条件类型。"""
        self.bc_v[i]=bc
    def set_rho_BC(self,i,bc):
        """设置密度边界条件类型。"""
        self.bc_rho[i]=bc
    def set_v_BC_value(self,i,v):
        """设置速度边界数值（向量或标量）。"""
        self.v_BC[i]=v
    def set_v_BCs_value(self,vs):
        """批量设置速度边界数值（长度应为 6）。"""
        for i in range(6):
            self.set_v_BC_value(i,vs[i])
    def set_rho_BC_value(self,i,r):
        """设置密度边界数值。"""
        self.rho_BC[i]=r
    def set_rho_BCs_value(self,rhos):
        """批量设置密度边界数值（长度应为 6）。"""
        for i in range(6):        
            self.set_rho_BC_value(i,rhos[i])
    def set_flow_BC_value(self,i,f):
        """设置流量边界数值。"""
        self.flow_BC[i]=f
    def set_flow_BCs_value(self,fs):
        """批量设置流量边界数值（当前实现仅前 4 面）。"""
        for i in range(4):
            self.set_flow_BC_value(i,fs[i])


    def set_TS_BC(self,i,BC):
        """固体温度场：设置边界类型。"""
        self.TS.set_BC(i,BC)
    def set_TS_BCs(self,BCs):
        """固体温度场：批量设置六面边界类型。"""
        self.TS.set_BCs(BCs)
    def set_TS_BC_value(self,i,T):
        """固体温度场：设置边界温度。"""
        self.TS.set_s_BC_value(i,T)
    def set_TS_BCs_value(self,Ts):
        """固体温度场：批量设置边界温度。"""
        self.TS.set_s_BCs_value(Ts)
    def set_TF_BC(self,i,BC):
        """流体温度场：设置边界类型。"""
        self.TF.set_BC(i,BC)
    def set_TF_BCs(self,BCs):
        """流体温度场：批量设置六面边界类型。"""
        self.TF.set_BCs(BCs)
    def set_TF_BC_value(self,i,T):
        """流体温度场：设置边界温度。"""
        self.TF.set_s_BC_value(i,T)
    def set_TF_BCs_value(self,Ts):
        """流体温度场：批量设置边界温度。"""
        self.TF.set_s_BCs_value(Ts)
    
    def set_specie_BC(self,specie,i,BC):
        """单物种：设置边界类型。"""
        self.species[self.specieName.index(specie)].set_BC(i,BC)
    def set_specie_BCs(self,specie,BCs):
        """单物种：批量设置六面边界类型。"""
        self.species[self.specieName.index(specie)].set_BCs(BCs)
    def set_species_BC(self,i,BC):
        """所有物种：统一设置单侧边界类型。"""
        for specie in self.species:
            specie.set_BC(i,BC)
    def set_species_BCs(self,BCs):# 定义所有物种的边界条件
        """所有物种：批量设置六面边界类型。"""
        for specie in self.species:
            specie.set_BCs(BCs)
    def set_specie_BC_value(self,specie,i,S):
        """单物种：设置边界数值。"""
        self.species[self.specieName.index(specie)].set_s_BC_value(i,S)
    def set_specie_BCs_value(self,specie,Ss):
        """单物种：批量设置边界数值。"""
        self.species[self.specieName.index(specie)].set_s_BCs_value(Ss)
    def set_specie_BC_flux(self,specie, i,f):
        """单物种：设置边界通量。"""
        self.species[self.specieName.index(specie)].set_s_BC_flux(i,f)
    def set_specie_BCs_flux(self,specie, fs):
        """单物种：批量设置边界通量。"""
        self.species[self.specieName.index(specie)].set_s_BCs_flux(fs)
    
    def load_stl(self,stl_path,scale = [1.0,1.0,1.0],translate = [0,0,0],rotate = [0,0,0],**kwargs):
        """
        将 STL 体素化到模拟域，返回体素网格与表面数组。

        Parameters
        ----------
        stl_path : str or PathLike
            STL 文件路径。
        scale, translate, rotate : list[float]
            缩放、平移、旋转参数。
        logger : optional
            通过 ``kwargs['logger']`` 传入日志对象。
        """
        from ..GEO.STL import StlReader
        stlReader = StlReader(self.X,self.Y,self.Z,self.dx,3,logger=kwargs["logger"])
        return stlReader.voxel_stl(stl_path,scale,translate,rotate)
        
    
    def load_cantera(self,file):
        """从 cantera yaml 机理文件加载物种与反应。

        Parameters
        ----------
        file : str or PathLike
            ``cantera`` 兼容的机理 yaml 文件路径。
        Notes
        -----
        - 3D 版使用 `set_specie_mole` 登记摩尔质量。
        """
        yaml = YAML()
        with open(file,"r") as f:
            data = yaml.load(f)
            for specie_info in data["species"]:
                name = specie_info["name"]
                mmass = 0.0
                for elem,number in specie_info["composition"].items():
                    mmass+=number*constant.MOLEMASS[elem]
                if name.endswith("(S)"):
                    self.set_specie(name,Fix = True,molemass=mmass)
                else:
                    self.set_specie(name,Fix = False,molemass=mmass)
                self.set_specie_NASA7(name,specie_info["thermo"]["temperature-ranges"],specie_info["thermo"]["data"])
                
            for reaction_info in data["reactions"]:
                A = reaction_info["rate-constant"]["A"]
                Ea = reaction_info["rate-constant"]["Ea"]
                b = reaction_info["rate-constant"]["b"]
                self.add_reaction(reaction_info["equation"],A,Ea,b,unit = SPECIE_UNIT.MOLE,fixDH = False)
    
# 输出 可视化
@ti.data_oriented
class LBM3D_OUTPUT(LBM3D_BASE):
    def get_max_v(self): # 获得最大速度，用于判断模型是否发散
        """返回当前域内速度模的最大值（格子单位）。"""
        self.max_v[None] = -1e10        
        self.cal_max_v()
        return self.max_v[None]
    def get_max_T(self):
        """返回流体温度的最大值（物理单位）。

        当未启用温度模块时，返回 -1。
        """
        if self.TEMPERATURE:
            self.max_T[None]= -1e10
            self.cal_max_T()
            return self.TF.get_physical_value(self.max_T[None])
        else:
            return -1
    def get_min_T(self):
        """返回流体温度的最小值（当前实现为格子单位）。

        当未启用温度模块时，返回 -1。
        Notes
        -----
        - 若需物理单位，可在调用处使用 `TF.get_physical_value(...)` 转换。
        """
        if self.TEMPERATURE:
            self.min_T[None]= 1e10
            self.cal_min_T()
            return self.TF.get_physical_value(self.min_T[None])
        else:
            return -1
    @ti.kernel
    def cal_max_v(self):
        """内部 Taichi kernel：计算最大速度模（LU）。"""
        for I in ti.grouped(self.rho):
            ti.atomic_max(self.max_v[None], self.v[I].norm())
    @ti.kernel
    def cal_max_T(self):
        """内部 Taichi kernel：计算最大温度（LU）。"""
        for I in ti.grouped(self.rho):
            ti.atomic_max(self.max_T[None], self.TF.S[I])
    @ti.kernel
    def cal_min_T(self):
        """内部 Taichi kernel：计算最小温度（LU）。"""
        for I in ti.grouped(self.rho):
            ti.atomic_min(self.min_T[None], self.TF.S[I])
    def log_info(self):
        """生成当前步简要日志字符串。"""
        p = f"    t(LU)={self.tLattice} : Max velocity magnitude (LU) : {self.get_max_v():.7f}, "
        if self.TEMPERATURE:
            p +=f"Min temperature: {self.get_min_T():.7f} K, Max temperature : {self.get_max_T():.7f}"
        return p
    def export_snapshot(self,config):
        """导出快照（预留接口）。当前未实现。"""
        pass

    def export_LBM(self,path):
        """导出 LBM 快照为 JSON。

        文件包含基础配置、边界条件以及关键场数据。
        该格式用于检查/复现实验设置，不替代 VTK 可视化输出。
        """
        lbm_info = {"TYPE":"snapshot"}
        # config
        ## BASIC
        basic_info = {
        "name":self.name,
        "X":self.X,
        "Y":self.Y,
        "Z":self.Z,
        "DX":self.dx,
        "DT":self.dt,
        "PORO":self.PORO,
        "CHEMISTRY":self.CHEMISTRY,
        "TEMPERATURE":self.TEMPERATURE,
        "RADIATION":self.RADIATION
        }
        # lbm_info["BASIC"]=basic_info
        # ## FLOW
        # flow_info = {}
        # flow_info["viscosity"] = {
        #     "function":"uniform",
        #     "value":1e-5
        #     }
        # flow_BC = {}
        # for i in range(6):
        #     flow_BC_side = {}
        #     if self.bc[i]==BC_FLOW.periodic:
        #         flow_BC_side["type"]="periodic"
        #     elif self.bc[i]==BC_FLOW.inlet:
        #         flow_BC_side["type"]="inlet"
        #         flow_BC_side["velocity"] = list(self.v_BC[i])
        #     elif self.bc[i]==BC_FLOW.outlet:
        #         flow_BC_side["type"]="outlet"
        #         flow_BC_side["rho"]= self.rho_BC[i]
        #     elif self.bc[i]==BC_FLOW.wall:
        #         flow_BC_side["type"]="wall"
        #     elif self.bc[i]==BC_FLOW.symmetric:
        #         flow_BC_side["type"]="symmetric"
        #     flow_BC[self.sideName[i]]=flow_BC_side
        # flow_info["boundaryCondition"]=flow_BC
        # flow_info["f"]=self.f.to_numpy().tolist()
        # flow_info["v"] = self.v.to_numpy().tolist()
        # lbm_info["FLOW"] = flow_info
        # ## SOLID
        # solid = self.solid.to_numpy().tolist()
        # lbm_info["SOLID"]=solid
        # if self.TEMPERATURE:
        #     ## TEMPERATURE FLUID
        #     TF_info = {}
        #     lbm_info["TEMPERATURE_FLUID"]=TF_info
        #     ## TEMPERATURE SOLID
        #     TS_info = {}
        #     lbm_info["TEMPERATURE_SOLID"]=TS_info
        #     ## RADIATION
        #     if self.RADIATION:
        #         radiation_info = {}
        #         lbm_info["RADIATION"]=radiation_info
        # ## SPECIES
        # if self.CHEMISTRY:
        #     species_info = {}
        # with open(path,"w") as f:
        #     json.dump(lbm_info,f,indent=4)
   

    def export_VTK_pyevtk(self): # 导出为vtk 到指定文件夹中
        """使用 pyevtk 导出 `.vtr` 并更新 `.pvd` 列表。

        输出：
        - `exportPath/name_t.vtr`
        - `exportPath/name.pvd` 累计时间步

        要求：
        - 请先调用 `set_vtk_path()` 设置 `exportPath`。
        """
        from pyevtk.hl import gridToVTK
        filename = os.path.join(self.exportPath,self.name+"_"+str(self.tLattice))
        gridToVTK(
                filename,
                self.x,
                self.y,
                self.z,
                # cellData={"pressure": pressure},
                pointData=self.get_data_pyevtk()
            )
        self.PVD.addVTK(self.tLattice*self.dt,os.path.basename(filename)+".vtr")
        self.PVD.writePVD()
    def export_VTK_pyvista(self): # 导出为vtk 到指定文件夹中
        """使用 pyvista 导出 `.vts` 并更新 `.pvd` 列表。

        输出：
        - `exportPath/name_t.vts`
        - `exportPath/name.pvd` 累计时间步

        要求：
        - 请先调用 `set_vtk_path()` 设置 `exportPath`。
        """
        import pyvista as pv
        filename = os.path.join(self.exportPath,self.name+"_"+str(self.tLattice)+".vts")
        grid = pv.StructuredGrid(self.meshX,self.meshY,self.meshZ)
        grid.point_data.update(self.get_data_pyvista())
        grid.save(filename)
        self.PVD.addVTK(self.tLattice*self.dt,os.path.basename(filename))
        self.PVD.writePVD()
    def get_data_pyvista(self):
        """组装 pyvista 友好的点数据字典。

        Returns
        -------
        dict
            键包含：
            - `solid` : (N,) 0/1 掩膜
            - `rho` : (N,) 密度
            - `velocity` : (N, 3) 速度向量（LU）
            - 温度开启时附加：`rho_solid`, `Tf`, `Ts`, `dTf`, `dTs`,
              `solid_fluid_exchange_surface`, `solid_fluid_exchange_coef`, 以及辐射相关字段
            - 化学开启时附加：各物种 `name`、`d name`
        Notes
        -----
        - `Tf`/`Ts` 已转换为物理单位（K）。
        - 各数组以 Fortran-order 打平成 `N = nx*ny*nz`。
        """
        data = {    "solid":self.solid.to_numpy().ravel(order="F"),
                    "rho": self.rho.to_numpy().ravel(order="F"),
                    "velocity": self.v.to_numpy().reshape(-1,3,order="F")
                }
        # if self.PORO:
        #     data["solid_init"]=self.rho1.to_numpy()
        if self.TEMPERATURE:
            data["rho_solid"] = self.rhos.to_numpy().ravel(order="F")
            data["Tf"]  = self.TF.get_physical_value(self.TF.S.to_numpy().ravel(order="F"))
            data["Ts"]  = self.TS.get_physical_value(self.TS.S.to_numpy().ravel(order="F"))
            data["dTf"] = self.TF.dS.to_numpy().ravel(order="F")
            data["dTs"] = self.TS.dS.to_numpy().ravel(order="F")
            data["solid_fluid_exchange_surface"] = self.TS.exchangeSurface.to_numpy().ravel(order="F")
            data["solid_fluid_exchange_coef"] = self.TS.exchangeCoef.to_numpy().ravel(order="F")
            if self.RADIATION:
                data["radiation_surface"] = self.TS.radiation_surface.to_numpy().ravel(order="F")
                data["emissivity"] = self.TS.emissivity.to_numpy().ravel(order="F")
                # if self.TS.radiation_model == RADIATION_MODEL.REAL_RADIATION:
                #     data["Real Radiation"] = self.TS.real_radiation.to_numpy().ravel(order="F")
        if self.CHEMISTRY:
            for specie in self.species:
                if specie.FIX and not specie.name.endswith("(S)"):
                    data[specie.name+"(S)"]=specie.S.to_numpy().ravel(order="F")
                else:
                    data[specie.name]=specie.S.to_numpy().ravel(order="F")
                    data["d "+specie.name] = specie.dS.to_numpy().ravel(order="F")
        # for key,value in data.items():
        #     print(key,np.shape(value))
        return data    

    def get_data_pyevtk(self): # 获取所有数据（字典）
        """组装 pyevtk 友好的点数据字典。

        Returns
        -------
        dict
            键包含：
            - `solid` : (nx,ny,nz)
            - `rho` : (nx,ny,nz)
            - `velocity` : 三个分量三元组 `(vx, vy, vz)`，每个为 (nx,ny,nz)
            - 温度/化学相关字段与 `get_data_pyvista` 类似
        Notes
        -----
        - `Tf`/`Ts` 在此接口也转换为物理单位（K）。
        """
        data = {    "solid":self.solid.to_numpy(),
                    "rho": self.rho.to_numpy(),
                    "velocity": (
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,0]), 
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,1]), 
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,2]), 
                        ),
                }
        # if self.PORO:
        #     data["solid_init"]=self.rho1.to_numpy()
        if self.TEMPERATURE:
            data["rho_solid"] = self.rhos.to_numpy()
            data["Tf"]  = self.TF.get_physical_value(self.TF.S.to_numpy())
            data["Ts"]  = self.TS.get_physical_value(self.TS.S.to_numpy())
            data["dTf"] = self.TF.dS.to_numpy()
            data["dTs"] = self.TS.dS.to_numpy()
            data["solid_fluid_exchange_surface"] = self.TS.exchangeSurface.to_numpy()
            data["solid_fluid_exchange_coef"] = self.TS.exchangeCoef.to_numpy()
            if self.RADIATION:
                data["radiation_surface"] = self.TS.radiation_surface.to_numpy()
                data["emissivity"] = self.TS.emissivity.to_numpy()
                # if self.TS.radiation_model == RADIATION_MODEL.REAL_RADIATION:
                #     data["Real Radiation"] = self.TS.real_radiation.to_numpy()
        if self.CHEMISTRY:
            for specie in self.species:
                if specie.FIX and not specie.name.endswith("(S)"):
                    data[specie.name+"(S)"]=specie.S.to_numpy()
                else:
                    data[specie.name]=specie.S.to_numpy()
                    data["d "+specie.name] = specie.dS.to_numpy()
        return data    
    def export_variable(self, name,iter):
        """将 `get_variable(iter*dt)` 的结果保存为 JSON 文件。

        文件路径：`result/{name}/{name}_{iter}.json`。
        若目录不存在会自动创建。
        """
        path = os.path.join("result",name)
        os.makedirs(path,exist_ok=True)
        with open(os.path.join(path,name+"_"+str(iter)+".json"),'w') as f:
            json.dump(self.get_variable(iter*self.dt),f)
    def get_variable(self,t):
        """收集用户注册的变量回调并返回字典。

        每个回调函数应形如 `lambda self: (name, value)`。
        返回值中包含 `time: t` 键。
        """
        res = {}
        res["time"] = t
        for func in self.GetVariableFunc:
            name,value = func(self)
            res[name]=value
        return res
    @ti.kernel
    def check_python(self):
        """运行调试检查（Taichi kernel 入口）。"""
        self.check()
    @ti.func
    def check(self):
        """内部调试函数：示例性打印/计算，供开发时验证。"""
        s0 = [0,int(self.ny/2),int(self.nz/2)]
        s1 = [50,int(self.ny/2),int(self.nz/2)]
        print(self.TF.S[s1])
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
