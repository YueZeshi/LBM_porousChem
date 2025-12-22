import taichi as ti
import numpy as np
import os
from pyevtk.hl import gridToVTK
import pickle
import json

from ruamel.yaml import YAML
from ._core import LBM2D_BASE
from LBM.GEO.G2D import Mesh2D
from ..util.flag import *
from ..util import constant
from ._scalarField import ScalarField
from ._chemical import Specie,Reaction
from ._thermal import TemperatureFluid,TemperatureSolid
@ti.data_oriented
class LBM2D_INPUT(LBM2D_BASE):
    #----------
    # 用户使用函数
    #----------
    # 初始化场
    # 设置场初值
    def set_vtk_path(self,path):
        self.path = path
        self.PVD.path = path
    def init_field(self,field,param):
        if(type(param) in [float,int]):
            data = param*np.ones(shape=(self.nx,self.ny,self.nz),dtype=np.float32) 
            field.from_numpy(data)
        if(type(param) is str):
            in_dat = np.loadtxt(param,dtype=np.float32)
            in_dat = np.reshape(in_dat, (self.nx,self.ny,self.nz),order='F')
            field.from_numpy(in_dat)
        if(type(param)==np.ndarray):
            field.from_numpy(param)

        
    def init_field2(self,field,param1,param2): 
        if(type(param1) in [float,int]):
            dat1 = param1*np.ones(shape=(self.nx,self.ny,self.nz))   
        if(type(param1) is str):
            dat1 = np.loadtxt(param1)
            dat1 = np.reshape(dat1, (self.nx,self.ny,self.nz),order='F')
        if(type(param1)==np.ndarray):
            dat1 = param1
        if(type(param2) in [float,int]):
            dat2 = param2*np.ones(shape=(self.nx,self.ny,self.nz))   
        if(type(param2) is str):
            dat2 = np.loadtxt(param2)
            dat2 = np.reshape(dat2, (self.nx,self.ny,self.nz),order='F')
        data = np.concatenate((dat1,dat2),axis = 3)
        field.from_numpy(data)   
    def init_field3(self,field,param1,param2,param3): 
        if(type(param1) in [float,int]):
            dat1 = param1*np.ones(shape=(self.nx,self.ny,self.nz))   
        if(type(param1) is str):
            dat1 = np.loadtxt(param1)
            dat1 = np.reshape(dat1, (self.nx,self.ny,self.nz),order='F')
        if(type(param1)==np.ndarray):
            dat1 = np.array(param1)
        if(type(param2) in [float,int]):
            dat2 = param2*np.ones(shape=(self.nx,self.ny,self.nz))   
        if(type(param2) is str):
            dat2 = np.loadtxt(param2)
            dat2 = np.reshape(dat2, (self.nx,self.ny,self.nz),order='F')
        if(type(param2)==np.ndarray):
            dat2 = np.array(param2)
        if(type(param3) in [float,int]):
            dat3 = param3*np.ones(shape=(self.nx,self.ny,self.nz))   
        if(type(param3) is str):
            dat3 = np.loadtxt(param3)
            dat3 = np.reshape(dat3, (self.nx,self.ny,self.nz),order='F')
        if(type(param3)==np.ndarray):
            dat3 = np.array(param3)
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
        self.viscosity = new_viscosity.__get__(self,LBM2D_INPUT)
    def set_viscosity_func(self,func):# 根据方程定义黏度
        self.viscosity = func.__get__(self,LBM2D_INPUT)
    ## 多孔介质
    def set_poro_Darcy(self,coefDarcy,unit = "lattice"): # L-2
        self.poro_model = PORO_MODEL.DARCY
        if unit=="SI":
            coefDarcy *= self.dx**2
        self.init_field(self.coefDarcy,coefDarcy)
    def set_poro_Darcy_Forchheimer(self,coefDarcy, coefForchheimer,unit = "lattice"):
        self.poro_model = PORO_MODEL.DARCYFORCHHEIMER
        if unit == "SI":
            coefDarcy*=self.dx**2 # L-2
            coefForchheimer*=self.dx #L-1
        self.init_field(self.coefDarcy,coefDarcy)
        self.init_field(self.coefForchheimer,coefForchheimer)
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
    def set_radiation(self,model,param):
        if model == RADIATION_MODEL.SURFACE_UNIFORM:
            self.TS.radiation_model = model
            self.TS.Tambient = float(param)
        if model == RADIATION_MODEL.REAL_RADIATION:
            self.TS.radiation_model = model
            self.TS.real_radiation = ti.field(float,shape=(self.nx,self.ny,self.nz))
            self.init_field(self.TS.real_radiation,param)
    ## 浓度场 (物种密度)
    def set_specie(self,specie,FIX = False):
        self.species.append(Specie(specie,self,FIX))
        self.specieName.append(specie)

    def set_specie_mole(self,specie,Fix = False,molemass = 1.0):
        self.species.append(Specie(specie,self,Mmass = molemass,FIX=Fix))
        self.specieName.append(specie)
    def set_specie_NASA7(self,specieName:str,TRange:list[float],coef:list):
        specie = self.species[self.specieName.index(specieName)]
        for i in range(len(TRange)):
            specie.Trange[i] = TRange[i]
        if len(coef)==1:
            for i in range(7):
                specie.NASAcoef1[i] = coef[0][i]
                specie.NASAcoef2[i] = coef[0][i]
        else:
            for i in range(7):
                specie.NASAcoef1[i] = coef[0][i]
                specie.NASAcoef2[i] = coef[1][i]
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
        self.init_field(self.species[self.specieName.index(name)].S,param)
    def set_specie_diff(self,name,diff,unit="lattice"):
        if unit=="SI":
            diff *=self.dt/self.dx**2
        @ti.func
        def new_diff(self,i):
            return diff
        self.set_specie_diff_func(name,new_diff)
    def set_specie_diff_func(self,name,func):
        self.species[self.specieName.index(name)].coefDiff = func.__get__(self.species[self.specieName.index(name)],ScalarField)
    def set_specie_capacity(self,name:str,cm:float): # 质量热容
        self.set_specie_NASA7(name,[0,0,0],[[0,cm,0,0,0,0,0]]) # Jkg-1K-1 to kJkg-1K-1

    def set_specie_conductivity(self,name,lamb): # 传热系数
        self.species[self.specieName.index(name)].default_conductivity = lamb
    # 定义化学反应
    def add_reaction(self,formula,A,Ea,b = 0,Tmin = 0,deltaH = 0,name="unnamed",unit=SPECIE_UNIT.MASS,fixDH = True):
        self.reactions.add_reaction(Reaction(formula,A,Ea,b,Tmin,deltaH,self,name,unit,fixDH))
    
    # 设置边界条件
    def set_BC(self,i,bc): # v and rho can't be fixed together - overconstrain
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
        for i in range(4):
            self.set_BC(i,BCs[i])
    def set_v_BC(self,i,bc):
        self.bc_v[i]=bc
    def set_rho_BC(self,i,bc):
        self.bc_rho[i]=bc
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
    def set_flow_BC_value(self,i,v):
        self.flow_BC[i]=v
    def set_flow_BCs_value(self,vs,unit = "lattice"):
        for i in range(4):
            self.set_flow_BC_value(i,vs[i])

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
        for specie in self.species:
            specie.set_BCs(i,BC)
    def set_species_BCs(self,BCs):# 定义所有物种的边界条件
        for specie in self.species:
            specie.set_BCs(BCs)
    def set_specie_BC_value(self,name,i,S):
        self.species[self.specieName.index(name)].set_s_BC_value(i,S)
    def set_specie_BCs_value(self,name,Ss):
        self.species[self.specieName.index(name)].set_s_BCs_value(Ss)
    def set_specie_BC_flux(self,name, i,f):
        self.species[self.specieName.index(name)].set_s_BC_flux(i,f)
    def set_specie_BCs_flux(self,name, fs):
        self.species[self.specieName.index(name)].set_s_BCs_flux(fs)
    def voxel_stl(self,stl_path,scale = 1.0,translate = [0,0,0],rotate = [0,0,0]):
        if not scale:
            scale = 1.0
        if not translate:
            translate = [0,0,0]
        if not rotate:
            rotate = [0,0,0]
        import pyvista as pv
        from scipy.spatial.transform import Rotation
        # 1. 加载STL
        mesh = pv.read(stl_path)
        mesh.scale(scale,inplace=True)
        mesh.rotate_z(rotate[2],inplace = True)
        mesh.rotate_y(rotate[1],inplace = True)
        mesh.rotate_x(rotate[0],inplace = True)
        # mesh.rotate(Rotation.from_euler('ZYX',rotate[::-1],degrees=True),inplace = True)
        mesh.translate(translate,inplace = True)
        # 2. 创建体素网格
        voxels = pv.DataSetFilters.voxelize(mesh) # 先变换再体素化，体素化之后再变换会使得网格错位，规则网格无法正确采样
        # voxels.scale(scale,inplace = True)
        # ugrid = pv.UnstructuredGrid()
        # ugrid.rotate_x()
        # 3. 转换为规则网格
        grid = pv.StructuredGrid(self.meshX,self.meshY,self.meshZ)
        
        # 4. 采样到规则网格
        sampled = grid.sample(voxels)
        
        # 5. 提取标量数据为数组
        voxel_array = np.array(sampled['vtkValidPointMask'].reshape(grid.dimensions, order='F'),dtype = float)
        return voxel_array,grid
    
    def load_cantera(self,file):
        """
        load_yaml 读取yaml机理文件 cantera格式
        
        :param file: file path
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
                    self.set_specie_mole(name,Fix = True,molemass=mmass)
                else:
                    self.set_specie_mole(name,Fix = False,molemass=mmass)
                self.set_specie_NASA7(name,specie_info["thermo"]["temperature-ranges"],specie_info["thermo"]["data"])
                
            for reaction_info in data["reactions"]:
                A = reaction_info["rate-constant"]["A"]
                Ea = reaction_info["rate-constant"]["Ea"]
                b = reaction_info["rate-constant"]["b"]
                self.add_reaction(reaction_info["equation"],A,Ea,b,unit = SPECIE_UNIT.MOLE,fixDH = False)
    
    
# 输出 可视化
@ti.data_oriented
class LBM2D_OUTPUT(LBM2D_BASE):
    def get_max_v(self): # 获得最大速度，用于判断模型是否发散
        self.max_v[None] = -1e10        
        self.cal_max_v()
        return self.max_v[None]
    def get_min_T(self):
        if self.TEMPERATURE:
            self.min_T[None]= 1e10
            self.cal_min_T()
            return self.TF.get_physical_value(self.min_T[None])
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
    def log_info(self):
        p = f"    t(LU)={self.tLattice} : Max velocity magnitude (LU) : {self.get_max_v():.7f}, "
        if self.TEMPERATURE:
            p +=f"Min temperature: {self.get_min_T():.7f} K"
        return p
    def export_snapshot(self,config):

        pass

    def export_VTK(self): # 导出为vtk 到指定文件夹中
        filename = os.path.join(self.exportPath,self.name+"_"+str(self.tLattice))
        gridToVTK(
                filename,
                self.x,
                self.y,
                self.z,
                # cellData={"pressure": pressure},
                pointData=self.get_data()
            )
        self.PVD.addVTK(self.tLattice*self.dt,os.path.basename(filename)+".vtr")
        self.PVD.writePVD()
    def get_data(self): # 获取所有数据（字典）
        data = {    "solid":self.solid.to_numpy(),
                    "rho": self.rho.to_numpy(),
                    "velocity": (
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,0]), 
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,1]), 
                        np.ascontiguousarray(self.v.to_numpy()[:,:,:,2]), 
                        ),
                }
        if self.PORO:
            # data["solid"]=self.solid.to_numpy()
            data["solid_init"]=self.rho1.to_numpy()
        if self.TEMPERATURE:
            data["Tf"]  = self.TF.get_physical_value(self.TF.S.to_numpy())
            data["Ts"]  = self.TS.get_physical_value(self.TS.S.to_numpy())
            if self.RADIATION:
                data["Radiation_surface"] = self.TS.radiation_surface.to_numpy()
                if self.TS.radiation_model == RADIATION_MODEL.REAL_RADIATION:
                    data["Real Radiation"] = self.TS.real_radiation.to_numpy()
        if self.CHEMISTRY:
            for specie in self.species:
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
    def check_python(self):
        self.check_kernel()
    @ti.kernel
    def check_kernel(self):
        self.check()
    @ti.func
    def check(self):
        s1 = [int(self.nx/2),int(self.ny/2),int(self.nz/2)]
        print(self.TF.S[s1], self.TF.physical_value(self.TF.S[s1]),self.TF.coefDiff(s1))
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
