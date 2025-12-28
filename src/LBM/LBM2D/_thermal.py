from re import S
from sympy import EvaluationFailed
import taichi as ti
from ..util.flag import *
from ._scalarField import ScalarField

@ti.data_oriented
class TemperatureFluid(ScalarField):
    def __init__(self,name,lb2d,FIX = False):
        super().__init__(name,lb2d,FIX)
        self.thermal_diff_model = THERMAL_DIFF_MODEL.NONE
        self.Pr = 0.71
        self.thermal_diff = 3e-5
        self.conductivity_model = CONDUCTIVITY_MODEL.NONE
        self.cond = 1000
        self.cond_poly = [0,0,0,0]
        self.capacity_model = CAPACITY_MODEL.NONE
        self.cm = 1000
        self.cm_poly = [0,0,0,0,0]
        self.Trange = [0,0,0]
        self.NASA_coef = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    @ti.func
    def geq5(self,k,T,x,y,z):
        u = self.LBM.v[x,y,z]
        eu = self.LBM.e5[k].dot(u)
        uv = u.dot(u)
        geqout=0.0
        geqout += self.LBM.w5[k]*T*(1.0 + 3.0*eu+4.5*eu*eu-1.5*uv)
        return geqout
    @ti.func
    def coefDiff(self, i):
        D = 0.0
        if ti.static(self.thermal_diff_model==THERMAL_DIFF_MODEL.CONSTANT):
            D = self.thermal_diff*self.LBM.dt/self.LBM.dx**2
        elif ti.static(self.thermal_diff_model==THERMAL_DIFF_MODEL.PRANDTL):
            D = self.LBM.kinetic_viscosity(i)/self.Pr
        elif ti.static(self.thermal_diff_model==THERMAL_DIFF_MODEL.DERIVED):
            D = self.conductivity(i)/self.LBM.rho[i]/self.capacity_m(i)*self.LBM.dt/self.LBM.dx**2
        # if ti.static(self.LBM.CHEMISTRY):
        #     rhoc = 0.0
        #     k = 0.0
        #     for specie in ti.static(list(self.LBM.species)):
        #         if ti.static(not specie.FIX):
        #             k+=specie.S[i]*specie.conductivity(i)
        #             rhoc += specie.S[i]*self.LBM.rho[i]*specie.capacity_m(self.physical_value(self.S[i]))
        #     D += k*self.LBM.dt/rhoc/self.LBM.dx**2
        # else:
        #     D += self.default_coefDiff
        # if D == 0:
        #     D +=self.default_coefDiff
        return D    
    @ti.func
    def conductivity(self,i):
        kappa = 0.0
        if ti.static(self.conductivity_model==CONDUCTIVITY_MODEL.CONSTANT):
            kappa = self.cond
        elif ti.static(self.conductivity_model==CONDUCTIVITY_MODEL.POLYNOMIAL):
            T = self.physical_value(self.S[i])
            kappa = self.cond_poly[0]+T*(self.cond_poly[1]+T*(self.cond_poly[2]*T+T*(self.cond_poly[3]+T*self.cond_poly[4])))        
        elif ti.static(self.conductivity_model==CONDUCTIVITY_MODEL.MIXTURE):
            pass
        return kappa
    @ti.func
    def capacity_m(self,i): # 流体质量热容
        cm = 0.0
        if ti.static(self.capacity_model==CAPACITY_MODEL.CONSTANT):
            cm = self.cm
        elif ti.static(self.capacity_model==CAPACITY_MODEL.POLYNOMIAL):
            T = self.physical_value(self.S[i])
            cm = self.cm_poly[0]+T*(self.cm_poly[1]+T*(self.cm_poly[2]*T+T*(self.cm_poly[3]+T*self.cm_poly[4])))        
        elif ti.static(self.capacity_model==CAPACITY_MODEL.NASA7):
            T = self.physical_value(self.S[i])
            if T <self.Trange[1]:
                cm = self.NASA_coef[0][0]+T*(self.NASA_coef[0][1]+T*(self.NASA_coef[0][2]*T+T*(self.NASA_coef[0][3]+T*self.NASA_coef[0][4])))
            else:
                cm = self.NASA_coef[1][0]+T*(self.NASA_coef[1][1]+T*(self.NASA_coef[1][2]*T+T*(self.NASA_coef[1][3]+T*self.NASA_coef[1][4])))
        elif ti.static(self.capacity_model==CAPACITY_MODEL.MIXTURE):
            if ti.static(self.LBM.CHEMISTRY):
                for specie in ti.static(list(self.LBM.species)):
                    if ti.static(not specie.FIX):
                        cm += specie.capacity_m(self.physical_value(self.S[i]))*specie.S[i]*self.LBM.rho[i]
            else:
                cm += 100
        return cm

@ti.data_oriented
class TemperatureSolid(ScalarField):
    def __init__(self,name,lb2d,FIX = False,isRadiation = False):
        super().__init__(name,lb2d,FIX)
        self.rhoSolid = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.exchangeCoef = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.exchangeSurface = ti.field(float,shape=(self.nx,self.ny,self.nz))
        if isRadiation:
            self.radiation_model = RADIATION_MODEL.NONE # 辐射模型
            self.radiation_surface = ti.field(float,shape = (self.nx,self.ny,self.nz)) # S/V L-1
            self.emissivity = ti.field(float,shape=(self.nx,self.ny,self.nz))
        self.thermal_diff_model = THERMAL_DIFF_MODEL.DERIVED
        self.thermal_diff = 3e-5
        self.Pr = 0.71
        self.conductivity_model = CONDUCTIVITY_MODEL.MIXTURE
        self.cond = 0.2
        self.cond_poly = [0,0,0,0]
        self.capacity_model = CAPACITY_MODEL.MIXTURE
        self.cm = 1000
        self.cm_poly = [0,0,0,0,0]
        self.Trange = [0,0,0]
        self.NASA_coef = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    @ti.func
    def geq5(self,k,T,x,y,z):
        geqout = 0.0
        geqout += self.LBM.w5[k]*T
        return geqout
    @ti.func
    def coefDiff(self, i):
        D = 0.0
        if ti.static(self.thermal_diff_model==THERMAL_DIFF_MODEL.CONSTANT):
            D = self.thermal_diff*self.LBM.dt/self.LBM.dx**2
        elif ti.static(self.thermal_diff_model==THERMAL_DIFF_MODEL.DERIVED):
            D = self.conductivity(i)/self.LBM.rhos[i]/self.capacity_m(i)*self.LBM.dt/self.LBM.dx**2
        # if ti.static(self.LBM.CHEMISTRY):
        #     rhoc = 0.0
        #     k = 0.0
        #     for specie in ti.static(list(self.LBM.species)):
        #         if ti.static(not specie.FIX):
        #             k+=specie.S[i]*specie.conductivity(i)
        #             rhoc += specie.S[i]*self.LBM.rho[i]*specie.capacity_m(self.physical_value(self.S[i]))
        #     D += k*self.LBM.dt/rhoc/self.LBM.dx**2
        # else:
        #     D += self.default_coefDiff
        # if D == 0:
        #     D +=self.default_coefDiff
        return D    
    @ti.func
    def conductivity(self,i):
        kappa = 0.0
        if ti.static(self.conductivity_model==CONDUCTIVITY_MODEL.CONSTANT):
            kappa = self.cond
        elif ti.static(self.conductivity_model==CONDUCTIVITY_MODEL.POLYNOMIAL):
            T = self.physical_value(self.S[i])
            kappa = self.cond_poly[0]+T*(self.cond_poly[1]+T*(self.cond_poly[2]*T+T*(self.cond_poly[3])))        
        elif ti.static(self.conductivity_model==CONDUCTIVITY_MODEL.MIXTURE):
            pass
        return kappa
    @ti.func
    def capacity_m(self,i): # 质量热容
        cm = 0.0
        if ti.static(self.capacity_model==CAPACITY_MODEL.CONSTANT):
            cm = self.cm
        elif ti.static(self.capacity_model==CAPACITY_MODEL.POLYNOMIAL):
            T = self.physical_value(self.S[i])
            cm = self.cm_poly[0]+T*(self.cm_poly[1]+T*(self.cm_poly[2]*T+T*(self.cm_poly[3]+T*self.cm_poly[4])))        
        elif ti.static(self.capacity_model==CAPACITY_MODEL.NASA7):
            T = self.physical_value(self.S[i])
            if T <self.Trange[1]:
                cm = self.NASA_coef[0][0]+T*(self.NASA_coef[0][1]+T*(self.NASA_coef[0][2]*T+T*(self.NASA_coef[0][3]+T*self.NASA_coef[0][4])))
            else:
                cm = self.NASA_coef[1][0]+T*(self.NASA_coef[1][1]+T*(self.NASA_coef[1][2]*T+T*(self.NASA_coef[1][3]+T*self.NASA_coef[1][4])))
        elif ti.static(self.capacity_model==CAPACITY_MODEL.MIXTURE):
            if ti.static(self.LBM.CHEMISTRY):
                for specie in ti.static(list(self.LBM.species)):
                    if ti.static(not specie.FIX):
                        cm += specie.capacity_m(self.physical_value(self.S[i]))*specie.S[i]*self.LBM.rho[i]
            else:
                cm += 1000
        return cm

    
    @ti.func
    def radiation(self,i):# Wm-2K-4*m-1K4*{m2}=Wm-3 SI # 单位体积辐射 SI
        q = 0.0
        if ti.static(self.radiation_model==RADIATION_MODEL.SURFACE_UNIFORM):
            q += self.emissivity[i]*self.SIGMA*self.radiation_surface[i]/self.LBM.dx*(ti.pow(self.Tambient,4)-ti.pow(self.physical_value(self.S[i]),4))
        elif ti.static(self.radiation_model==RADIATION_MODEL.REAL_RADIATION):
            q += self.real_radiation[i]-self.SIGMA*self.radiation_surface[i]/self.LBM.dx*ti.pow(self.physical_value(self.S[i]),4)
        return q
    