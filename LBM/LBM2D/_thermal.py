from re import S
import taichi as ti
from ..util.flag import *
from ._scalarField import ScalarField

@ti.data_oriented
class TemperatureFluid(ScalarField):
    def __init__(self,name,nx,ny,nz,lb2d,FIX = False):
        super().__init__(name,nx,ny,nz,lb2d,FIX)
        self.default_cv = 1000
        self.default_coefDiff = 0.1
    @ti.func
    def capacity_v(self,i): # 流体的体积热容 SI
        cv = 0.0
        if ti.static(self.LBM.CHEMISTRY):
            for specie in ti.static(list(self.LBM.species.values())):
                if ti.static(not specie.FIX):
                    cv += specie.capacity_m(i)*specie.S[i]*self.LBM.rho[i]
        else:
            cv += self.default_cv
        if cv == 0.0:
           cv += self.default_cv
        return cv
    @ti.func
    def coefDiff(self, i):
        D = 0.0
        if ti.static(self.LBM.CHEMISTRY):
            rhoc = 0.0
            k = 0.0
            for specie in ti.static(list(self.LBM.species.values())):
                if ti.static(not specie.FIX):
                    k+=specie.S[i]*specie.conductivity(i)
                    rhoc += specie.S[i]*self.LBM.rho[i]*specie.capacity_m(i)
            D += k*self.LBM.dt/rhoc/self.LBM.dx**2
        else:
            D += self.default_coefDiff
        if D == 0:
            D +=self.default_coefDiff
        return D    
    @ti.func
    def geq5(self,k,T,x,y,z):
        u = self.LBM.v[x,y,z]
        eu = self.LBM.e5[k].dot(u)
        uv = u.dot(u)
        geqout=0.0
        geqout += self.LBM.w5[k]*T*(1.0 + 3.0*eu+4.5*eu*eu-1.5*uv)
        return geqout

@ti.data_oriented
class TemperatureSolid(ScalarField):
    SIGMA = 5.67e-8
    def __init__(self,name,nx,ny,nz,lb2d,FIX = False,isRadiation = False):
        super().__init__(name,nx,ny,nz,lb2d,FIX)
        self.exchangeCoef = ti.field(ti.f32,shape=(nx,ny,nz))
        self.exchangeSurface = ti.field(ti.f32,shape=(nx,ny,nz))
        self.default_cv = 100000.0
        self.default_coefDiff = 0.01
        if isRadiation:
            self.radiation_model = RADIATION_MODEL.NONE # 辐射模型
            self.radiation_surface = ti.field(ti.f32,shape = (self.nx,self.ny,self.nz)) # S/V L-1
    @ti.func
    def coefDiff(self, i):
        D = 0.0
        if self.LBM.solid==0:
            D += 10000
        elif ti.static(self.LBM.CHEMISTRY):
            rhoc = 0.0
            rho = 0.0
            k = 0.0
            for specie in ti.static(list(self.LBM.species.values())):
                if ti.static(specie.FIX):
                    k+=specie.S[i]*specie.conductivity(i)
                    rho += specie.S[i]
                    rhoc += specie.S[i]*specie.capacity_m(i)
            if rhoc != 0 and rho!=0:
                D += k*self.LBM.dt/rhoc/rho/self.LBM.dx**2
            else:
                D += self.default_coefDiff
    
        else:
            D += self.default_coefDiff
        if D == 0.0:
            D += self.default_coefDiff
        return D    

    @ti.func
    def capacity_v(self,i): # 所有物质的体积热容 按照孔隙率加权。流体密度为纯流体的密度 而固体密度为考虑孔隙率的密度 因此固体密度不需要再用孔隙率修正
        c = 0.0
        if ti.static(self.LBM.CHEMISTRY):
           for specie in ti.static(list(self.LBM.species.values())):
                if ti.static(specie.FIX):
                    c += specie.S[i]*specie.capacity_m(i)
        else:
            c += self.default_cv
        if c == 0.0:
           c += self.default_cv
        return  c # kgm-3Jkg-1K-1=Jm-3K-1 SI
    @ti.func
    def geq5(self,k,T,x,y,z):
        geqout = 0.0
        geqout += self.LBM.w5[k]*T
        return geqout
    
    @ti.func
    def radiation(self,i):# Wm-2K-4*m-1K4*{m2}=Wm-3 SI # 单位体积辐射 SI
        q = 0.0
        if ti.static(self.radiation_model==RADIATION_MODEL.SURFACE_UNIFORM):
            q += self.SIGMA*self.radiation_surface[i]/self.LBM.dx*(ti.pow(self.Tambient,4)-ti.pow(self.S[i],4))
        elif ti.static(self.radiation_model==RADIATION_MODEL.REAL_RADIATION):
            q += self.real_radiation[i]-self.SIGMA*self.radiation_surface[i]/self.LBM.dx*ti.pow(self.S[i],4)
        return q
    