import taichi as ti
from ..util.flag import *
from ._scalarField import ScalarField

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
        geqout=0.0
        geqout += self.LBM.w5[k]*T*(1.0 + 3.0*eu)
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
    @ti.func
    def geq5(self,k,T,x,y,z):
        geqout=0.0
        geqout += self.LBM.w5[k]*T
        return geqout
    