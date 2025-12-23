from re import escape
import taichi as ti
from ._core import STLBM2D_BASE

from ._chemical import Specie,Reaction
from ._thermal import TemperatureFluid,TemperatureSolid
from ..util.flag import *
@ti.data_oriented
class STLBM2D_BOUNDARY(STLBM2D_BASE):
    """
    边界条件实现的时候还需要考虑外力的影响
    """
    @ti.func
    def Boundary_condition(self):
        for j,k in ti.ndrange(self.ny,self.nz):            
            self.Boundary_condition_flow_0(0,j,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_0(0,j,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_0(0,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_0(0,j,k)
            self.Boundary_condition_flow_1(self.nx-1,j,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_1(self.nx-1,j,k)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_1(self.nx-1,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_1(self.nx-1,j,k)
        for i,k in ti.ndrange(self.nx,self.nz):
            self.Boundary_condition_flow_2(i,0,k)
            if ti.static(self.CHEMISTRY):
               for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_2(i,0,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_2(i,0,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_2(i,0,k)
            self.Boundary_condition_flow_3(i,self.ny-1,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_3(i,self.ny-1,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_3(i,self.ny-1,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_3(i,self.ny-1,k)

    @ti.func
    def Boundary_condition_flow_0(self,x,y,z):
        if ti.static(self.bc_v[0]==BC.fixedValue):
            self.v[0,y,z] = self.v_BC[0]
        elif ti.static(self.bc_v[0]==BC.zeroGradient):
            self.v[0,y,z] = self.v[1,y,z]
        if ti.static(self.bc_rho[0]==BC.fixedValue):
            self.rho[0,y,z] = self.rho_BC[0]
        elif ti.static(self.bc_rho[0]==BC.zeroGradient):
            self.rho[0,y,z] = self.rho[1,y,z]
    @ti.func
    def Boundary_condition_flow_1(self,x,y,z):        
        if ti.static(self.bc_v[1]==BC.fixedValue):
            self.v[self.nx-1,y,z] = self.v_BC[1]
        elif ti.static(self.bc_v[1]==BC.zeroGradient):
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
        if ti.static(self.bc_rho[1]==BC.fixedValue):
            self.rho[self.nx-1,y,z] = self.rho_BC[1]
        elif ti.static(self.bc_rho[1]==BC.zeroGradient):
            self.rho[self.nx-1,y,z] = self.rho[self.nx-2,y,z]
    @ti.func
    def Boundary_condition_flow_2(self,x,y,z):
        if ti.static(self.bc_v[2]==BC.fixedValue):
            self.v[x,0,z] = self.v_BC[2]
        elif ti.static(self.bc_v[2]==BC.zeroGradient):
            self.v[x,0,z] = self.v[x,1,z]
        if ti.static(self.bc_rho[2]==BC.fixedValue):
            self.rho[x,0,z] = self.rho_BC[2]
        elif ti.static(self.bc_rho[2]==BC.zeroGradient):
            self.rho[x,0,z] = self.rho[x,1,z]
    @ti.func
    def Boundary_condition_flow_3(self,x,y,z):
        if ti.static(self.bc_v[3]==BC.fixedValue):
            self.v[x,self.ny-1,z] = self.v_BC[3]
        elif ti.static(self.bc_v[3]==BC.zeroGradient):
            self.v[x,self.ny-1,z] = self.v[x,self.ny-2,z]
        if ti.static(self.bc_rho[3]==BC.fixedValue):
            self.rho[x,self.ny-1,z] = self.rho_BC[3]
        elif ti.static(self.bc_rho[3]==BC.zeroGradient):
            self.rho[x,self.ny-1,z] = self.rho[x,self.ny-2,z]
