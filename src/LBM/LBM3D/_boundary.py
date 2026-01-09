import taichi as ti
from ._core import LBM3D_BASE

from ._chemical import Specie,Reaction
from ._thermal import TemperatureFluid,TemperatureSolid
from ..util.flag import *
@ti.data_oriented
class LBM3D_BOUNDARY(LBM3D_BASE):
    """
    边界条件实现的时候还需要考虑外力的影响
    """
    def updateBC(self, t):
        for func in self.UpdateBCfunc:
            func(self,t)
    @ti.kernel
    def Boundary_condition_NEE(self):
        for j,k in ti.ndrange(self.ny,self.nz):            
            self.Boundary_condition_flow_NEE_0(0,j,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_0(0,j,k)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_0(0,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_0(0,j,k)
            self.Boundary_condition_flow_NEE_1(self.nx-1,j,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_1(self.nx-1,j,k)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_1(self.nx-1,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_1(self.nx-1,j,k)
        for i,k in ti.ndrange(self.nx,self.nz):
            self.Boundary_condition_flow_NEE_2(i,0,k)
            if ti.static(self.CHEMISTRY):
               for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_2(i,0,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_2(i,0,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_2(i,0,k)
            self.Boundary_condition_flow_NEE_3(i,self.ny-1,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_3(i,self.ny-1,k)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_3(i,self.ny-1,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_3(i,self.ny-1,k)
        for i,j in ti.ndrange(self.nx,self.ny):
            self.Boundary_condition_flow_NEE_4(i,j,0)
            if ti.static(self.CHEMISTRY):
               for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_4(i,j,0)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_4(i,j,0) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_4(i,j,0)
            self.Boundary_condition_flow_NEE_5(i,j,self.nz-1)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_5(i,j,self.nz-1)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_5(i,j,self.nz-1) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_5(i,j,self.nz-1)
            

    @ti.func
    def Boundary_condition_flow_NEE_0(self,x,y,z):
        if ti.static(self.bc_v[0]==BC.fixedValue):
            self.v[0,y,z] = self.v_bc_profile[0][0,y,z]  
        elif ti.static(self.bc_v[0]==BC.zeroGradient):
            self.v[0,y,z] = self.v[1,y,z]
        if ti.static(self.bc_rho[0]==BC.fixedValue):
            self.rho[0,y,z] = self.rho_BC[0]
        elif ti.static(self.bc_rho[0]==BC.zeroGradient):
            self.rho[0,y,z] = self.rho[1,y,z]# 2*self.rho[1,y,z]-self.rho[2,y,z]
        for s in ti.static(range(19)):
            self.f[0,y,z][s] = self.feq19(s,0,y,z)+(self.f[1,y,z][s]-self.feq19(s,1,y,z))
    @ti.func
    def Boundary_condition_flow_NEE_1(self,x,y,z):
        if ti.static(self.bc_v[1]==BC.fixedValue):
            self.v[self.nx-1,y,z] = self.v_bc_profile[1][0,y,z]
        elif ti.static(self.bc_v[1]==BC.zeroGradient):
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
        if ti.static(self.bc_rho[1]==BC.fixedValue):
            self.rho[self.nx-1,y,z] = self.rho_BC[1]
        elif ti.static(self.bc_rho[1]==BC.zeroGradient):
            self.rho[self.nx-1,y,z] = self.rho[self.nx-2,y,z]
        for s in ti.static(range(19)):
            self.f[self.nx-1,y,z][s] = self.feq19(s,self.nx-1,y,z)+(self.f[self.nx-2,y,z][s]-self.feq19(s,self.nx-2,y,z))
    @ti.func
    def Boundary_condition_flow_NEE_2(self,x,y,z):
        if ti.static(self.bc_v[2]==BC.fixedValue):
            self.v[x,0,z] = self.v_bc_profile[2][x,0,z]
        elif ti.static(self.bc_v[2]==BC.zeroGradient):
            self.v[x,0,z] = self.v[x,1,z]
        if ti.static(self.bc_rho[2]==BC.fixedValue):
            self.rho[x,0,z] = self.rho_BC[2]
        elif ti.static(self.bc_rho[2]==BC.zeroGradient):
            self.rho[x,0,z] = self.rho[x,1,z]
        for s in ti.static(range(19)):
            self.f[x,0,z][s] = self.feq19(s,x,0,z)+(self.f[x,1,z][s]-self.feq19(s,x,1,z))
    @ti.func
    def Boundary_condition_flow_NEE_3(self,x,y,z):
        if ti.static(self.bc_v[3]==BC.fixedValue):
            self.v[x,self.ny-1,z] = self.v_bc_profile[3][x,0,z]
        elif ti.static(self.bc_v[3]==BC.zeroGradient):
            self.v[x,self.ny-1,z] = self.v[x,self.ny-2,z]
        if ti.static(self.bc_rho[3]==BC.fixedValue):
            self.rho[x,self.ny-1,z] = self.rho_BC[3]
        elif ti.static(self.bc_rho[3]==BC.zeroGradient):
            self.rho[x,self.ny-1,z] = self.rho[x,self.ny-2,z]
        for s in ti.static(range(19)):
            self.f[x,self.ny-1,z][s] = self.feq19(s,x,self.ny-1,z)+(self.f[x,self.ny-2,z][s]-self.feq19(s,x,self.ny-2,z))
    @ti.func
    def Boundary_condition_flow_NEE_4(self,x,y,z):
        if ti.static(self.bc_v[4]==BC.fixedValue):
            self.v[x,y,0] = self.v_bc_profile[4][x,y,0]
        elif ti.static(self.bc_v[4]==BC.zeroGradient):
            self.v[x,y,0] = self.v[x,y,1]
        if ti.static(self.bc_rho[4]==BC.fixedValue):
            self.rho[x,y,0] = self.rho_BC[4]
        elif ti.static(self.bc_rho[4]==BC.zeroGradient):
            self.rho[x,y,0] = self.rho[x,y,0]
        for s in ti.static(range(19)):
            self.f[x,y,0][s] = self.feq19(s,x,y,0)+(self.f[x,y,1][s]-self.feq19(s,x,y,1))
    @ti.func
    def Boundary_condition_flow_NEE_5(self,x,y,z):
        if ti.static(self.bc_v[5]==BC.fixedValue):
            self.v[x,y,self.nz-1] = self.v_bc_profile[5][x,y,0]
        elif ti.static(self.bc_v[5]==BC.zeroGradient):
            self.v[x,y,self.nz-1] = self.v[x,y,self.nz-2]
        if ti.static(self.bc_rho[5]==BC.fixedValue):
            self.rho[x,y,self.nz-1] = self.rho_BC[5]
        elif ti.static(self.bc_rho[5]==BC.zeroGradient):
            self.rho[x,y,self.nz-1] = self.rho[x,y,self.nz-2]
        for s in ti.static(range(19)):
            self.f[x,y,self.nz-1][s] = self.feq19(s,x,y,self.nz-1)+(self.f[x,y,self.nz-2][s]-self.feq19(s,x,y,self.nz-2))
    """equilibrium"""
    @ti.func
    def Boundary_condition_ES(self):
        for j,k in ti.ndrange(self.ny,self.nz):            
            self.Boundary_condition_flow_ES_0(0,j,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_0(0,j,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_0(0,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_0(0,j,k)
            self.Boundary_condition_flow_ES_1(self.nx-1,j,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_1(self.nx-1,j,k)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_1(self.nx-1,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_1(self.nx-1,j,k)
        for i,k in ti.ndrange(self.nx,self.nz):
            self.Boundary_condition_flow_ES_2(i,0,k)
            if ti.static(self.CHEMISTRY):
               for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_2(i,0,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_2(i,0,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_2(i,0,k)
            self.Boundary_condition_flow_ES_3(i,self.ny-1,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_3(i,self.ny-1,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_3(i,self.ny-1,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_3(i,self.ny-1,k)
        for i,j in ti.ndrange(self.nx,self.ny):
            self.Boundary_condition_flow_ES_4(i,j,0)
            if ti.static(self.CHEMISTRY):
               for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_4(i,j,0)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_4(i,j,0) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_4(i,j,0)
            self.Boundary_condition_flow_ES_5(i,j,self.nz-1)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_5(i,j,self.nz-1)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_5(i,j,self.nz-1) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_5(i,j,self.nz-1)
    @ti.func
    def Boundary_condition_flow_ES_0(self,x,y,z):
        if ti.static(self.bc_v[0]==BC.fixedValue):
            self.v[0,y,z] = self.v_BC[0]
        elif ti.static(self.bc_v[0]==BC.zeroGradient):
            self.v[0,y,z] = self.v[1,y,z]
        if ti.static(self.bc_rho[0]==BC.fixedValue):
            self.rho[0,y,z] = self.rho_BC[0]
        elif ti.static(self.bc_rho[0]==BC.zeroGradient):
            self.rho[0,y,z] = self.rho[1,y,z]
        for s in ti.static(range(19)):
            self.f[0,y,z][s] = self.feq19(s,0,y,z)
    @ti.func
    def Boundary_condition_flow_ES_1(self,x,y,z):
        if ti.static(self.bc_v[1]==BC.fixedValue):
            self.v[self.nx-1,y,z] = self.v_BC[1]
        elif ti.static(self.bc_v[1]==BC.zeroGradient):
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
        if ti.static(self.bc_rho[1]==BC.fixedValue):
            self.rho[self.nx-1,y,z] = self.rho_BC[1]
        elif ti.static(self.bc_rho[1]==BC.zeroGradient):
            self.rho[self.nx-1,y,z] = self.rho[self.nx-2,y,z]
        for s in ti.static(range(19)):
            self.f[self.nx-1,y,z][s] = self.feq19(s,self.nx-1,y,z)
    @ti.func
    def Boundary_condition_flow_ES_2(self,x,y,z):
        if ti.static(self.bc_v[2]==BC.fixedValue):
            self.v[x,0,z] = self.v_BC[2]
        elif ti.static(self.bc_v[2]==BC.zeroGradient):
            self.v[x,0,z] = self.v[x,1,z]
        if ti.static(self.bc_rho[2]==BC.fixedValue):
            self.rho[x,0,z] = self.rho_BC[2]
        elif ti.static(self.bc_rho[2]==BC.zeroGradient):
            self.rho[x,0,z] = self.rho[x,1,z]
        for s in ti.static(range(19)):
            self.f[x,0,z][s] = self.feq19(s,x,0,z)
    @ti.func
    def Boundary_condition_flow_ES_3(self,x,y,z):
        if ti.static(self.bc_v[3]==BC.fixedValue):
            self.v[x,self.ny-1,z] = self.v_BC[3]
        elif ti.static(self.bc_v[3]==BC.zeroGradient):
            self.v[x,self.ny-1,z] = self.v[x,self.ny-2,z]
        if ti.static(self.bc_rho[3]==BC.fixedValue):
            self.rho[x,self.ny-1,z] = self.rho_BC[3]
        elif ti.static(self.bc_rho[3]==BC.zeroGradient):
            self.rho[x,self.ny-1,z] = self.rho[x,self.ny-2,z]
        for s in ti.static(range(19)):
            self.f[x,self.ny-1,z][s] = self.feq19(s,x,self.ny-1,z)
    @ti.func
    def Boundary_condition_flow_ES_4(self,x,y,z):
        if ti.static(self.bc_v[4]==BC.fixedValue):
            self.v[x,0,z] = self.v_BC[4]
        elif ti.static(self.bc_v[4]==BC.zeroGradient):
            self.v[x,0,z] = self.v[x,1,z]
        if ti.static(self.bc_rho[4]==BC.fixedValue):
            self.rho[x,0,z] = self.rho_BC[4]
        elif ti.static(self.bc_rho[4]==BC.zeroGradient):
            self.rho[x,0,z] = self.rho[x,1,z]
        for s in ti.static(range(19)):
            self.f[x,0,z][s] = self.feq19(s,x,0,z)
    @ti.func
    def Boundary_condition_flow_ES_5(self,x,y,z):
        if ti.static(self.bc_v[5]==BC.fixedValue):
            self.v[x,self.ny-1,z] = self.v_BC[5]
        elif ti.static(self.bc_v[5]==BC.zeroGradient):
            self.v[x,self.ny-1,z] = self.v[x,self.ny-2,z]
        if ti.static(self.bc_rho[5]==BC.fixedValue):
            self.rho[x,self.ny-1,z] = self.rho_BC[5]
        elif ti.static(self.bc_rho[5]==BC.zeroGradient):
            self.rho[x,self.ny-1,z] = self.rho[x,self.ny-2,z]
        for s in ti.static(range(19)):
            self.f[x,self.ny-1,z][s] = self.feq19(s,x,self.ny-1,z)
    