from re import escape
import taichi as ti
from ._core import LBM2D_BASE

from ._chemical import Specie,Reaction
from ._thermal import TemperatureFluid,TemperatureSolid
from ..util.flag import *
@ti.data_oriented
class LBM2D_BOUNDARY(LBM2D_BASE):
    """
    边界条件实现的时候还需要考虑外力的影响
    """
    @ti.func
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

    @ti.func
    def Boundary_condition_flow_NEE_0(self,x,y,z):
        if ti.static(self.bc_v[0]==BC.fixedValue):
            self.v[0,y,z] = self.v_bc_profile[0][0,y,z]
        elif ti.static(self.bc_v[0]==BC.zeroGradient):
            self.v[0,y,z] = 2*self.v[1,y,z]-self.v[2,y,z]
        if ti.static(self.bc_rho[0]==BC.fixedValue):
            self.rho[0,y,z] = self.rho_bc_profile[0][0,y,z]
        elif ti.static(self.bc_rho[0]==BC.zeroGradient):
            self.rho[0,y,z] = 2*self.rho[1,y,z]-self.rho[2,y,z]
        for s in ti.static(range(9)):
            self.f[0,y,z][s] = self.feq9(s,0,y,z)+(self.f[1,y,z][s]-self.feq9(s,1,y,z))
    @ti.func
    def Boundary_condition_flow_NEE_1(self,x,y,z):
        if ti.static(self.bc_v[1]==BC.fixedValue):
            self.v[self.nx-1,y,z] = self.v_BC[1]
        elif ti.static(self.bc_v[1]==BC.zeroGradient):
            self.v[self.nx-1,y,z] = 2*self.v[self.nx-2,y,z]-self.v[self.nx-3,y,z]
        if ti.static(self.bc_rho[1]==BC.fixedValue):
            self.rho[self.nx-1,y,z] = self.rho_BC[1]
        elif ti.static(self.bc_rho[1]==BC.zeroGradient):
            self.rho[self.nx-1,y,z] = 2*self.rho[self.nx-2,y,z]-self.rho[self.nx-3,y,z]
        for s in ti.static(range(9)):
            self.f[self.nx-1,y,z][s] = self.feq9(s,self.nx-1,y,z)+(self.f[self.nx-2,y,z][s]-self.feq9(s,self.nx-2,y,z))
    @ti.func
    def Boundary_condition_flow_NEE_2(self,x,y,z):
        if ti.static(self.bc_v[2]==BC.fixedValue):
            self.v[x,0,z] = self.v_BC[2]
        elif ti.static(self.bc_v[2]==BC.zeroGradient):
            self.v[x,0,z] = 2*self.v[x,1,z]-self.v[x,2,z]
        if ti.static(self.bc_rho[2]==BC.fixedValue):
            self.rho[x,0,z] = self.rho_BC[2]
        elif ti.static(self.bc_rho[2]==BC.zeroGradient):
            self.rho[x,0,z] = 2*self.rho[x,1,z]-self.rho[x,2,z]
        for s in ti.static(range(9)):
            self.f[x,0,z][s] = self.feq9(s,x,0,z)+(self.f[x,1,z][s]-self.feq9(s,x,1,z))
    @ti.func
    def Boundary_condition_flow_NEE_3(self,x,y,z):
        if ti.static(self.bc_v[3]==BC.fixedValue):
            self.v[x,self.ny-1,z] = self.v_BC[3]
        elif ti.static(self.bc_v[3]==BC.zeroGradient):
            self.v[x,self.ny-1,z] = 2*self.v[x,self.ny-2,z]-self.v[x,self.ny-3,z]
        if ti.static(self.bc_rho[3]==BC.fixedValue):
            self.rho[x,self.ny-1,z] = self.rho_BC[3]
        elif ti.static(self.bc_rho[3]==BC.zeroGradient):
            self.rho[x,self.ny-1,z] = 2*self.rho[x,self.ny-2,z]-self.rho[x,self.ny-3,z]
        for s in ti.static(range(9)):
            self.f[x,self.ny-1,z][s] = self.feq9(s,x,self.ny-1,z)+(self.f[x,self.ny-2,z][s]-self.feq9(s,x,self.ny-2,z))
    """NEBB"""
    @ti.func
    def Boundary_condition_NEBB(self):
        for j,k in ti.ndrange(self.ny-2,self.nz):     
            j = j+1       
            self.Boundary_condition_flow_NEBB_0(0,j,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_0(0,j,k)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_0(0,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_0(0,j,k)
            self.Boundary_condition_flow_NEBB_1(self.nx-1,j,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_1(self.nx-1,j,k)
            if ti.static(self.TEMPERATURE):
                self.TF.Boundary_condition_scalar_1(self.nx-1,j,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_1(self.nx-1,j,k)
        for i,k in ti.ndrange(self.nx,self.nz):
            # i = i+1
            self.Boundary_condition_flow_NEBB_2(i,0,k)
            if ti.static(self.CHEMISTRY):
               for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_2(i,0,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_2(i,0,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_2(i,0,k)
            self.Boundary_condition_flow_NEBB_3(i,self.ny-1,k)
            if ti.static(self.CHEMISTRY):
                for specie in ti.static(list(self.species)):
                    if ti.static(not specie.FIX):
                        specie.Boundary_condition_scalar_3(i,self.ny-1,k)
            if ti.static(self.TEMPERATURE):
                # print("boundary IE")
                self.TF.Boundary_condition_scalar_3(i,self.ny-1,k) # note: 后更新焓的边界条件，使用到的热容计算需要边界区域的物质浓度
                self.TS.Boundary_condition_scalar_3(i,self.ny-1,k)

    @ti.func
    def Boundary_condition_flow_NEBB_0(self,x,y,z): # inlet 1 5 8 
        if ti.static(self.bc[0]==BC_FLOW.wall):
            rho = (self.F[0,y,z][0]+self.F[0,y,z][2]+self.F[0,y,z][4]+2*(self.F[0,y,z][3]+self.F[0,y,z][6]+self.F[0,y,z][7]))
            self.F[0,y,z][1] = self.F[0,y,z][3]
            self.F[0,y,z][5] = self.F[0,y,z][7] + 0.5*(self.F[0,y,z][4]-self.F[0,y,z][2])
            self.F[0,y,z][8] = self.F[0,y,z][6] + 0.5*(self.F[0,y,z][2]-self.F[0,y,z][4])
            self.rho[0,y,z] = rho
            self.v[0,y,z] = ti.Vector([0.0,0.0,0.0])
        if ti.static(self.bc[0]==BC_FLOW.inlet):
            u = self.v_BC[0][0]
            v = self.v_BC[0][1]
            rho = (self.F[0,y,z][0]+self.F[0,y,z][2]+self.F[0,y,z][4]+2*(self.F[0,y,z][3]+self.F[0,y,z][6]+self.F[0,y,z][7]))/(1-u)
            self.F[0,y,z][1] = self.F[0,y,z][3] + 2/3*rho*u
            self.F[0,y,z][5] = self.F[0,y,z][7] + 0.5*(self.F[0,y,z][4]-self.F[0,y,z][2]) + 1/6*rho*u+0.5*rho*v
            self.F[0,y,z][8] = self.F[0,y,z][6] + 0.5*(self.F[0,y,z][2]-self.F[0,y,z][4]) + 1/6*rho*u-0.5*rho*v
            self.rho[0,y,z] = rho
            self.v[0,y,z] = self.v_BC[1]
        if ti.static(self.bc[0]==BC_FLOW.outlet):
            rho = self.rho_BC[0]
            v = 0.0
            u = -1+(2*(self.F[0,y,z][3]+self.F[0,y,z][6]+self.F[0,y,z][7])\
                    +self.F[0,y,z][0]+self.F[0,y,z][2]+self.F[0,y,z][4])/rho
            self.rho[0,y,z] = rho
            self.v[0,y,z] = ti.Vector([u,v,0.0])
            self.F[0,y,z][1] = self.F[0,y,z][3] + 2/3*rho*u
            self.F[0,y,z][5] = self.F[0,y,z][7] + 0.5*(self.F[0,y,z][4]-self.F[0,y,z][2]) + 1/6*rho*u
            self.F[0,y,z][8] = self.F[0,y,z][6] + 0.5*(self.F[0,y,z][2]-self.F[0,y,z][4]) + 1/6*rho*u
    @ti.func
    def Boundary_condition_flow_NEBB_1(self,x,y,z):# 3 6 7 outlet
        if ti.static(self.bc[1]==BC_FLOW.wall):
            rho = (self.F[self.nx-1,y,z][0]+self.F[self.nx-1,y,z][2]+self.F[self.nx-1,y,z][4]+2*(self.F[self.nx-1,y,z][3]+self.F[self.nx-1,y,z][6]+self.F[self.nx-1,y,z][7]))
            self.F[self.nx-1,y,z][1] = self.F[self.nx-1,y,z][3]
            self.F[self.nx-1,y,z][5] = self.F[self.nx-1,y,z][7] + 0.5*(self.F[self.nx-1,y,z][4]-self.F[self.nx-1,y,z][2])
            self.F[self.nx-1,y,z][8] = self.F[self.nx-1,y,z][6] + 0.5*(self.F[self.nx-1,y,z][2]-self.F[self.nx-1,y,z][4])
            self.rho[self.nx-1,y,z] = rho
            self.v[self.nx-1,y,z] = ti.Vector([0.0,0.0,0.0])
        if ti.static(self.bc[1]==BC_FLOW.inlet):
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
        if ti.static(self.bc[1]==BC_FLOW.outlet):
            rho = self.rho_BC[1]
            v = 0.0
            u = -1+(2*(self.F[self.nx-1,y,z][1]+self.F[self.nx-1,y,z][5]+self.F[self.nx-1,y,z][8])\
                    +self.F[self.nx-1,y,z][0]+self.F[self.nx-1,y,z][2]+self.F[self.nx-1,y,z][4])/rho
            self.rho[self.nx-1,y,z] = rho
            self.v[self.nx-1,y,z] = ti.Vector([u,v,0.0])
            self.F[self.nx-1,y,z][3] = self.F[self.nx-1,y,z][1] - 2/3*rho*u
            self.F[self.nx-1,y,z][6] = self.F[self.nx-1,y,z][8] + 0.5*(self.F[self.nx-1,y,z][4]-self.F[self.nx-1,y,z][2]) - 1/6*rho*u
            self.F[self.nx-1,y,z][7] = self.F[self.nx-1,y,z][5] + 0.5*(self.F[self.nx-1,y,z][2]-self.F[self.nx-1,y,z][4]) - 1/6*rho*u
    @ti.func
    def Boundary_condition_flow_NEBB_2(self,x,y,z): # wall 2 5 6
        if ti.static(self.bc[2]==BC_FLOW.wall):
            rho = (self.F[x,0,z][0]+self.F[x,0,z][1]+self.F[x,0,z][3]+2*(self.F[x,0,z][4]+self.F[x,0,z][7]+self.F[x,0,z][8]))
            self.F[x,0,z][2] = self.F[x,0,z][4]
            self.F[x,0,z][5] = self.F[x,0,z][7] + 0.5*(self.F[x,0,z][3]-self.F[x,0,z][1])
            self.F[x,0,z][6] = self.F[x,0,z][8] + 0.5*(self.F[x,0,z][1]-self.F[x,0,z][3])
            self.rho[x,0,z] = rho
            self.v[x,0,z] = ti.Vector([0.0,0.0,0.0])
        if ti.static(self.bc[2]==BC_FLOW.inlet):
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
        if ti.static(self.bc[2]==BC_FLOW.outlet):
            rho = self.rho_BC[2]
            v = 0.0
            u = -1+(2*(self.F[self.nx-1,y,z][1]+self.F[self.nx-1,y,z][5]+self.F[self.nx-1,y,z][8])+self.F[self.nx-1,y,z][0]+self.F[self.nx-1,y,z][2]+self.F[self.nx-1,y,z][4])/rho
            self.rho[self.nx-1,y,z] = rho
            self.v[self.nx-1,y,z] = ti.Vector([u,v,0.0])    
    @ti.func
    def Boundary_condition_flow_NEBB_3(self,x,y,z): # wall
        if ti.static(self.bc[3]==BC_FLOW.wall):
            rho = (self.F[x,self.ny-1,z][0]+self.F[x,self.ny-1,z][1]+self.F[x,self.ny-1,z][3]+2*(self.F[x,self.ny-1,z][2]+self.F[x,self.ny-1,z][5]+self.F[x,self.ny-1,z][6]))
            self.F[x,self.ny-1,z][4] = self.F[x,self.ny-1,z][2]
            self.F[x,self.ny-1,z][7] = self.F[x,self.ny-1,z][5] + 0.5*(self.F[x,self.ny-1,z][1]-self.F[x,self.ny-1,z][3])
            self.F[x,self.ny-1,z][8] = self.F[x,self.ny-1,z][6] + 0.5*(self.F[x,self.ny-1,z][3]-self.F[x,self.ny-1,z][1])
            self.rho[x,self.ny-1,z] = rho
            self.v[x,self.ny-1,z] = ti.Vector([0.0,0.0,0.0])
        if ti.static(self.bc[3]==BC_FLOW.inlet):
            self.v[self.nx-1,y,z] = self.v[self.nx-2,y,z]
        if ti.static(self.bc[3]==BC_FLOW.outlet):
            rho = self.rho_BC[3]
            v = 0.0
            u = -1+(2*(self.F[self.nx-1,y,z][1]+self.F[self.nx-1,y,z][5]+self.F[self.nx-1,y,z][8])+self.F[self.nx-1,y,z][0]+self.F[self.nx-1,y,z][2]+self.F[self.nx-1,y,z][4])/rho
            self.rho[self.nx-1,y,z] = rho
            self.v[self.nx-1,y,z] = ti.Vector([u,v,0.0])
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
        for s in ti.static(range(9)):
            self.f[0,y,z][s] = self.feq9(s,0,y,z)
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
        for s in ti.static(range(9)):
            self.f[self.nx-1,y,z][s] = self.feq9(s,self.nx-1,y,z)
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
        for s in ti.static(range(9)):
            self.f[x,0,z][s] = self.feq9(s,x,0,z)
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
        for s in ti.static(range(9)):
            self.f[x,self.ny-1,z][s] = self.feq9(s,x,self.ny-1,z)

    def updateBC(self,t):
        # internal update
        self.updateBC_kernel()
        # external update
        for func in self.UpdateBCfunc:
            func(self,t)
    @ti.kernel
    def updateBC_kernel(self):
        if ti.static(self.bc[0]==BC_FLOW.inlet_flow):
            flow0 = 0.0
            for j,k in ti.ndrange(self.ny,self.nz):
                flow0 += self.rho[0,j,k]*self.v[0,j,k][0]
            coef = self.flow_BC[0]/flow0
            a = .1
            for j,k in ti.ndrange(self.ny,self.nz):
                self.v_bc_profile[0][0,j,k] = (a*coef+1-a)*self.v[1,j,k]
                self.rho_bc_profile[0][0,j,k] = self.rho[1,j,k]
            
        # if ti.static(self.bc[0]==BC_FLOW.inlet_flow):
        # for j,k in ti.ndrange(self.ny,self.nz):
        # if ti.static(self.bc[0]==BC_FLOW.inlet_flow):
        # for i,k in ti.ndrange(self.nx,self.nz):
        # if ti.static(self.bc[0]==BC_FLOW.inlet_flow):
        # for i,k in ti.ndrange(self.nx,self.nz):