"""
generation of 3D geometry mesh using taichi
"""
from typing import Literal
import numpy as np
import taichi as ti

TYPE_V3 = ti.types.vector(3,float)

@ti.data_oriented
class Mesh3D:
    def __init__(self,nx,ny,nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.V = ti.field(float,shape=(nx,ny,nz))
        self.S = ti.field(float,shape=(nx,ny,nz))
        self.d = ti.field(float,shape=(nx+1,ny+1,nz+1))

    def export_numpy(self):
        return self.V.to_numpy(),self.S.to_numpy()
    
    def CreateMesh3D(self,shape:Literal["sphere","cylinder"],border:Literal["decimal","integer"]="decimal",*params)->None:
        if shape.lower()=="sphere":
            if border=="decimal":
                self.CreateMesh3D_Sphere_Decimal(ti.Vector(params[0]),params[1])
            elif border=="integer":
                self.CreateMesh3D_Cylinder_Integer(ti.Vector(params[0]),params[2])
            else:
                raise ValueError
        if shape.lower()=="cylinder":
            if border=="decimal":
                self.CreateMesh3D_Cylinder_Decimal(ti.Vector(params[0]),ti.Vector(params[1]).normalized(),params[2],params[3])
            elif border=="integer":
                self.CreateMesh3D_Cylinder_Integer(ti.Vector(params[0]),ti.Vector(params[1]).normalized(),params[2],params[3])
            else:
                raise ValueError
    @ti.kernel
    def CreateMesh3D_Sphere_Decimal(self, center:TYPE_V3, radius:float): # type: ignore
        for i in ti.grouped(self.d): # 计算并存储各个格子顶点的距离值
            point = i-ti.Vector([0.5,0.5,0.5])
            self.d[i] = self.sphere(point,center,radius)
        for i in ti.grouped(self.S):
            v,s = self.calculate_v_s(\
                self.d[i],self.d[i+ti.Vector([1,0,0])],self.d[i+ti.Vector([0,1,0])],self.d[i+ti.Vector([0,0,1])],\
                self.d[i+ti.Vector([1,1,0])],self.d[i+ti.Vector([1,0,1])],self.d[i+ti.Vector([0,1,1])],self.d[i+ti.Vector([1,1,1])])
            self.V[i] += v
            self.S[i] += s
    @ti.kernel
    def CreateMesh3D_Sphere_Integer(self, center:TYPE_V3, radius:float): # type: ignore
        for i in ti.grouped(self.d): # 计算并存储各个格子顶点的距离值
            point = i-ti.Vector([0.5,0.5,0.5])
            self.d[i] = self.sphere(point,center,radius)
        for i in ti.grouped(self.S):
            v,s = self.calculate_v_s(\
                self.d[i],self.d[i+ti.Vector([1,0,0])],self.d[i+ti.Vector([0,1,0])],self.d[i+ti.Vector([0,0,1])],\
                self.d[i+ti.Vector([1,1,0])],self.d[i+ti.Vector([1,0,1])],self.d[i+ti.Vector([0,1,1])],self.d[i+ti.Vector([1,1,1])])
            if v>0:
                self.V[i] += 1
            self.S[i] += s
    @ti.kernel
    def CreateMesh3D_Cylinder_Decimal(self, center:TYPE_V3, up:TYPE_V3,radius:float,height:float): # type: ignore
        for i in ti.grouped(self.d): # 计算并存储各个格子顶点的距离值
            point = i-ti.Vector([0.5,0.5,0.5])
            self.d[i] = self.cylinder(point,center,up,radius,height)
        for i in ti.grouped(self.S):
            v,s = self.calculate_v_s(\
                self.d[i],self.d[i+ti.Vector([1,0,0])],self.d[i+ti.Vector([0,1,0])],self.d[i+ti.Vector([0,0,1])],\
                self.d[i+ti.Vector([1,1,0])],self.d[i+ti.Vector([1,0,1])],self.d[i+ti.Vector([0,1,1])],self.d[i+ti.Vector([1,1,1])])
            self.V[i] += v
            self.S[i] += s
    @ti.kernel
    def CreateMesh3D_Cylinder_Integer(self, center:TYPE_V3, up:TYPE_V3,radius:float,height:float): # type: ignore
        for i in ti.grouped(self.d): # 计算并存储各个格子顶点的距离值
            point = i-ti.Vector([0.5,0.5,0.5])
            self.d[i] = self.cylinder(point,center,up,radius,height)
        for i in ti.grouped(self.S):
            v,s = self.calculate_v_s(\
                self.d[i],self.d[i+ti.Vector([1,0,0])],self.d[i+ti.Vector([0,1,0])],self.d[i+ti.Vector([0,0,1])],\
                self.d[i+ti.Vector([1,1,0])],self.d[i+ti.Vector([1,0,1])],self.d[i+ti.Vector([0,1,1])],self.d[i+ti.Vector([1,1,1])])
            if v>0:
                self.V[i] += 1
            self.S[i] += s
    @ti.func
    def sphere(self,point,center,radius:float):
        v = point-center
        return v.norm()-radius
    @ti.func
    def cylinder(self,point,center,up,radius,height):
        v = point-center
        z = v.dot(up)
        r= ti.math.length(v-z*up)
        d1 = ti.abs(z)-height/2
        d2 = r-radius
        return max(d1,d2)
    @ti.func
    def calculate_v_s(self,d1,d2,d3,d4,d5,d6,d7,d8):
        v = 0.0
        s = 0.0
        in_point_count = 0
        for d in ti.static([d1,d2,d3,d4,d5,d6,d7,d8]):
            if d < 0:
                in_point_count+=1
        if in_point_count==8:
            v = 1.0
            s = 0.0
        elif in_point_count==0:
            v = 0.0
            s = 0.0
        else:
            d_p = 0.0
            d_n = 0.0
            for d in ti.static([d1,d2,d3,d4,d5,d6,d7,d8]):
                if d > 0:
                    d_p += d
                else:
                    d_n += d
            ratio = -d_n/(d_p-d_n)
            v = ratio
            s = ti.math.sin(ratio*ti.math.pi)**2
        return v,s