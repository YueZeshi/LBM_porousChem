"""
generation of 2D geometry mesh using taichi
"""
import numpy as np
import taichi as ti

TYPE_V2 = ti.types.vector(2,ti.f32)
TYPE_V3 = ti.types.vector(3,ti.f32)
@ti.data_oriented
class Mesh2D:
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny
        self.S = ti.field(ti.f32,shape=(nx,ny,1))
        self.L = ti.field(ti.f32,shape=(nx,ny,1))
        self.d = ti.field(ti.f32,shape=(nx+1,ny+1,1))

    def export_numpy(self):
        return self.S.to_numpy(),self.L.to_numpy()
    
    def CreateMesh2D(self,shape:str,*params):
        if shape.lower()=="circle":
            self.CreateMesh2DCircle(params[0][0],params[0][1],params[1]) 
        if shape.lower()=="rectangular":
            self.CreateMesh2DRectangle(ti.Vector(params[0]),ti.Vector(params[1]))

    @ti.kernel
    def CreateMesh2DRectangle(self,vertex1:TYPE_V3,vertex2:TYPE_V3):# type: ignore
        for i,j in ti.ndrange(self.nx+1,self.ny+1): # 计算并存储各个格子顶点的距离值
            point = ti.Vector([i,j,0])
            self.d[i,j,0] = self.rectangle(point,vertex1,vertex2)
        for i in ti.grouped(self.S):
            s,l = self.calculate_s_l(self.d[i],self.d[i+ti.Vector([1,0,0])],self.d[i+ti.Vector([0,1,0])],self.d[i+ti.Vector([1,1,0])])
            self.S[i] = s
            self.L[i] = l
        
    @ti.kernel
    def CreateMesh2DCircle(self, center_x:float,center_y:float, radius:float): # type: ignore
        for i,j in ti.ndrange(self.nx+1,self.ny+1): # 计算并存储各个格子顶点的距离值
            point = ti.Vector([i,j,0])
            self.d[i,j,0] = self.circle(point,ti.Vector([center_x,center_y,0]),radius)
        for i in ti.grouped(self.S):
            s,l = self.calculate_s_l(self.d[i],self.d[i+ti.Vector([1,0,0])],self.d[i+ti.Vector([0,1,0])],self.d[i+ti.Vector([1,1,0])])
            self.S[i] = s
            self.L[i] = l

    @ti.func
    def circle(self,point,center,radius:float):
        v = point-center
        return v.norm()-radius
    @ti.func 
    def rectangle(self,point,v1,v2):
        dh = abs(point[0]-(v1[0]+v2[0]/2))-abs(v2[0]-v1[0])/2
        dv = abs(point[1]-(v1[1]+v2[1]/2))-abs(v2[1]-v1[1])/2
        return max(dh,dv)
    @ti.func
    def calculate_s_l(self,d1,d2,d3,d4):
        s = 0.0
        l = 0.0
        in_point_count = 0
        for d in ti.static([d1,d2,d3,d4]):
            if d <= 0:
                in_point_count+=1
        if in_point_count==4:
            s = 1.0
            l = 0.0
        elif in_point_count==0:
            s = 0.0
            l = 0.0
        elif in_point_count==1:
            x = 0.0
            y = 0.0
            if d1 < 0:
                x = -d1/(d3-d1)
                y = -d1/(d2-d1)
            if d2 < 0:
                x = -d2/(d1-d2)
                y = -d2/(d4-d2)
            if d3 < 0:
                x = -d3/(d1-d3)
                y = -d3/(d4-d3)
            if d4 < 0:
                x = -d4/(d2-d4)
                y = -d4/(d3-d4)
            l = ti.math.sqrt(x**2+y**2)
            s = x*y/2
        elif in_point_count==2:
            x = 0.0
            y = 0.0
            if d1*d2<0: # 交点在12 34
                if d1<0: # 13 内
                    x = -d1/(d2-d1)
                    y = -d3/(d4-d3)
                else: # 24 内
                    x = -d2/(d1-d2)
                    y = -d4/(d3-d4)
            else: # 交点在13 24
                if d1<0: # 12 内
                    x = -d1/(d3-d1)
                    y = -d2/(d4-d2)
                else: # 34 内
                    x = -d3/(d1-d3)
                    y = -d4/(d2-d4)
            s = (x+y)/2
            l = ti.math.sqrt(1+(x-y)**2)
        elif in_point_count==3:
            x  = 0.0
            y = 0.0
            if d1 > 0:
                x = d1/(d1-d3)
                y = d1/(d1-d2)
            if d2 > 0:
                x = d2/(d2-d1)
                y = d2/(d2-d4)
            if d3 > 0:
                x = d3/(d3-d1)
                y = d3/(d3-d4)
            if d4 > 0:
                x = d4/(d4-d2)
                y = d4/(d4-d3)
                    
            l = ti.math.sqrt(x**2+y**2)
            s = 1 - x*y/2
        return s,l