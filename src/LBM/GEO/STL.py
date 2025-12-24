
import os
import numpy as np
from ..util.path import root_path
import logging

class StlGenerator:
    def __init__(self,logger:logging.Logger):
        self.logger = logger
        self.stl_repo = os.path.join(root_path(),"data","stl")
        if not os.path.exists(self.stl_repo):
            os.mkdir(self.stl_repo)

    def create_sphere(self):
        """
        origin = center of sphere
        """
        path = os.path.join(self.stl_repo,"sphere.stl")
        if not os.path.exists(path):
            import cadquery as cq
            self.logger.info("Unit sphere creating...")
            sphere = cq.Workplane("XY").sphere(radius=0.5)
            cq.exporters.export(sphere,path)    
            self.logger.info("Unit sphere created.")
        else:
            self.logger.info("Unit sphere found.")
        return path
    
    def create_cylinder(self):
        """
        origin = center of cylinder
        """    
        path = os.path.join(self.stl_repo,"cylinder.stl")
        if not os.path.exists(path):
            import cadquery as cq
            self.logger.info("Unit cylinder creating...")
            cylinder = cq.Workplane("XY").cylinder(height = 1,radius = 0.5)
            cq.exporters.export(cylinder,path)
            self.logger.info("Unit cylinder created.")
        else:
            self.logger.info("Unit cylinder found.")
        return path


    def create_box(self):
        """
        origin = center of box
        """
        path = os.path.join(self.stl_repo,"box.stl")
        if not os.path.exists(path):
            import cadquery as cq
            self.logger.info("Unit box creating...")
            box = cq.Workplane("XY").box(1,1,1).translate((-0.5,-0.5,-0.5))
            cq.exporters.export(box,path)
            self.logger.info("Unit box created.")
        else:
            self.logger.info("Unit box found.")
        return path

    def create_cone(self):
        """
        origin = center of cone
        """
        path = os.path.join(self.stl_repo,"cone.stl")
        if not os.path.exists(path):
            import cadquery as cq
            self.logger.info("Unit cone creating...")        
            cone = (cq.Workplane("XZ")
                .moveTo(0, -0.5)
                .lineTo(0.5, -0.5)
                .lineTo(0, 0.5)
                .close()
                .revolve(360, (0, 0), (0, 1)))
            cq.exporters.export(cone,path)
            self.logger.info("Unit cone created.")
        else:
            self.logger.info("Unit cone found.")
        return path

class StlReader:
    def __init__(self,x,y,z,dx,dimension,logger:logging.Logger):
        self.X = x
        self.Y = y
        self.Z = z
        self.nx = int(x/dx)
        self.ny = int(y/dx)
        self.nz = int(z/dx)
        self.dx = dx
        self.dimension = dimension
        self.logger = logger
        if not self.logger:
            self.logger = logging.Logger("default logger")
        self.e = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]

    def voxel_stl(self,stl_path,scale = 1.0,translate = [0,0,0],rotate = [0,0,0]):
        if scale is None:
            scale = 1.0
        if translate is None:
            translate = [0,0,0]
        if rotate is None:
            rotate = [0,0,0]
        import pyvista as pv
        import taichi as ti
        # 1. 加载STL
        self.logger.info(f"Stl {stl_path} loading...")
        mesh = pv.read(stl_path)
        mesh.scale(scale,inplace=True)
        mesh.rotate_z(rotate[2],inplace = True)
        mesh.rotate_y(rotate[1],inplace = True)
        mesh.rotate_x(rotate[0],inplace = True)
        # mesh.rotate(Rotation.from_euler('ZYX',rotate[::-1],degrees=True),inplace = True)
        mesh.translate(translate,inplace = True)
        # 2. 创建体素网格
        voxels = pv.DataSetFilters.voxelize(mesh) # 先变换再体素化，体素化之后再变换会使得网格错位，规则网格无法正确采样
        # 3. 转换为规则网格
        padded_x_min = -self.dx
        padded_x_max = self.X+self.dx
        padded_y_min = -self.dx
        padded_y_max = self.Y+self.dx
        padded_z_min = -self.dx
        padded_z_max = self.Z+self.dx
        
        padded_x = np.linspace(padded_x_min,padded_x_max,self.nx+2)
        padded_y = np.linspace(padded_y_min,padded_y_max,self.ny+2)
        padded_z = np.linspace(padded_z_min,padded_z_max,self.nz+2)
        
        X,Y,Z = np.meshgrid(padded_x,padded_y,padded_z,indexing='ij')
        grid = pv.StructuredGrid(X,Y,Z)
        # 4. 采样到规则网格
        sampled = grid.sample(voxels)
        
        # 5. 提取标量数据为数组
        voxel_array = np.array(sampled['vtkValidPointMask'].reshape(grid.dimensions, order='F'),dtype = float)
        voxelField = ti.field(float,shape = voxel_array.shape)
        surfaceField = ti.field(float,shape = voxel_array.shape)
        voxelField.from_numpy(voxel_array)
        @ti.kernel
        def extract_surface_2D():
            for i in ti.grouped(voxelField):
                if voxelField[i]==0 or i[0]<1 or i[0]>self.nx or i[1]<1 or i[1]>self.ny: 
                    surfaceField[i] = 0 
                else:
                    num_solid_neighbor = 0
                    for j in ti.static(range(4)):
                        if voxelField[i+self.e[j]] > 0.0:
                            num_solid_neighbor += 1
                    if num_solid_neighbor==4:
                        surfaceField[i] = 0
                    elif num_solid_neighbor == 3:
                        surfaceField[i] = 1.0
                    elif num_solid_neighbor == 2:
                        surfaceField[i] = 1.414
                    elif num_solid_neighbor == 1:
                        surfaceField[i] = 2.0
                    elif num_solid_neighbor == 0:
                        surfaceField[i] = 4.0
        @ti.kernel
        def extract_surface_3D():
            for i in ti.grouped(voxelField):
                if voxelField[i]==0 or i[0]<1 or i[0]>self.nx or i[1]<1 or i[1]>self.ny or i[2]<1 or i[2]>self.nx: 
                    surfaceField[i] = 0 
                else:
                    num_solid_neighbor = 0
                    for j in ti.static(range(6)):
                        if voxelField[i+self.e[j]] > 0.0:
                            num_solid_neighbor += 1
                    if num_solid_neighbor==6:
                        surfaceField[i] = 0
                    elif num_solid_neighbor == 5:
                        surfaceField[i] = 1.0
                    elif num_solid_neighbor == 4:
                        surfaceField[i] = 1.414
                    elif num_solid_neighbor == 3:
                        surfaceField[i] = 2.0
                    elif num_solid_neighbor == 2:
                        surfaceField[i] = 3.0
                    elif num_solid_neighbor == 1:
                        surfaceField[i] = 4.0
                    elif num_solid_neighbor == 0:
                        surfaceField[i] = 6.0
        if self.dimension==3:
            extract_surface_3D()
        elif self.dimension==2:
            extract_surface_2D()
        else:
            self.logger.warning("Stl reader dimension error.")
        surface_array = surfaceField.to_numpy()
        self.logger.info(f"Stl {stl_path} loaded.")
        return voxel_array[1:-1,1:-1,1:-1],surface_array[1:-1,1:-1,1:-1]
    
