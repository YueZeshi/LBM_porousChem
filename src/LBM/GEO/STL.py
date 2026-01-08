"""STL 几何辅助工具。

包含两部分：
- `StlGenerator`：依赖 CadQuery 生成简单几何（球/圆柱/盒/圆锥）的单位 STL 文件并缓存到仓库。
- `StlReader`：读取 STL，通过 PyVista 体素化并采样到规则网格，返回体素占据与近似表面长度/面积权重。

依赖：
- 生成器使用 `cadquery`（可选，若已存在对应 STL 则不需要）。
- 读取器使用 `pyvista` 与 `taichi`。
"""

import os
import numpy as np
from ..util.path import root_path
import logging

class StlGenerator:
    """简单几何 STL 生成器。

    Parameters
    ----------
    logger : logging.Logger
        日志记录器，用于打印创建/复用信息。
    """
    def __init__(self,logger:logging.Logger):
        self.logger = logger
        self.stl_repo = os.path.join(root_path(),"data","stl")
        if not os.path.exists(self.stl_repo):
            os.mkdir(self.stl_repo)

    def create_sphere(self):
        """生成单位球 STL（半径 0.5，原点为球心）。

        若仓库中已有则直接复用。

        Returns
        -------
        str
            STL 文件路径。
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
        """生成单位圆柱 STL（半径 0.5，高度 1.0，原点为中心）。

        若仓库中已有则直接复用。

        Returns
        -------
        str
            STL 文件路径。
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
        """生成单位立方体 STL（边长 1.0，原点为中心）。

        通过平移使盒体覆盖 [-0.5,0.5]^3。

        Returns
        -------
        str
            STL 文件路径。
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
        """生成单位圆锥 STL（底半径约 0.5，高 1.0，原点为几何中心）。

        Returns
        -------
        str
            STL 文件路径。
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
    """STL 体素化与采样读取器。

    Parameters
    ----------
    x, y, z : float
        模拟域在三个方向的尺寸（物理单位）。
    dx : float
        网格步长（物理单位）。
    dimension : int
        几何维度，2 或 3。用于选择表面提取规则。
    logger : logging.Logger
        日志记录器；若为 None，将创建默认记录器。
    """
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
        """体素化 STL 并采样到规则网格，返回占据与近似表面权重。

        流程：读取与几何变换 → 体素化 → 构造带边界的规则网格 → 采样 →
        邻域检测计算近似表面权重（2D/3D 不同规则）。

        Parameters
        ----------
        stl_path : str or PathLike
            STL 文件路径。
        scale : float or list[float], default 1.0
            缩放因子（标量或各向异性列表）。
        translate : list[float]
            三维平移量。
        rotate : list[float]
            欧拉角旋转（度），顺序为 [rx, ry, rz]。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - `voxel_array` : (nx, ny, nz) 体素占据数组（0/1/浮点）
            - `surface_array` : (nx, ny, nz) 近似表面权重（与邻域空/实关系有关）

        Notes
        -----
        - 体素化使用 `pv.DataSetFilters.voxelize`，采样基于结构化网格。
        - 输出剔除了一圈边界（返回截取的 `1:-1`）。
        """
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
    
