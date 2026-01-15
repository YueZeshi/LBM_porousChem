import pyvista as pv
import numpy as np
import sys

def stl_to_voxel_pyvista(stl_path, resolution=10):
    """
    使用PyVista进行体素化
    
    参数:
    stl_path: STL文件路径
    resolution: 网格分辨率
    
    返回:
    voxel_array: 3D二值数组
    """
    # 1. 加载STL
    mesh = pv.read(stl_path)
    # 2. 创建体素网格
    voxels = mesh.voxelize()
    # 3. 转换为规则网格
    x = np.linspace(0,100,10)
    y = np.linspace(0,100,10)
    z = np.linspace(0,100,10)
    X,Y,Z = np.meshgrid(x,y,z)
    grid = pv.StructuredGrid(X,Y,Z)

    # 4. 采样到规则网格
    sampled = grid.sample(mesh)
    
    # 5. 提取标量数据为数组
    voxel_array = sampled['vtkValidPointMask'].reshape(grid.dimensions, order='F')
    
    return voxel_array,grid

if __name__ =="__main__":
    stl = sys.argv[1]
    a,b = stl_to_voxel_pyvista(stl)
    print(a.sum(),b)