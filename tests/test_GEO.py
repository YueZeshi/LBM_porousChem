import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

ti.init(arch=ti.cpu)
def voxel_stl(stl_path,scale = 1.0,translate = [0,0,0],rotate = [0,0,0]):
        if not scale:
            scale = 1.0
        if not translate:
            translate = [0,0,0]
        if not rotate:
            rotate = [0,0,0]
        import pyvista as pv
        # 1. 加载STL
        mesh = pv.read(stl_path)
        mesh.scale(scale,inplace=True)
        mesh.rotate_z(rotate[2],inplace = True)
        mesh.rotate_y(rotate[1],inplace = True)
        mesh.rotate_x(rotate[0],inplace = True)
        # mesh.rotate(Rotation.from_euler('ZYX',rotate[::-1],degrees=True),inplace = True)
        mesh.translate(translate,inplace = True)
        # surface = mesh.extract_surface()
        print(mesh)
        # print(surface)
        # 2. 创建体素网格
        voxels = pv.DataSetFilters.voxelize(mesh) # 先变换再体素化，体素化之后再变换会使得网格错位，规则网格无法正确采样
        # voxelsSurface = pv.DataSetFilters.voxelize(surface)
       # 3. 转换为规则网格
        x = 1
        y = 1
        z = 0.01
        nx = 200
        ny = 200
        nz = 1
        dx = 0.01
        padded_x_min = -dx
        padded_x_max = x+dx
        padded_y_min = -dx
        padded_y_max = y+dx
        padded_z_min = -dx
        padded_z_max = z+dx
        padded_x = np.linspace(padded_x_min,padded_x_max,nx+2)
        padded_y = np.linspace(padded_y_min,padded_y_max,ny+2)
        padded_z = np.linspace(padded_z_min,padded_z_max,nz+2)
        X,Y,Z = np.meshgrid(padded_x,padded_y,padded_z,indexing = 'ij')
        grid = pv.StructuredGrid(X,Y,Z)
        # gridSurface = pv.StructuredGrid(meshX,meshY,meshZ) 
        # 4. 采样到规则网格
        sampled = grid.sample(voxels)
        # sampledSurface = gridSurface.sample(voxelsSurface)
        # 5. 提取标量数据为数组
        voxel_array = np.array(sampled['vtkValidPointMask'].reshape(grid.dimensions, order='F'),dtype = float)
        # surface_array = np.array(sampledSurface['vtkValidPointMask'].reshape(gridSurface.dimensions, order='F'),dtype = float)
        voxelField = ti.field(float,shape = voxel_array.shape)
        surfaceField = ti.field(float,shape = voxel_array.shape)
        voxelField.from_numpy(voxel_array)
        @ti.kernel
        def extract_surface_only():
            e1 = ti.Vector([1,0,0])
            e2 = ti.Vector([0,1,0])
            e3 = ti.Vector([-1,0,0])
            e4 = ti.Vector([0,-1,0])
            for i in ti.grouped(voxelField):
                if voxelField[i]==0 or i[0]<1 or i[0]>nx or i[1]<1 or i[1]>nx or i[2]<1 or i[2]>nx: 
                    surfaceField[i] = 0 
                else:
                    num_solid_neighbor = 0
                    if voxelField[i+e1]>0:
                        num_solid_neighbor+=1
                    if voxelField[i+e2]>0:
                        num_solid_neighbor+=1
                    if voxelField[i+e3]>0:
                        num_solid_neighbor+=1
                    if voxelField[i+e4]>0:
                        num_solid_neighbor+=1
                    if num_solid_neighbor==4:
                        surfaceField[i] = 0
                    else:
                        surfaceField[i] = num_solid_neighbor  
        extract_surface_only()
        surface_array = surfaceField.to_numpy()
        return voxel_array[1:-1,1:-1,1:-1],surface_array[1:-1,1:-1,1:-1]
a,b = voxel_stl("data/stl/Cylinder_test2.stl",scale = 10,translate=(0.5,0.5,0),rotate=(0,90,0))
print(np.shape(a))
print(np.shape(b))
fig,ax = plt.subplots(1,2)
ax[0].imshow(a)
ax[1].imshow(b)
plt.show()
plt.show()