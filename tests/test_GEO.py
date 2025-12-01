from LBM.GEO.G2D import Mesh2D
import taichi as ti
import matplotlib.pyplot as plt
ti.init(arch=ti.cpu)

m2d = Mesh2D(5,5)
m2d.CreateMesh2DCircle(2.5,2.5,2)
s,l = m2d.export_numpy()
plt.imshow(s)
plt.show()