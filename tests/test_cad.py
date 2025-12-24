# import numpy as np
# import matplotlib.pyplot as plt
import cadquery as cq
import pyvista as pv
from LBM.GEO.STL import StlGenerator
import logging

# 方法1：最简单的可视化
def visualize_stl_basic(stl_path):
    """基础STL可视化"""
    
    # 加载STL
    mesh = pv.read(stl_path)
    
    print(f"STL信息:")
    print(f"  顶点数: {mesh.n_points}")
    print(f"  三角形数: {mesh.n_cells}")
    print(f"  边界框: {mesh.bounds}")
    print(f"  表面积: {mesh.area:.2f}")
    print(f"  体积: {mesh.volume:.2f}")
    
    # 创建绘图器
    plotter = pv.Plotter()
    
    # 添加网格
    plotter.add_mesh(mesh, 
                    color='lightblue',
                    show_edges=True,      # 显示边缘
                    edge_color='black',   # 边缘颜色
                    line_width=1,         # 线宽
                    opacity=1.0)          # 透明度
    
    # 添加坐标轴
    plotter.show_axes()
    
    # 添加标题
    plotter.add_title(f"STL模型: {stl_path}", font_size=16)
    
    # 显示
    plotter.show()

# 使用示例


if __name__=="__main__":
    logger = logging.Logger(__name__)
    meshGenerator = StlGenerator(logger)
    meshGenerator.create_cone()
    path = "data/stl/cone.stl"
    # path = root_path()
    print(path)
    r = 0.5
    visualize_stl_basic(path)
