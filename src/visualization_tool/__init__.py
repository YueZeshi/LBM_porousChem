"""LBM 可视化工具包。

包含：
- `PVDWriter`：生成 `.pvd` 集合文件，组织时间序列 VTK。
- `PvViewer`：基于 PyVista 的交互式 PVD 浏览器，支持播放/跳帧/配色等。

示例用法：
	from LBM.visualization_tool import PVDWriter, PvViewer
	pvd = PVDWriter(path="output", name="case")
	pvd.addVTK(time=0.0, filename="case_0.vts")
	pvd.writePVD()

	viewer = PvViewer(pvd_file="output/case.pvd")
	viewer.show()
"""

from .PVD import PVDWriter
from .PvViewer import PvViewer

__all__ = ["PVDWriter", "PvViewer"]
