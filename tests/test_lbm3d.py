
import pytest
import numpy as np
import taichi as ti
from LBM.LBM3D import LBM3DSolver

ti.init(arch=ti.cpu)

def test_lbm3d_base_init():
	lbm = LBM3DSolver(X=1.0, Y=1.0, Z=1.0, dx=0.1, dt=0.01)
	# 检查基本属性
	assert lbm.nx == 10
	assert lbm.ny == 10
	assert lbm.nz == 10
	assert hasattr(lbm, 'rho')
	assert hasattr(lbm, 'v')
	assert hasattr(lbm, 'solid')
	# 检查边界条件
	assert hasattr(lbm, 'bc')
	# 检查物理场 shape
	assert lbm.rho.shape == (lbm.nx, lbm.ny, lbm.nz)
	assert lbm.v.shape == (lbm.nx, lbm.ny, lbm.nz)
	# 检查默认方法可调用
	lbm.default_init()
	lbm.static_init_kernel()
	lbm.updateBC(0)
	lbm.__call__()
	repr(lbm)


