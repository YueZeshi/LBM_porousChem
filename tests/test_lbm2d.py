import pytest
import numpy as np
import taichi as ti
from LBM.LBM2D import LBM2DSolver

ti.init(arch=ti.cpu)

def test_lbm2d_base_init():
    lbm = LBM2DSolver(X=1.0, Y=1.0, Z=0.001, dx=0.01, dt=0.01)
    # 检查基本属性
    assert lbm.nx == 100
    assert lbm.ny == 100
    assert lbm.nz == 1
    assert hasattr(lbm, 'rho')
    assert hasattr(lbm, 'v')
    assert hasattr(lbm, 'solid')
    # 检查边界条件
    assert len(lbm.bc) == 4
    # 检查物理场 shape
    assert lbm.rho.shape == (lbm.nx, lbm.ny, lbm.nz)
    assert lbm.v.shape == (lbm.nx, lbm.ny, lbm.nz)
    # 检查默认方法可调用
    lbm.default_init()
    lbm.static_init_kernel()
    lbm.updateBC(0)
    lbm.__call__()
    repr(lbm)
