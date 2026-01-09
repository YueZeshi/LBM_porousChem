import os
import pytest
from LBM.GEO.STL import StlGenerator
import pyvista as pv
import logging

def test_create_cone_and_read(tmp_path):
    """测试锥体 STL 文件生成与读取"""
    logger = logging.getLogger("test")
    gen = StlGenerator(logger)
    stl_path = tmp_path / "cone.stl"
    # 生成STL文件
    gen.create_cone(str(stl_path))
    assert stl_path.exists(), "STL 文件未生成"
    # 用 pyvista 读取并检查基本属性
    mesh = pv.read(str(stl_path))
    assert mesh.n_points > 0
    assert mesh.n_cells > 0
    assert mesh.volume > 0
    assert mesh.area > 0
    # 检查边界框合理性
    bounds = mesh.bounds
    assert bounds[1] > bounds[0]
    assert bounds[3] > bounds[2]
    assert bounds[5] > bounds[4]

def test_invalid_path():
    """测试异常路径处理"""
    logger = logging.getLogger("test")
    gen = StlGenerator(logger)
    with pytest.raises(Exception):
        gen.create_cone("/invalid/path/cone.stl")
