

import numpy as np
import pytest
import logging
from LBM.GEO.STL import StlGenerator, StlReader

@pytest.mark.skipif(not hasattr(StlReader, 'voxel_stl'), reason="StlReader.voxel_stl not available")
def test_stlreader_voxel_stl(tmp_path):
    import pyvista as pv
    # 生成一个简单立方体 STL
    stl_path = tmp_path / "cube.stl"
    cube = pv.Cube(center=(0.5,0.5,0.005), x_length=1, y_length=1, z_length=0.01)
    cube.save(str(stl_path))
    # 构造 StlReader
    logger = logging.getLogger("test")
    reader = StlReader(x=1, y=1, z=0.01, dx=0.01, dimension=2, logger=logger)
    voxels, surface = reader.voxel_stl(str(stl_path), scale=1.0)
    assert voxels.shape == surface.shape
    assert voxels.ndim == 3
    assert np.any(voxels > 0)
    assert np.any(surface >= 0)