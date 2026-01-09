

# test_util.py
import numpy as np
import pytest
from LBM.util import math, path, constant, flag

def test_vectors_to_euler_identity():
    v = np.array([1, 0, 0])
    angles = math.vectors_to_euler(v, v)
    assert np.allclose(angles, 0)

def test_vectors_to_euler_opposite():
    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    angles = math.vectors_to_euler(v1, v2)
    assert np.isfinite(angles).all()

def test_root_path_found(tmp_path, monkeypatch):
    # 在临时目录下创建 pyproject.toml
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("test")
    monkeypatch.chdir(root)
    assert path.root_path() == str(root)

def test_root_path_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        path.root_path()

def test_constant_values():
    assert constant.c > 1e8
    assert "O" in constant.MOLEMASS
    assert constant.MOLEMASS["H"] == pytest.approx(1.008, rel=1e-3)

def test_flag_enum():
    assert flag.VISCOSITY_MODEL.CONSTANT.value == 1
    assert flag.PORO_MODEL.ERGUN.name == "ERGUN"
    assert isinstance(flag.BC_MODEL.CONSTANT, flag.BC_MODEL)