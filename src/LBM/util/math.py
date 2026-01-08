"""数值与几何小工具。"""
import numpy as np
from scipy.spatial.transform import Rotation

def vectors_to_euler(v1, v2, seq='XYZ', degrees=False):
    """计算将向量 `v1` 旋转到 `v2` 的欧拉角。

    Parameters
    ----------
    v1, v2 : array-like, shape (3,)
        起始与目标方向向量。
    seq : str, default 'XYZ'
        欧拉角旋转序列，传入 `scipy.spatial.transform.Rotation.as_euler` 支持的序列。
    degrees : bool, default False
        为 True 时返回角度制，否则返回弧度制。

    Returns
    -------
    np.ndarray, shape (3,)
        欧拉角向量，对应 `seq` 指定顺序。

    Notes
    -----
    - 当两向量共线（同向/反向）时采用稳定的特例处理。
    - 内部基于旋转轴-角表示并转换为欧拉角。
    """
    v1, v2 = np.asarray(v1), np.asarray(v2)
    
    # 归一化
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # 特殊情况处理
    if np.allclose(v1, v2):
        return np.zeros(3)
    
    if np.allclose(v1, -v2):
        # 向量相反
        if abs(v1[0]) > 1e-10:
            axis = np.array([-v1[1], v1[0], 0])
        else:
            axis = np.array([0, -v1[2], v1[1]])
        axis = axis / np.linalg.norm(axis)
        rot = Rotation.from_rotvec(axis * np.pi)
    else:
        # 一般情况
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        rot = Rotation.from_rotvec(axis * angle)
    
    return rot.as_euler(seq, degrees=degrees)
