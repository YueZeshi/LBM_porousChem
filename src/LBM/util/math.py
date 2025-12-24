import numpy as np
from scipy.spatial.transform import Rotation

def vectors_to_euler(v1, v2, seq='XYZ', degrees=False):
    """一步计算两个向量间的欧拉角"""
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
