"""路径工具：定位项目根目录。"""
import os

def root_path():
    """查找项目根路径（向上最多 20 层）。

    以存在标记文件 `pyproject.toml` 为根目录判据，从当前工作目录
    向上逐层查找，最多 20 层。

    Returns
    -------
    str
        根目录的绝对路径。

    Raises
    ------
    ValueError
        当超过最大层数仍未找到时抛出。
    """
    path = os.path.abspath(os.curdir)
    tag_file = "pyproject.toml" # 使用该文件定位根路径
    found = False
    max_depth = 20
    depth = 0
    while not found and depth < max_depth :
        for p in os.listdir(path):
            if p==tag_file:
                found = True
                break
        if not found : 
            path = os.path.dirname(path)
            depth += 1
    if depth < max_depth:
        return path
    else:
        raise ValueError("Root path not found.")
        