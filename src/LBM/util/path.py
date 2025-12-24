import os

def root_path():
    path = os.path.abspath(os.curdir)
    tag_file = "pyproject.toml" # 使用该文件定位根路径
    found = False
    max_depth = 5
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
        