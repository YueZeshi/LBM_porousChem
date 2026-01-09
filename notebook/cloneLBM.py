import os
import sys
import argparse
import shutil

if __name__ == "__main__":
    target = sys.argv[1]
    new_path = os.path.join(os.path.pardir,target)
    # os.makedirs(new_path,exist_ok=True)
    shutil.copytree(".",new_path,ignore=shutil.ignore_patterns("*.pyc","__pycache__","*.vtu","*.vtk","*.vtr","*.dat","*.png","*.log","temp","result",".venv","build",".venv","*.egg-info"),dirs_exist_ok=True)