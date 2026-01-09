import os
import sys
import argparse
import shutil

if __name__ == "__main__":
    # os.makedirs(new_path,exist_ok=True)
    shutil.rmtree("result",ignore_errors=True)
    shutil.rmtree("temp",ignore_errors=True)