import click
import os
from ruamel.yaml import YAML
import shutil
import config
#
@click.command()
@click.option('--config','-c',default ='config.yaml',help = 'the configuration of simulation case')
@click.option('--clear',is_flag=True,help = 'clear past result folder')
def run(config,clear):
    if clear:
        dirs = os.listdir()
        for dir in dirs:
            if os.path.isdir(dir):
                isDelete = input(f"Do you want to delete {dir} folder (y/[n]):")
                if isDelete=='y':
                    shutil.rmtree(dir,ignore_errors=True)
                    print(f"Delete {dir} folder.")
                else:
                    print(f"Conserve {dir} folder.")
    elif os.path.exists(config):
        yaml = YAML()
        with open(config,'r',encoding="utf8") as f:
            data = yaml.load(f)
            if data["basic"]["dimension"] == 2:
                from app import application_2D
                application_2D(data)
            if data["basic"]["dimension"] == 3:
                from app import application_3D
                application_3D(data)
            if data["basic"]["dimension"] not in [2,3]:
                raise ValueError("The dimention value in the configuration file is invalid. Please set it to 2 or 3 depending on your case.")
    else:
        raise ValueError("The configuration file not exist. Please specify the input configuration file name or create the default file : config.yaml.")


paraview_path = config.paraview_path
@click.command()
@click.option('--data','-d',default = None ,help="The data folder")
def paraview(data):
    if data: # data explicit
        if os.path.exists(data):
            if os.path.isdir(data): # find pvd in folder
                found = False
                for filename in os.listdir(data):
                    if filename.endswith(".pvd"):
                        data = os.path.join(data,filename)
                        found = True
                        break
                if not found:
                    raise ValueError(f"No pvd file in {data}")
        else:
            raise ValueError(f"Pvd file {data} not exist")
    else: # data file implicit
        found = False
        for dir in os.listdir():
            if os.path.isdir(dir) and not found:
                for filename in os.listdir(dir):
                    if filename.endswith(".pvd"):
                        data = os.path.join(dir,filename)
                        found = True
                        break
        if not found:
            raise ValueError("No pvd file in current folder.")
        else:
            print(f"Found {data} pvd file")
    print(f"Activating paraView {data}...")
    import subprocess
    subprocess.run([paraview_path,f"--data={data}"])
        