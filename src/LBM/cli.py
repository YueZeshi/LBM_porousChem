import click
import os
from ruamel.yaml import YAML
import shutil
from .util.path import root_path
import logging
import sys
#
@click.command()
@click.option('--config','-c',default ='config.yaml',help = 'the configuration of simulation case')
@click.option('--verbose','-v',default=1,help = "log level 0:warning;1:info;2:debug")
@click.option('--clear',is_flag=True,help = 'clear past result folder')
def run(config,verbose,clear):
    verbose_level = {
        0:logging.WARNING,
        1:logging.INFO,
        2:logging.DEBUG
    }
    logger = logging.getLogger("LBM 2D logger")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(verbose_level[verbose])
    if not clear:
        if not os.path.exists(config):
            found = False
            for p in os.listdir(os.curdir):
                if p.endswith(".yaml"):
                    logger.info(f"{config} not found. Found {p} yaml file instead.")
                    config  = p
                    found = True
                    break
            if not found:
                raise ValueError("The configuration file not exist. Please specify the input configuration file name or create the default file : config.yaml.")
        yaml = YAML()
        with open(config,'r',encoding="utf8") as f:
            logger.info(f"Loading {config} file")
            config = yaml.load(f)
            from .app import application
            application(config,logger)
    else:                
        dirs = os.listdir()
        for dir in dirs:
            if os.path.isdir(dir):
                isDelete = input(f"Do you want to delete {dir} folder (y/[n]):")
                if isDelete=='y':
                    shutil.rmtree(dir,ignore_errors=True)
                    print(f"Delete {dir} folder.")
                else:
                    print(f"Conserve {dir} folder.")
    

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
    paraview_path = False
    while not paraview_path:
        yaml = YAML()
        config_path = os.path.join(root_path(),"src","LBM","config.yaml")
        if os.path.exists(config_path):# config file exist
            with open(config_path,"r+") as f:
                config = yaml.load(f)
                paraview_path = config.get("paraview_path",False)
                if paraview_path: # found paraview
                    import subprocess
                    try:
                        subprocess.run([paraview_path,f"--data={data}"])
                    except FileNotFoundError:
                        input_path = input(f"Paraview path {paraview_path} not valid. \nPlease input the paraview absolute path:")
                        config["paraview_path"] = input_path
                        f.seek(0)
                        yaml.dump(config,f) 
                        paraview_path= False
                else: # not found paraview
                    input_path = input("Paraview not found. \nPlease input the paraview absolute path:")
                    config["paraview_path"]=input_path
                    f.truncate(0)
                    f.seek(0)
                    yaml.dump(config,f)              
        else: # no config file
            with open(config_path,"w") as f:
                config = {"name":"config.yaml"}
                yaml.dump(config,f)
        