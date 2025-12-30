import subprocess
import click
import os
paraview_path = r"D:\Softwares\ParaView\bin\paraview.exe"
@click.command()
@click.option('--data','-d',default = "output",help="The data folder")
def paraview(data):
    if os.path.exists(data):
        print("Activating paraView...")
        subprocess.run([paraview_path,f"--data={data}"])
    else:
        print("Data folder not found")