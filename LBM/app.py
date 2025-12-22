import numpy as np
import ruamel.yaml
import ruamel.yaml.comments
import taichi as ti
import os
import time
import ruamel
from LBM.util.flag import *

def application_2D(config:ruamel.yaml.comments.CommentedMap):
    # Implementation for 2D application
    startTime = time.time()
    from LBM.LBM2D import LBM2DSolver
    ARCH = config["basic"].get("arch")
    try:
        if ARCH=="gpu":
            ti.init(arch=ti.gpu)
        elif ARCH == "cpu":
            ti.init(arch=ti.cpu)
        elif ARCH == "vulkan":
            ti.init(arch=ti.vulkan)
        else:
            ti.init(arch=ti.cpu)
            print(f"ARCH {ARCH} not valid.")
            ARCH = "cpu"
    except:
        ti.init(arch=ti.cpu)
        print(f"ARCH {ARCH} not valid.")
        ARCH = "cpu"
    print(f"Running 2D LBM on {ARCH}...")

    x = config["spaceControl"]["geometry"][0]
    y = config["spaceControl"]["geometry"][1]
    dt = config["timeControl"]["DT"]
    dx = config["spaceControl"]["DX"]
    isThermal = config["module"].get("temperature")
    isPoro = config["module"].get("porous_media")
    isChemical = config["module"].get("chemistry")
    isRadiation = config["module"].get("radiation")
    name = config["basic"].get("name")
    # initial setting
    lb = LBM2DSolver(x,y,dx,dt,name,isThermal,isChemical,isPoro,isRadiation)
    
    lb.source_term_model = SOURCE_TERM.MICRO
    lb.force_term_model = FORCE_TERM.GUO
    
    # viscosity
    lb.set_viscosity(config["flowProperties"]["viscosity"])
    # thermal
    if isThermal:
        pass
    # chemistry
    if isChemical:
        # load chemical mechanism
        cantera_file = config["chemicalProperties"]["canteraFile"]
        lb.load_cantera(cantera_file)

    # boundary condition
    boundaryName = ["left","right","down","up"]
    for i in range(4):
        bc =config["boundaryCondition"][boundaryName[i]] 
        bc_flow = bc["flow"]
        if bc_flow["type"]=="inlet":
            v = bc_flow["velocity"]
            lb.set_BC(i,BC_FLOW.inlet)
            lb.set_v_BC_value(i,v)
            rho = bc_flow.get("rho")
            if rho:
                lb.set_rho_BC_value(i,rho)
        elif bc_flow["type"]=="outlet":
            rho = bc_flow["rho"]
            lb.set_BC(i,BC_FLOW.outlet)
            lb.set_rho_BC_value(i,rho)
        elif bc_flow["type"]=="wall":
            lb.set_BC(i,BC_FLOW.wall)
        ## bc of temperature field
        if isThermal:
            bc_thermal = bc["thermal"]
            if bc_thermal["type"]=="inlet":
                v = bc["velocity"]
                lb.set_BC(i,BC_FLOW.inlet)
                lb.set_v_BC_value(i,v)
            elif bc["type"]=="outlet":
                rho = bc["rho"]
                lb.set_BC(i,BC_FLOW.outlet)
                lb.set_rho_BC_value(i,rho)
            elif bc["type"]=="wall":
                lb.set_BC(i,BC_FLOW.wall)

    # load all geometry information in geo_dict dictionary name:(surface array, border array)
    geo_dict = {}
    geo_infos = config.get("geometry")
    if geo_infos:
        from LBM.GEO.G2D import Mesh2D
        for geo_name in geo_infos:
            shape = geo_infos[geo_name]["type"]
            if shape == "circle":
                m2d = Mesh2D(x,y,dx)
                center = geo_infos[geo_name]["center"]
                radius = geo_infos[geo_name]["radius"]
                m2d.CreateMesh2DCircle(center[0],center[1],radius)
                s,l = m2d.export_numpy()
                geo_dict[geo_name] = (s,l)
            elif shape == "rectangle":
                m2d = Mesh2D(x,y,dx)
                point1 = geo_infos[geo_name]["point1"]
                point2 = geo_infos[geo_name]["point2"]
                m2d.CreateMesh2DRectangle(point1[0],point1[1],point2[0],point2[1])
                s,l = m2d.export_numpy()
                geo_dict[geo_name] = (s,l)
            elif shape == "stl":
                # read stl
                path = geo_infos[geo_name].get("path")
                trans = geo_infos[geo_name].get("translate")
                rot = geo_infos[geo_name].get("rotate")
                scale = geo_infos[geo_name].get("scale")
                mesh,grid = lb.voxel_stl(path,scale,trans,rot)
                geo_dict[geo_name] = (mesh,mesh)


    # set initial condition including setting soid phase
    initialCondition = config["initialCondition"]
    ## solid phase
    solidIC = initialCondition.get("solid")
    if solidIC:
        for solid in solidIC.values():
            if isPoro:
                eps = solid["porosity"]
                if solid["zone"]=="ALL":
                    s = np.zeros((lb.nx,lb.ny,lb.nz))
                    for geo_name,geo_data in geo_dict.items():
                        s += geo_data[0]
                    s = (1-eps)*s
                    lb.init_field(lb.solid,s)
                elif type(solid["zone"]) is ruamel.yaml.comments.CommentedSeq: # []
                    s = np.zeros((lb.nx,lb.ny,lb.nz))
                    for zoneName in solid["zone"]:
                        s += geo_dict[zoneName][0]
                    s = (1-eps)*s
                    lb.init_field(lb.solid,s)
                elif type(solid["zone"]) is str:
                    s = geo_dict[solid["zone"]][0]
                    s = (1-eps)*s
                    lb.init_field(lb.solid,s)
            else:
                if solid["zone"]=="ALL":
                    s = np.zeros((lb.nx,lb.ny,lb.nz))
                    for geo_name,geo_data in geo_dict.items():
                        s += geo_data[0]
                    s = (s>0)*1
                    lb.init_field(lb.solid,s)
                elif type(solid["zone"]) is ruamel.yaml.comments.CommentedSeq: # []
                    s = np.zeros((lb.nx,lb.ny,lb.nz))
                    for zoneName in solid["zone"]:
                        s += geo_dict[zoneName][0]
                    s = (s>0)*1
                    lb.init_field(lb.solid,s)
                elif type(solid["zone"]) is str:
                    s = geo_dict[solid["zone"]][0]
                    s = (s>0)*1
                    lb.init_field(lb.solid,s)

    lb.init_simulation()
    # time control of simulation 
    lb.tLattice = int(config["timeControl"]["startTime"]/lb.dt)
    endTimeLattice = config["timeControl"]["endTime"]/lb.dt
    printInterval = int(config["outputControl"]["log"]["interval"]/lb.dt)
    exportInterval = int(config["outputControl"]["vtk"]["interval"]/lb.dt)
    exportPath = config["outputControl"]["vtk"]["path"]
    if not exportPath:
        exportPath = "output"
    # clear past result
    if config["outputControl"]["vtk"]["clear"]:
        import shutil
        shutil.rmtree(exportPath,ignore_errors=True)
    lb.exportPath = exportPath
    lb.PVD.exportPath = exportPath
    os.makedirs(exportPath,exist_ok=True)
    snapshotInterval = int(config["outputControl"]["snapshot"]["interval"]/lb.dt)
    snapshotPath = config["outputControl"]["snapshot"]["path"]
    lb.snapshotPath = snapshotPath
    preTime = time.time()
    latticeUpdateBetweenLog = lb.nx*lb.ny*lb.nz*printInterval
    while lb.tLattice<=endTimeLattice:
        if lb.tLattice % printInterval==0:
            calTime = time.time()-preTime
            preTime = time.time()
            if calTime ==0:
                MLUPS = 0
            else:
                MLUPS = latticeUpdateBetweenLog/calTime/1e6
            print(f"Execution time:{calTime:.2f} s , Collapsed time:{(time.time()-startTime):.2f} s, MLUPS = {MLUPS:.2f}")
            print(lb.log_info())
        if lb.tLattice % exportInterval==0:
            lb.export_VTK()
        if lb.tLattice%snapshotInterval==0:
            lb.export_snapshot(config)
        lb.step()
def application_3D(config:ruamel.yaml.comments.CommentedMap):
    # Implementation for 3D application
    startTime = time.time()
    from LBM.LBM3D import LBM3DSolver
    ARCH = config["basic"].get("arch")
    ti.reset()
    ti.reset()
    try:
        if ARCH=="gpu":
            ti.init(arch=ti.gpu)
        elif ARCH == "cpu":
            ti.init(arch=ti.cpu)
        elif ARCH == "vulkan":
            ti.init(arch=ti.vulkan)
        elif ARCH == "cuda":
            ti.init(arch=ti.cuda)
        else:
            ti.init(arch=ti.cpu)
            print(f"ARCH {ARCH} not valid.")
            ARCH = "cpu"
    except:
        ti.init(arch=ti.cpu)
        print(f"ARCH {ARCH} not valid.")
        ARCH = "cpu"
    print(f"Running 3D LBM on {ARCH}...")
    # print(ti.cfg.cpu_max_num_threads,ti.cfg.default_cpu_block_dim,ti.cfg.default_gpu_block_dim)

    x = config["spaceControl"]["geometry"][0]
    y = config["spaceControl"]["geometry"][1]
    z = config["spaceControl"]["geometry"][2]
    dt = config["timeControl"]["DT"]
    dx = config["spaceControl"]["DX"]
    isThermal = config["module"].get("temperature")
    isPoro = config["module"].get("porous_media")
    isChemical = config["module"].get("chemistry")
    isRadiation = config["module"].get("radiation")
    name = config["basic"].get("name")
    # initial setting
    lb = LBM3DSolver(x,y,z,dx,dt,name,isThermal,isChemical,isPoro,isRadiation)
    print("LBM created.")
    lb.source_term_model = SOURCE_TERM.MICRO
    lb.force_term_model = FORCE_TERM.GUO
    
    # viscosity
    lb.set_viscosity(config["flowProperties"]["viscosity"])
    # thermal
    if isThermal:
        pass
    # chemistry
    if isChemical:
        # load chemical mechanism
        if config["chemicalProperties"].get("type")=="cantera":
            print("Cantera loading...")
            cantera_file = config["chemicalProperties"]["canteraFile"]
            lb.load_cantera(cantera_file)
            print("Cantera loaded.")

    # boundary condition
    print("Boundary condition setting...")
    boundaryName = ["left","right","forward","back","down","up"]
    for i in range(6):
        bc =config["boundaryCondition"][boundaryName[i]] 
        bc_flow = bc["flow"]
        if bc_flow["type"]=="inlet":
            v = bc_flow["velocity"]
            lb.set_BC(i,BC_FLOW.inlet)
            lb.set_v_BC_value(i,v)
            rho = bc_flow.get("rho")
            if rho:
                lb.set_rho_BC_value(i,rho)
        elif bc_flow["type"]=="outlet":
            rho = bc_flow["rho"]
            lb.set_BC(i,BC_FLOW.outlet)
            lb.set_rho_BC_value(i,rho)
        elif bc_flow["type"]=="wall":
            lb.set_BC(i,BC_FLOW.wall)
        ## bc of temperature field
        if isThermal:
            bc_thermal = bc["thermal"]
            if bc_thermal["type"]=="inlet":
                v = bc["velocity"]
                lb.set_BC(i,BC_FLOW.inlet)
                lb.set_v_BC_value(i,v)
            elif bc["type"]=="outlet":
                rho = bc["rho"]
                lb.set_BC(i,BC_FLOW.outlet)
                lb.set_rho_BC_value(i,rho)
            elif bc["type"]=="wall":
                lb.set_BC(i,BC_FLOW.wall)
    print("Boundary conditions set.")
    # load all geometry information in geo_dict dictionary name:(surface array, border array)
    print("Geometry loading...")
    geo_dict = {}
    geo_infos = config.get("geometry")
    if geo_infos:
        for geo_name in geo_infos:
            shape = geo_infos[geo_name]["type"]
            if shape == "circle":        
                from LBM.GEO.G3D import Mesh3D
                m2d = Mesh3D(x,y,dx)
                center = geo_infos[geo_name]["center"]
                radius = geo_infos[geo_name]["radius"]
                m2d.CreateMesh2DCircle(center[0],center[1],radius)
                s,l = m2d.export_numpy()
                geo_dict[geo_name] = (s,l)
            elif shape == "rectangle":
                from LBM.GEO.G3D import Mesh3D
                m2d = Mesh3D(x,y,dx)
                point1 = geo_infos[geo_name]["point1"]
                point2 = geo_infos[geo_name]["point2"]
                m2d.CreateMesh2DRectangle(point1[0],point1[1],point2[0],point2[1])
                s,l = m2d.export_numpy()
                geo_dict[geo_name] = (s,l)
            elif shape == "stl":
                # read stl
                path = geo_infos[geo_name].get("path")
                print(f"STL {path} loading...")
                trans = geo_infos[geo_name].get("translate")
                rot = geo_infos[geo_name].get("rotate")
                scale = geo_infos[geo_name].get("scale")
                mesh,grid = lb.voxel_stl(path,scale,trans,rot)
                print(f"STL {path} loaded.")
                geo_dict[geo_name] = (mesh,mesh)


    # set initial condition including setting soid phase
    print("Initial conditions setting...")
    initialCondition = config["initialCondition"]
    ## solid phase
    solidIC = initialCondition.get("solid")
    if solidIC:
        for solid in solidIC.values():
            if isPoro:
                eps = solid["porosity"]
                if solid["zone"]=="ALL":
                    s = np.zeros((lb.nx,lb.ny,lb.nz))
                    for geo_name,geo_data in geo_dict.items():
                        s += geo_data[0]
                    s = (1-eps)*s
                    lb.init_field(lb.solid,s)
                elif type(solid["zone"]) is ruamel.yaml.comments.CommentedSeq: # []
                    s = np.zeros((lb.nx,lb.ny,lb.nz))
                    for zoneName in solid["zone"]:
                        s += geo_dict[zoneName][0]
                    s = (1-eps)*s
                    lb.init_field(lb.solid,s)
                elif type(solid["zone"]) is str:
                    s = geo_dict[solid["zone"]][0]
                    s = (1-eps)*s
                    lb.init_field(lb.solid,s)
            else:
                if solid["zone"]=="ALL":
                    s = np.zeros((lb.nx,lb.ny,lb.nz))
                    for geo_name,geo_data in geo_dict.items():
                        s += geo_data[0]
                    s = (s>0)*1
                    lb.init_field(lb.solid,s)
                elif type(solid["zone"]) is ruamel.yaml.comments.CommentedSeq: # []
                    s = np.zeros((lb.nx,lb.ny,lb.nz))
                    for zoneName in solid["zone"]:
                        s += geo_dict[zoneName][0]
                    s = (s>0)*1
                    lb.init_field(lb.solid,s)
                elif type(solid["zone"]) is str:
                    s = geo_dict[solid["zone"]][0]
                    s = (s>0)*1
                    lb.init_field(lb.solid,s)
    print("Initial conditions set.")
    print("LBM initializing...")
    lb.init_simulation()
    print("LBM initializd.")
    # time control of simulation 
    print("Time control and path setting...")
    lb.tLattice = int(config["timeControl"]["startTime"]/lb.dt)
    endTimeLattice = config["timeControl"]["endTime"]/lb.dt
    printInterval = int(config["outputControl"]["log"]["interval"]/lb.dt)
    exportInterval = int(config["outputControl"]["vtk"]["interval"]/lb.dt)
    exportPath = config["outputControl"]["vtk"]["path"]
    if not exportPath:
        exportPath = "output"
    # clear past result
    if config["outputControl"]["vtk"]["clear"]:
        import shutil
        shutil.rmtree(exportPath,ignore_errors=True)
    lb.exportPath = exportPath
    lb.PVD.exportPath = exportPath
    os.makedirs(exportPath,exist_ok=True)
    snapshotInterval = int(config["outputControl"]["snapshot"]["interval"]/lb.dt)
    snapshotPath = config["outputControl"]["snapshot"]["path"]
    lb.snapshotPath = snapshotPath
    preTime = time.time()
    latticeUpdateBetweenLog = lb.nx*lb.ny*lb.nz*printInterval    
    print("Time control and path set.")
    print("LBM running...")
    last_print_time = time.time()
    while lb.tLattice<=endTimeLattice:
        if lb.tLattice % printInterval==0:
            calTime = time.time()-preTime
            preTime = time.time()
            if calTime ==0:
                MLUPS = 0
            else:
                MLUPS = latticeUpdateBetweenLog/calTime/1e6
            print(f"\rExecution time:{calTime:.2f} s , Collapsed time:{(time.time()-startTime):.2f} s, MLUPS = {MLUPS:.2f}")
            print(lb.log_info())
        if lb.tLattice % exportInterval==0:
            lb.export_VTK()
        if lb.tLattice%snapshotInterval==0:
            lb.export_snapshot(config)
        if time.time()-last_print_time>10:
            last_print_time = time.time()
            print(f"\r{lb.tLattice}",end="",flush=True)
        lb.step()
    print("LBM finished.")
    