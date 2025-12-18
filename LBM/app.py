import numpy as np
import ruamel.yaml
import ruamel.yaml.comments
import taichi as ti
import os
import time
import ruamel
from LBM.GEO.G2D import Mesh2D
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
    isThermal = config["module"]["temperature"]
    isPoro = config["module"]["porous_media"]
    isChemical = config["module"]["chemistry"]
    isRadiation = config["module"]["radiation"]
    name = config["basic"]["name"]
    # initial setting
    lb = LBM2DSolver(x,y,dx,dt,name,isThermal,isPoro,isChemical,isRadiation)
    
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

    # load all geometry information in geo_dict dictionary name:(surface, border)
    geo_dict = {}
    geo_infos = config.get("geometry")
    if geo_infos is not None:
        for geo_name in geo_infos:
            m2d = Mesh2D(x,y,dx)
            shape = geo_infos[geo_name]["type"]
            if shape == "circle":
                center = geo_infos[geo_name]["center"]
                radius = geo_infos[geo_name]["radius"]
                m2d.CreateMesh2DCircle(center[0],center[1],radius)
            elif shape == "rectangle":
                point1 = geo_infos[geo_name]["point1"]
                point2 = geo_infos[geo_name]["point2"]
                m2d.CreateMesh2DRectangle(point1[0],point1[1],point2[0],point2[1])
            s,l = m2d.export_numpy()
            geo_dict[geo_name] = (s,l)

    # set initial condition including setting soid phase
    initialCondition = config["initialCondition"]
    ## solid phase
    solidIC = initialCondition.get("solid")
    if solidIC:
        if isPoro:
            pass
        else:
            if solidIC["zone"]=="ALL":
                s = np.zeros((lb.nx,lb.ny,lb.nz))
                for geo_name,geo_data in geo_dict.items():
                    s += geo_data[0]
                s = (s>0)*1
                lb.init_field(lb.solid,s)
        ## 
    # if config["TYPE"]=="config":
    #     # set initial field
        # solid_np = np.zeros((nx,ny,1))
        # for region in config["SOLID"].values():
        #     m2d = Mesh2D(nx,ny,1)
        #     if region["shape"]=="cylinder":
        #         center = np.array(region["center"])/dx
        #         height = region["height"]/dx
        #         radius = region["radius"]/dx
        #         up = np.array(region["up"])
        #         if region["type"]=="filled":
        #             m2d.CreateMesh3D_Cylinder_Integer(center,up,radius,height)
        #             V = m2d.V.to_numpy()
        #             if region["mode"]=="overlay":
        #                 solid_np = V
        #             elif region["mode"]=="addition":
        #                 solid_np += V
        #         elif region["type"]=="porous":
        #             m3d.CreateMesh3D_Cylinder_Decimal(center,region["up"],radius,height)
        #             V = m3d.V.to_numpy()
        #             if region["mode"]=="overlay":
        #                 solid_np = V
        #             elif region["mode"]=="addition":
        #                 solid_np += V
        # lb.init_field(lb.solid,solid_np)
    #     lb.init_field3(lb.v,config["FLOW"]["initialCondition"][0],config["FLOW"]["initialCondition"][1],config["FLOW"]["initialCondition"][2])
    # elif config["TYPE"]=="snapshot":
    #     # set all fields
    #     lb.init_field(lb.v,np.array(config["FLOW"]["v"]))
    #     lb.init_field(lb.f,np.array(config["FLOW"]["f"]))
    #     lb.init_field(lb.solid,np.array(config["SOLID"]))
    #     if isThermal:
    #         pass
    #     if isChemical:
    #         pass
    #     if isRadiation:
    #         pass
    # # print(lb.solid)



    lb.init_simulation()
    # time control of simulation 
    lb.tLattice = int(config["timeControl"]["startTime"]/lb.dt)
    endTimeLattice = config["timeControl"]["endTime"]/lb.dt
    printInterval = int(config["outputControl"]["log"]["interval"]/lb.dt)
    exportInterval = int(config["outputControl"]["vtk"]["interval"]/lb.dt)
    print(exportInterval,printInterval)
    exportPath = config["outputControl"]["vtk"]["path"]
    if not exportPath:
        exportPath = "output"
    # clear past result
    if config["outputControl"]["vtk"]["clear"]:
        import shutil
        shutil.rmtree(exportPath,ignore_errors=True)
    lb.path = exportPath
    lb.PVD.path = exportPath
    os.makedirs(exportPath,exist_ok=True)
    snapshotInterval = int(config["outputControl"]["snapshot"]["interval"]/lb.dt)
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
        # if lb.tLattice%snapshotInterval==0:
        #     lb.export_snapshot(config)
        lb.step()
def application_3D(config):
    # Implementation for 3D application
    from LBM.LBM3D import LBM3DSolver
    print("Running 3D application with config:")
    print(config)