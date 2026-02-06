import numpy as np
import logging
import ruamel.yaml
import ruamel.yaml.comments
import taichi as ti
import os
import time
import ruamel
import logging
import sys

from LBM.LBM2D._chemical import Specie
from .util.flag import *
from .util.constant import MOLEMASS

def application(config:ruamel.yaml.comments.CommentedMap,logger:logging.Logger):
    debug = logger.level==logging.DEBUG
    startTime = time.time()
    dim = 2
    ARCH = "cpu"
    name = ""
    logger.info("Loading basic...")
    basic = config.get("basic")
    if basic is None:
        logger.error("basic missing")
        # raise ValueError("Basic is none")  
    else:
        for key,value in basic.items():
            logger.info(f"{key} : {value}")

        dim = basic.get("dimension")
        if dim==2:
            from .LBM2D import LBM2DSolver as lbm
        elif dim==3:
            from .LBM3D import LBM3DSolver as lbm
        else:
            logger.error("The dimension value in the configuration file is invalid. Please set it to 2 or 3 depending on your case.")
            # raise ValueError("dim")
        
        ARCH = basic.get("arch","cpu")
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
                logger.warning(f"ARCH {ARCH} not valid. Use cpu by default.")
                ARCH = "cpu"
        except:
            ti.init(arch=ti.cpu)
            logger.warning(f"ARCH {ARCH} not valid. Use cpu by default.")
            ARCH = "cpu"

        name = basic.get("name")
    
    logger.info("Loading spaceControl...")  
    spaceControl = config.get("spaceControl")
    if spaceControl is None:
        logger.error("spaceControl missing.")
    else:
        geometry = spaceControl.get("geometry")
        if geometry is None:
            logger.error("spaceControl.geometry missing.")
        else:
            x = geometry[0]
            y = geometry[1]
            if dim == 3:
                z = geometry[2]
            else:
                z = 1
        dx = spaceControl.get("DX")
        if dx is None:
            logger.error("spaceControl.DX missing")
    
    logger.info("Loading timeControl.dt...")
    timeControl = config.get("timeControl")
    if timeControl is None:
        logger.error("timeControl missing.")
    else:
        dt = timeControl["DT"]
        if dt is None:
            logger.error("timeControl.DT missing.")
        
    logger.info("Loading module...")
    module = config.get("module")
    if module is None:
        logger.error("module missing")
    else:
        isThermal = module.get("temperature")
        isPoro = module.get("porous_media")
        isChemical = module.get("chemistry")
        isRadiation = module.get("radiation")
    
    # initial setting
    lb = lbm(x,y,z,dx,dt,name,isThermal,isChemical,isPoro,isRadiation)
    logger.info("LBM created")
    # viscosity
    logger.info("Loading flowProperties...")
    flowProperties = config.get("flowProperties")
    if flowProperties is None:
        logger.error("flowProperties missing.")
    else:
        viscosity = flowProperties.get("viscosity")
        viscosityType = viscosity.get("type")
        if viscosityType == "constant":
            nu = viscosity.get("value")
            lb.set_viscosity(nu)
        elif viscosityType == "sutherland":
            As = viscosity.get("As")
            Ts = viscosity.get("Ts")
            lb.set_viscosity_sutherland(As,Ts)
        elif viscosityType == "mixture":
            lb.set_viscosity_mixture()
    
    # thermal
    logger.info("Loading thermalPropertiesFluid...")
    thermalPropertiesFluid = config.get("thermalPropertiesFluid")
    if isThermal and (thermalPropertiesFluid is not None):
        normalize = thermalPropertiesFluid.get("normalize")
        if normalize is not None:
            Trange = normalize.get("Trange",[0,1])
            lb.set_fluid_Trange(Trange)
        thermalDiff = thermalPropertiesFluid.get("thermalDiff")
        if thermalDiff is not None:
            thermalDiffType = thermalDiff.get("type")
            if thermalDiffType=="constant":
                diff = thermalDiff.get("value")
                lb.set_fluid_thermal_diff(diff)
            elif thermalDiffType=="Prandtl":
                Pr = thermalDiff.get("Pr")
                lb.set_fluid_Prandtl(Pr)
            elif thermalDiffType=="derived":
                lb.set_fluid_thermal_diff_derived()
        conductivity = thermalPropertiesFluid.get("conductivity")
        if conductivity is not None:
            conductivityType = conductivity.get("type")
            if conductivityType == "constant":
                lamb = conductivity.get("value")
                lb.set_fluid_conductivity(lamb)
            elif conductivityType == "polynomial":
                data = conductivity.get("data")
                lb.set_fluid_conductivity_poly(data)
            elif conductivityType == "mixture":
                lb.set_fluid_conductivity_mixture()
            else:
                logger.warning(f"Conductivity type {conductivityType} of thermalPropertiesFluid not valid. The valid type : constant|Prandtl|polynomial|mixture")
        capacity = thermalPropertiesFluid.get("capacity")   
        if capacity is not None:
            capacityType = capacity.get("type")
            if capacityType == "constant":
                value = capacity.get("value")
                lb.set_fluid_capacity(value)
            elif capacityType =="polynomial":
                data = capacity.get("data")
                lb.set_fluid_capacity_poly(data)
            elif capacityType == "NASA7":
                Trange = capacity.get("Trange")
                data = capacity.get("data")
                lb.set_fluid_capacity_NASA7(Trange,data)
            elif capacityType == "mixture":
                lb.set_fluid_capacity_mixture()
            else:
                logger.warning(f"Capacity type {capacityType} of thermalPropertiesFluid not valid. The valid type : constant|polynomial|NASA7|mixture")
        
    logger.info("Loading thermalPropertiesSolid...")
    thermalPropertiesSolid = config.get("thermalPropertiesSolid")
    if isThermal and (thermalPropertiesSolid is not None):
        normalize = thermalPropertiesSolid.get("normalize")
        if normalize is not None:
            Trange = normalize.get("Trange",[0,1])
            lb.set_solid_Trange(Trange)
        thermalDiff = thermalPropertiesSolid.get("thermalDiff")
        if thermalDiff is not None:
            thermalDiffType = thermalDiff.get("type")
            if thermalDiffType=="constant":
                diff = thermalDiff.get("value")
                lb.set_solid_thermal_diff(diff)
            elif thermalDiffType=="derived":
                lb.set_solid_thermal_diff_derived()
        conductivity = thermalPropertiesSolid.get("conductivity")
        if conductivity is not None:
            conductivityType = conductivity.get("type")
            if conductivityType == "constant":
                lamb = conductivity.get("value")
                lb.set_solid_conductivity(lamb)
            elif conductivityType == "polynomial":
                data = conductivity.get("data")
                lb.set_solid_conductivity_poly(data)
            elif conductivityType == "mixture":
                lb.set_solid_conductivity_mixture()
            else:
                logger.warning(f"Conductivity type {conductivityType} of thermalPropertiesSolid not valid. The valid type : constant|polynomial|mixture")
        capacity = thermalPropertiesSolid.get("capacity")   
        if capacity is not None:
            capacityType = capacity.get("type")
            if capacityType == "constant":
                value = capacity.get("value")
                lb.set_solid_capacity(value)
            elif capacityType =="polynomial":
                data = capacity.get("data")
                lb.set_solid_capacity_poly(data)
            elif capacityType == "NASA7":
                Trange = capacity.get("Trange")
                data = capacity.get("data")
                lb.set_solid_capacity_NASA7(Trange,data)
            elif capacityType == "mixture":
                lb.set_solid_capacity_mixture()
            else:
                logger.warning(f"Capacity type {capacityType} of thermalPropertiesSolid not valid. The valid type : constant|polynomial|NASA7|mixture")
    
    logger.info("Loading radiationProperties...")
    radiationProperties = config.get("radiationProperties")
    if isRadiation and radiationProperties is not None:
        logger.info("Radiation module loading...")
        radiationType = radiationProperties.get("type")
        if radiationType=="mean_temp":
            Tambient = radiationProperties.get("Tambient",300)
            lb.set_radiation(RADIATION_MODEL.SURFACE_UNIFORM,Tambient)
    # chemistry
    logger.info("Loading chemicalProperties...")
    chemicalProperties = config.get("chemicalProperties")
    if isChemical and chemicalProperties is not None:
        logger.info("Chemistry module loading...")
        chemicalType = chemicalProperties.get("type")
        if chemicalType == "cantera":
            # load chemical mechanism
            logger.info("Cantera loading...")
            cantera_file = config["chemicalProperties"]["canteraFile"]
            lb.load_cantera(cantera_file)
            logger.info("Cantera loaded.")
        elif chemicalType == "input":
            species = chemicalProperties.get("species")
            if species is not None:
                for specie in species:
                    logger.info(f"Specie {specie} loading...")
                    composition = species[specie].get("composition")
                    mmass = 0.0
                    if composition is not None:
                        for key,value in composition.items():
                            mmass += value*MOLEMASS[key]
                    state = species[specie].get("state","fluid")
                    if state =="fluid":
                        lb.set_specie(specie,False,mmass)
                        viscosityProperty = species[specie].get("viscosity")
                        if viscosityProperty is not None:
                            viscosityType = viscosityProperty.get("type")
                            if viscosityType == "constant":
                                value = viscosityProperty.get("value",1e-5)
                                lb.set_specie_viscosity(specie,value)
                            elif viscosityType == "sutherland":
                                As = viscosityProperty.get("As",1e-6)
                                Ts = viscosityProperty.get("Ts",70)
                                lb.set_specie_viscosity_sutherland(specie,[As,Ts])
                            else:
                                logger.warning(f"Specie {specie} viscosity type {viscosityType} not valid. The valid value : constant|sutherland")
                        diffProperty = species[specie].get("diffusitivity")
                        if diffProperty is not None:
                            diffType = diffProperty.get("type")
                            if diffType=="constant":
                                diff = diffProperty.get("value")
                                lb.set_specie_diff(specie,diff)
                            elif diffType=="Schmidt":
                                Sch = diffProperty.get("Sc")
                                lb.set_specie_diff_Schmidt(specie,Sch)
                            else:
                                logger.warning(f"Specie {specie} diffusitivity type {diffType} not valid. The valid value : constant|Schmidt")
                        if isThermal:
                            thermodynamicProperty = species[specie].get("thermodynamic")
                            if thermodynamicProperty is not None:
                                thermodynamicType = thermodynamicProperty.get("type")
                                if thermodynamicType=="constant":
                                    capa = thermodynamicProperty.get("capacity")
                                    lb.set_specie_capacity(specie,capa)
                                    enthalpy = thermodynamicProperty.get("enthalpy")
                                    lb.set_specie_enthalpy(specie,enthalpy)
                                elif thermodynamicType=="NASA7":
                                    Trange = thermodynamicProperty.get("Trange")
                                    data = thermodynamicProperty.get("data")
                                    lb.set_specie_NASA7(specie,Trange,data)
                                else: 
                                    logger.warning("ss")
                            condProperty = species[specie].get("conductivity")
                            if condProperty is not None:
                                condType = condProperty.get("type")
                                if condType=="constant":
                                    cond = condProperty.get("value")
                                    lb.set_specie_conductivity(specie,cond)
                                elif condType=="polynomial":
                                    poly = condProperty.get("data")
                                    lb.set_specie_conductivity_poly(specie,poly)
                    elif state == "solid":
                        lb.set_specie(specie,True)
                        if isThermal:
                            thermodynamicProperty = species[specie].get("thermodynamic")
                            if thermodynamicProperty is not None:
                                thermodynamicType = thermodynamicProperty.get("type")
                                if thermodynamicType=="constant":
                                    capa = thermodynamicProperty.get("capacity")
                                    lb.set_specie_capacity(specie,capa)
                                    enthalpy = thermodynamicProperty.get("enthalpy")
                                    lb.set_specie_enthalpy(specie,enthalpy)
                                elif thermodynamicType=="NASA7":
                                    Trange = thermodynamicProperty.get("Trange")
                                    data = thermodynamicProperty.get("data")
                                    lb.set_specie_NASA7(specie,Trange,data)
                                else: 
                                    logger.warning("ss")
                            condProperty = species[specie].get("conductivity")
                            if condProperty is not None:
                                condType = condProperty.get("type")
                                if condType=="constant":
                                    cond = condProperty.get("value")
                                    lb.set_specie_conductivity(specie,cond)
                                elif condType=="polynomial":
                                    poly = condProperty.get("data")
                                    lb.set_specie_conductivity_poly(specie,poly)
                    else:
                        logger.warning(f"Specie state {state} not valid. The valid value : fluid|solid")
            reactions = chemicalProperties.get("reactions")
            if reactions is not None:
                for reaction in reactions:
                    coefRate = reaction["rate-constant"]
                    lb.add_reaction(reaction["equation"],coefRate["A"],coefRate["Ea"],coefRate["b"])

        else:
            logger.warning(f"Chemical type {chemicalType} of chemicalProperties not valid. The valid type : cantera|input")

    # boundary condition
    logger.info("Loading boundary condition...")
    boundaryName = ["left","right","down","up","forward","back"]
    boundaryNum = 4
    if dim==3:
        boundaryNum=6
    for i in range(boundaryNum):
        bc =config["boundaryCondition"][boundaryName[i]] 
        # bc of flow field
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
        elif bc_flow["type"]=="inlet_flowrate":
            F = bc_flow["flowrate"]
            lb.set_BC(i,BC_FLOW.inlet_flow)
            lb.set_flow_BC_value(i,F)
            rho = bc_flow.get("rho",1)
            if rho:
                lb.set_rho_BC_value(i,rho)
        elif bc_flow["type"]=="wall":
            lb.set_BC(i,BC_FLOW.wall)
        ## bc of temperature field
        if isThermal:
            bcThermalFluid = bc.get("thermalFluid")
            if bcThermalFluid is not None:
                bcType = bcThermalFluid.get("type")
                if bcType=="fixedValue":
                    value = bcThermalFluid.get("value",0.0)
                    lb.set_TF_BC(i,BC.fixedValue)
                    lb.set_TF_BC_value(i,value)
                elif bcType=="zeroGradient":
                    lb.set_TF_BC(i,BC.zeroGradient)
                elif bcType=="periodic":
                    lb.set_TF_BC(i,BC.periodic)
                else:
                    logger.warning(f"Boundary condition type {bcType} of thermalFluid not valid. The valid consitions : periodic|fixedValue|zeroGradient")
            bcThermalSolid = bc.get("thermalSolid")
            if bcThermalSolid is not None:
                bcType = bcThermalSolid.get("type")
                if bcType=="fixedValue":
                    value = bcThermalSolid.get("value",0)
                    lb.set_TS_BC(i,BC.fixedValue)
                    lb.set_TS_BC_value(i,value)
                elif bcType=="zeroGradient":
                    lb.set_TS_BC(i,BC.zeroGradient)
                elif bcType=="periodic":
                    lb.set_TS_BC(i,BC.periodic)
                elif bcType=="none":
                    pass
                else:
                    logger.warning(f"Boundary condition type {bcType} of thermalSolid not valid. The valid consitions : periodic|fixedValue|zeroGradient")        
        ## bc of specie field
        if isChemical:
            bcChemical = bc.get("chemical")
            if bcChemical is not None:
                bcType = bcChemical.get("type")
                if bcType=="fixedValue":
                    for specie in lb.specieName:
                        bcSpecie = bcChemical.get(specie,0)
                        lb.set_specie_BC(specie,i,BC.fixedValue)
                        lb.set_specie_BC_value(specie,i,bcSpecie)
                elif bcType=="zeroGradient":
                    lb.set_species_BC(i,BC.zeroGradient)
                elif bcType=="periodic":
                    lb.set_species_BC(i,BC.periodic)
                else:
                    logger.warning(f"Boundary condition type {bcType} of thermalFluid not valid. The valid consitions : periodic|fixedValue|zeroGradient")
    

    # load all geometry information in geo_dict dictionary name:(surface array, border array)
    logger.info("Loading geometry...")
    geo_dict = {}
    geo_infos = config.get("geometry")
    if geo_infos is None:
        logger.info("No Geometry specified.")
    else:
        for geo_name in geo_infos:
            shape = geo_infos[geo_name]["type"]
            path = ""
            translate = [0,0,0]
            rotate = [0,0,0]
            scale = [1,1,1]
            if shape == "sphere":
                from .GEO.STL import StlGenerator
                stl_generator = StlGenerator(logger)
                path = stl_generator.create_sphere()
                translate = geo_infos[geo_name].get("center",[0,0,0])
                scale = geo_infos[geo_name].get("radius",[0.5,0.5,0.5])*2
            elif shape == "cylinder":
                from .GEO.STL import StlGenerator
                stl_generator = StlGenerator(logger)
                path = stl_generator.create_cylinder()
                height = geo_infos[geo_name].get("height",1)
                center = geo_infos[geo_name].get("center",[0,0,0])
                radius = geo_infos[geo_name].get("radius",0.5)
                axis = geo_infos[geo_name].get("axis",[0,0,1])
                from .util.math import vectors_to_euler
                rotate = vectors_to_euler([0,0,1],axis,degrees = True)
                translate = center
                scale = [2*radius,2*radius,height]
            elif shape == "cone":
                from .GEO.STL import StlGenerator
                stl_generator = StlGenerator(logger)
                path = stl_generator.create_cone()
                center = geo_infos[geo_name].get("center",[0,0,0])
                height = geo_infos[geo_name].get("height",1)
                radius = geo_infos[geo_name].get("radius",0.5)
                axis = geo_infos[geo_name].get("axis",[0,0,1])
                from .util.math import vectors_to_euler
                rotate = vectors_to_euler([0,0,1],axis,degrees = True)
                translate = center
                scale = [2*radius,2*radius,height]
            elif shape == "box":
                from .GEO.STL import StlGenerator
                stl_generator = StlGenerator(logger)
                path = stl_generator.create_box()
                scale = geo_infos[geo_name].get("size",[1,1,1])
                center = geo_infos[geo_name].get("center",[0,0,0])
                translate = center
            elif shape == "stl":
                # read stl
                path = geo_infos[geo_name].get("path")
                translate = geo_infos[geo_name].get("translate")
                rotate = geo_infos[geo_name].get("rotate")
                scale = geo_infos[geo_name].get("scale",[1,1,1])
            else:
                logger.warning(f"Geometry {shape} not valid.")
            mesh,surface = lb.load_stl(path,scale,translate,rotate,logger=logger)
            if np.shape(mesh)[0]!=0:
                geo_dict[geo_name]=mesh,surface


    # set initial condition including setting soid phase
    logger.info("Loading initialCondition...")
    initialCondition = config["initialCondition"]
    if initialCondition is None:
        logger.error("initialConditon missing.")
    else:
        ## flow
        flowIC = initialCondition.get("flow")
        if flowIC is None:
            logger.warning("initialCondition.flow not specified.")
        else:
            rho = flowIC.get("rho",1.0)
            velocity = flowIC.get("velocity",[0,0,0])
            lb.init_field(lb.rho,rho)
            lb.init_field3(lb.v,float(velocity[0]),float(velocity[1]),float(velocity[2]))
        ## thermal
        if isThermal:
            thermalIC = initialCondition.get("thermal")        
            if thermalIC is None:
                logger.warning("initialCondition.thermal not specified.")
            else:
                TF = thermalIC.get("TF")
                TS = thermalIC.get("TS")
                T = thermalIC.get("T",0)
                if TF is not None:
                    lb.init_field(lb.TF.S,lb.TF.get_normalized_value(TF))
                else:
                    lb.init_field(lb.TF.S,lb.TF.get_normalized_value(T))
                if TS is not None:
                    lb.init_field(lb.TS.S,lb.TS.get_normalized_value(TS))
                else:
                    lb.init_field(lb.TS.S,lb.TS.get_normalized_value(T))
        ## chemical
        if isChemical:
            chemicalIC = initialCondition.get("chemical")        
            if chemicalIC is None:
                logger.warning("initialCondition.chemical not specified.")
            else:
                for specie in chemicalIC:
                    zone = chemicalIC[specie].get("zone")
                    value = chemicalIC[specie].get("value",0)
                    s = np.zeros_like(lb.rho.to_numpy())
                    if zone == "ALL":
                        lb.init_specie(specie,float(value))
                    elif type(zone) is ruamel.yaml.comments.CommentedSeq: # []
                        for zoneName in zone:
                            s += value*geo_dict[zoneName][0]
                        lb.init_specie(specie,s)    
                    elif type(zone) is str:
                        s += value*geo_dict[zone][0]
                        lb.init_specie(specie,s)
            
        ## solid phase
        solidIC = initialCondition.get("solid")
        if solidIC is None:
            logger.warning("initialCondition.solid not specified.")
        else:
            for solid in solidIC.values():
                solidType = solid.get("type")
                s = np.zeros((lb.nx,lb.ny,lb.nz))
                if isPoro and solidType=="poro":
                    eps = solid["porosity"]
                    zone =solid.get("zone")
                    rhos = solid.get("rho",1)
                    exchangeSurface = solid.get("exchangeSurface",1)
                    exchangeCoef = solid.get("exchangeCoef",1)
                    porousModel = solid.get("porousModel")
                    emis = solid.get("emisssivity",1.0)
                    border = np.zeros((lb.nx,lb.ny,lb.nz))
                    if zone =="ALL":
                        for geo_name,geo_data in geo_dict.items():
                            s += geo_data[0]
                            border +=geo_data[1]
                    elif type(zone) is ruamel.yaml.comments.CommentedSeq: # []
                        for zoneName in solid["zone"]:
                            s += geo_dict[zoneName][0]
                            border +=geo_dict[zoneName][1]
                    elif type(zone) is str:
                        s = geo_dict[solid["zone"]][0]
                        border += geo_dict[solid["zone"]][1] 
                    s = (s>0)*(1.0)
                    solid_fraction = (1-eps)*s
                    lb.add_solid(solid_fraction)
                    lb.add_rho_solid(rhos*s)
                    if isThermal:
                        lb.set_heat_exchange_surface(exchangeSurface*s)
                        lb.set_heat_exchange_coef(exchangeCoef*s)
                    if isRadiation:
                        lb.init_field(lb.TS.radiation_surface,border)
                        lb.init_field(lb.TS.emissivity,border*emis)
                    if porousModel=="darcy":
                        darcy = solid.get("darcy")
                        lb.set_poro_Darcy(s,darcy)
                    elif porousModel == "darcyForchheimer":
                        darcy = solid.get("darcy")
                        forchheimer = solid.get("forchheimer")
                        lb.set_poro_Darcy_Forchheimer(s,darcy,forchheimer)
                    elif porousModel == "ergun":
                        pass
                        
                elif solidType=="concrete":
                    if solid["zone"]=="ALL":
                        for geo_name,geo_data in geo_dict.items():
                            s += geo_data[0]
                    elif type(solid["zone"]) is ruamel.yaml.comments.CommentedSeq: # []
                        for zoneName in solid["zone"]:
                            s += geo_dict[zoneName][0]
                    elif type(solid["zone"]) is str:
                        s = geo_dict[solid["zone"]][0]
                    s = (s>0)*(1.0)
                    rhos = solid.get("rho",1)
                    lb.add_solid(s)
                    lb.add_rho_solid(rhos*s)
                elif solidType=="substract":
                    if solid["zone"]=="ALL":
                        for geo_name,geo_data in geo_dict.items():
                            s += geo_data[0]
                        s = (s>0)*(-1.0)
                    elif type(solid["zone"]) is ruamel.yaml.comments.CommentedSeq: # []
                        for zoneName in solid["zone"]:
                            s += geo_dict[zoneName][0]
                        s = (s>0)*(-1.0)
                    elif type(solid["zone"]) is str:
                        s = geo_dict[solid["zone"]][0]
                        s = (s>0)*(-1.0)
                    lb.add_solid(s)
    
    # time control of simulation 
    logger.info("Loading timeControl...")
    timeControl = config.get("timeControl")
    if timeControl is None:
        logger.error("timeControl missing.")
    else:
        simulationStartTime = timeControl.get("startTime",0)
        # if startTime is None:
        #     startTime = 0
        lb.tLattice = int(simulationStartTime/lb.dt)
        endTime = timeControl.get("endTime",0)
        endTimeLattice = endTime/lb.dt
    
    # output
    logger.info("Loading outputControl...")
    outputControl = config.get("outputControl")
    if outputControl is None:
        logger.error("outputControl missing.")
    else:
        logControl = outputControl.get("log")
        printInterval = logControl["interval"]/lb.dt
        exportControl = outputControl.get("vtk")
        interval = exportControl.get("interval",100)
        exportInterval = int(interval/lb.dt)
        exportPath = exportControl.get("path")
        if not exportPath:
            exportPath = "output"
        # clear past result
        
        if exportControl.get("clear"):
            import shutil
            shutil.rmtree(exportPath,ignore_errors=True)
        lb.exportPath = exportPath
        lb.PVD.exportPath = exportPath
        snapshot = config["outputControl"]["snapshot"]
        interval = snapshot.get("interval",100)
        snapshotInterval = int(interval/lb.dt)
        snapshotPath = snapshot.get("path")
        lb.snapshotPath = snapshotPath
        preTime = time.time()
        last_print_time = time.time()
        latticeUpdateBetweenLog = lb.nx*lb.ny*lb.nz*printInterval
        debugCheckInterval = 10
        debugControl = config.get("debugSetting")
        if debugControl:
            debugCheckInterval = debugControl.get("interval",10)

    logger.info("Simulation initializing...")
    lb.init_simulation()
    logger.info("Simulation initialized.")
    logger.info(lb.description())
    logger.info("Simulation running...")
    logger.info("(The first step will take long time because of the compilation time)")
    while lb.tLattice<=endTimeLattice:
        if lb.tLattice % printInterval==0:
            calTime = time.time()-preTime
            preTime = time.time()
            if calTime ==0:
                MLUPS = 0
            else:
                MLUPS = latticeUpdateBetweenLog/calTime/1e6
            logger.info(f"Execution time:{calTime:.2f} s , Collapsed time:{(time.time()-startTime):.2f} s, MLUPS = {MLUPS:.2f}")
            logger.info(lb.log_info())
        if lb.tLattice % exportInterval==0:
            lb.export_VTK_pyvista()
        if lb.tLattice%snapshotInterval==0:
            lb.export_snapshot(config)
        if debug:
            lb.check_python()
        if debug and time.time()-last_print_time>debugCheckInterval:
            last_print_time = time.time()
            logger.debug(lb.tLattice)
        lb.step()
    logger.info("LBM finished.")


