from ._core import LBM3D_BASE
from ..util.flag import *
class LBM3D_INFO(LBM3D_BASE):
    def __str__(self):
        return self.description()

    def __repr__(self):
        return self.__str__()

    def description(self):
        des = "\n------------------------------------------\n"        
        des += self.description_basic()
        des += "\n"
        des += self.description_flow()
        des += "\n"
        des += self.description_porous()
        des += "\n"
        des += self.description_thermal()
        des += "\n"
        des += self.description_chemical()
        des += "\n"
        des += self.description_BC()
        des += "------------------------------------------\n\n"
        return des
    def description_flow(self):
        des = "Flow properties : \n"
        des += f"    viscosity model : "
        if self.viscosity_model==VISCOSITY_MODEL.CONSTANT:
            des += f"constant : {self.visco}"
        elif self.viscosity_model==VISCOSITY_MODEL.SUTHERLAND:
            des += f"sutherland : As={self.sutherland_coef[0]}, Ts={self.sutherland_coef[1]}"
        elif self.viscosity_model==VISCOSITY_MODEL.MIXTURE:
            des += f"weighted average of components."
        else:
            des += "Not Valid"
        des += "\n"
        return des
    def description_porous(self):
        des = "Porous media model : "
        if self.PORO:
            des += f"{self.poro_model.name}"
            if self.poro_model==PORO_MODEL.SPHERICAL:
                des +=""
            elif self.poro_model==PORO_MODEL.DARCY:
                des +=""
            elif self.poro_model==PORO_MODEL.DARCYFORCHHEIMER:
                des +=""
            elif self.poro_model==PORO_MODEL.ERGUN:
                des +=""
        else:
            des += "Not activated."
        return des+"\n"
    def description_thermal(self):
        des = "Thermal module: "
        if self.TEMPERATURE:
            des += "\n"
            des += self.TF.__str__()
            des += self.TS.__str__()
        else:
            des += " Not activated.\n"
        return des
    def description_chemical(self)->str:
        des = "Chemical module: "
        if self.CHEMISTRY:
            des+="\n  The species involved are: \n"
            for specie in self.species:
                des += "    " + specie.__str__()
            des += self.reactions.__str__()
        else:
            des+=" Not activated.\n"
        return des
    def description_basic(self)->str:
        des = "Basic information of the LBM simulation:\n"
        des += " name : " + self.name+f"\n Size : {self.nx} x {self.ny} x {self.nz}\n dx : {self.dx}\n dt : {self.dt}\n"
        des += "Modules activated :\n"
        des += " -flow field\n"
        if self.PORO:
            des+=" -porous medium\n"
        if self.TEMPERATURE:
            des+=" -thermal transfer\n"
        if self.RADIATION:
            des+=" -radiation\n"
        if self.CHEMISTRY:
            des+=" -chemical reaction\n"        
        return des

    def description_BC(self)->str:
        des="Boundary condition model : "+ BC_MODEL(self.boundary_condition_model).name+"\n"
        des+="The boundary conditions of the flow field are set to :\n"
        sideName = ["left","right","bottom","top","forward","back"]
        for i in range(6):
            # des+=self.bc[i])
            if self.bc[i]==BC_FLOW.periodic:
                des+="    "+sideName[i]+": PERIODIC\n"
            if self.bc[i]==BC_FLOW.wall:
                des+="    "+sideName[i]+": WALL\n"
            if self.bc[i]==BC_FLOW.inlet:
                des+="    "+sideName[i]+": INLET\n"
            if self.bc[i]==BC_FLOW.outlet:
                des+="    "+sideName[i]+": OUTLET\n"
            if self.bc[i]==BC_FLOW.symmetric:
                des+="    "+sideName[i]+": SYMMETRIC\n"
            if self.bc[i]==BC_FLOW.inlet_flow:
                des+="    "+sideName[i]+": INLET FLOW\n"
        return des
    def check_valid(self):
        '''检查当前LBM是否有效'''
        # 检查惰性物质设置
        if self.CHEMISTRY:
            inert_count = sum(1 for specie in self.species if not specie.FIX and specie.isInert)
            if inert_count != 1:
                raise ValueError(f"Invalid LBM configuration: More than one inert specie found. Only one inert specie is allowed. {inert_count} inert species were found.")
