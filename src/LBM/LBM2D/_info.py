from ._core import LBM2D_BASE
from ..util.flag import *
class LBM2D_INFO(LBM2D_BASE):
    
    def __str__(self):
        return self.description()

    def __repr__(self):
        return self.__str__()

    def description(self):
        des = "\n------------------------------------------\n"        
        des += self.description_basic()+"\n"
        des += self.description_flow()+"\n"
        des += self.description_porous()+"\n"
        des += self.description_thermal()+"\n"
        des += self.description_chemical()+"\n"
        des += self.description_BC()+"\n"
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
        des = ""
        if self.PORO:
            des += f"Porous media model : {self.poro_model}"
            if self.poro_model==PORO_MODEL.SPHERICAL:
                des +=""
            elif self.poro_model==PORO_MODEL.DARCY:
                des +=""
            elif self.poro_model==PORO_MODEL.DARCYFORCHHEIMER:
                des +=""
            elif self.poro_model==PORO_MODEL.ERGUN:
                des +=""
        return des+"\n"
    def description_thermal(self):
        des = ""
        if self.TEMPERATURE:
            des += self.TF.__str__()
            des += self.TS.__str__()
        return des
    def description_chemical(self)->str:
        des = "Chemical module: \n"
        if self.CHEMISTRY:
            des+="  The species involved are: \n"
            for specie in self.species:
                des += "    " + specie.__str__()
            des += self.reactions.__str__()
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
        sideName = ["left","right","bottom","top"]
        for i in range(4):
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
    def log_info(self):
        """生成当前步简要日志字符串。

        内容包括 `t(LU)`、`max|v|`，在启用温度时附带 `T_min/T_max (K)`。
        """
        p = f"    t(s):{self.tLattice*self.dt} t(LU)={self.tLattice} : Max velocity magnitude (LU) : {self.get_max_v():.7f}, "
        if self.TEMPERATURE:
            p +=f"Min temperature: {self.get_min_T():.7f} K, Max temperature : {self.get_max_T():.7f}"
        return p