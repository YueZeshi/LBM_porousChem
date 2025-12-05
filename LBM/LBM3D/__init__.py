import taichi as ti
from ._core import LBM3D_BASE
from ._evolution import LBM3D_EVOLUTION
from ._io import LBM3D_INPUT,LBM3D_OUTPUT
from ._initialize import LBM3D_INITIALIZATION
from util.flag import *
@ti.data_oriented
class LBM3D_BASE(LBM3D_BASE,LBM3D_EVOLUTION,LBM3D_INPUT,LBM3D_OUTPUT,LBM3D_INITIALIZATION):
    """
    LBM3DSolver: Combined class for 3D LBM solver with evolution, I/O, and initialization functionalities
    """
    def __init__(self, X, Y, Z,dx = 0.001,dt = 0.001,name="default_LBM",isThermal = False,isChemical = False,isPoro = False,isRadiation = False):
        super().__init__(X, Y, Z,dx,dt,name,isThermal,isChemical,isPoro,isRadiation)
    def __str__(self):
        return f"LBM3DSolver: {self.name}, Size: {self.nx} x {self.ny} x {self.nz}, dx: {self.dx}, dt: {self.dt}, Thermal: {self.TEMPERATURE}, Chemical: {self.CHEMISTRY}, Poro: {self.PORO}, Radiation: {self.RADIATION}"
    def __repr__(self):
        return self.__str__()