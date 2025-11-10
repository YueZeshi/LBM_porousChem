import taichi as ti
from ._core import LBM2DSolver
from ._evolution import LBM2D_EVOLUTION
from ._boundary import LBM2D_BOUNDARY
from ._io import LBM2D_INPUT,LBM2D_OUTPUT
from ._init import LBM2D_INITIALIZATION
from ..util.flag import *
@ti.data_oriented
class LBM2DSolver(LBM2DSolver,LBM2D_EVOLUTION,LBM2D_BOUNDARY,LBM2D_INPUT,LBM2D_OUTPUT,LBM2D_INITIALIZATION):
    def __init__(self, X, Y,dx = 0.001,dt = 0.001,name="default_LBM",isThermal = False,isChemical = False,isPoro = False,isRadiation = False):
        super().__init__(X, Y,dx,dt,name,isThermal,isChemical,isPoro,isRadiation)
    def __str__(self):
        return f"LBM2DSolver: {self.name}, Size: {self.nx} x {self.ny} x {self.nz}, dx: {self.dx}, dt: {self.dt}, Thermal: {self.TEMPERATURE}, Chemical: {self.CHEMISTRY}, Poro: {self.PORO}, Radiation: {self.RADIATION}"
    def __repr__(self):
        return self.__str__()