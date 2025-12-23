import taichi as ti
from ._core import STLBM2D_BASE
from ._evolution import STLBM2D_EVOLUTION
from ._boundary import STLBM2D_BOUNDARY
from ._io import STLBM2D_INPUT,STLBM2D_OUTPUT
from ._initialize import STLBM2D_INITIALIZATION
from ..util.flag import *


@ti.data_oriented
class STLBM2DSolver(STLBM2D_EVOLUTION,STLBM2D_BOUNDARY,STLBM2D_INPUT,STLBM2D_OUTPUT,STLBM2D_INITIALIZATION,STLBM2D_BASE):
    def __init__(self, X, Y,dx = 1,dt =1,name="default_STLBM",isThermal = False,isChemical = False,isPoro = False,isRadiation = False):
        super().__init__(X, Y,dx,dt,name,isThermal,isChemical,isPoro,isRadiation)
    def __str__(self):
        return f"STLBM2DSolver: {self.name}, Size: {self.nx} x {self.ny} x {self.nz}, dx: {self.dx}, dt: {self.dt}, Thermal: {self.TEMPERATURE}, Chemical: {self.CHEMISTRY}, Poro: {self.PORO}, Radiation: {self.RADIATION}"
    def __repr__(self):
        return self.__str__()