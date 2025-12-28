from ast import Constant
from enum import Enum
class VISCOSITY_MODEL(Enum):
    NONE = 0
    CONSTANT = 1
    SUTHERLAND = 2
    MIXTURE = 3

class PORO_MODEL(Enum):
    # the porosity model
    SPHERICAL = 0 # ergun
    DARCY = 1 # linear
    DARCYFORCHHEIMER=2 # non-linear
    DARCY_HIGH = 3 # correction (not correct)
    ERGUN = 4

class CONDUCTIVITY_MODEL(Enum):
    NONE = 0
    CONSTANT = 1
    POLYNOMIAL = 2
    MIXTURE = 3

class CAPACITY_MODEL(Enum):
    NONE = 0
    CONSTANT = 1
    POLYNOMIAL = 2
    NASA7 = 3
    MIXTURE = 4

class THERMAL_DIFF_MODEL(Enum):
    NONE = 0
    CONSTANT = 1
    DERIVED = 2
    PRANDTL = 3


class SPECIE_UNIT(Enum):
    MASS = 1
    MOLE = 2

class REACTION_TYPE(Enum):
    ARREHNIUS = 0 # power law
    LMH = 1 # langemuir--

class FLUID_STATE_EQUATION(Enum):
    IDEAL_GAS = 0 # compressible model
    INCOMPRESSIBLE = 1

class RADIATION_MODEL(Enum):
    # radiation model
    NONE = 0 # no radiation
    SURFACE_UNIFORM = 1 # uniform environmental temperature
    REAL_RADIATION = 2 # baked radiation
    P1_APPROACH = 3 # not implemented

class SOURCE_TERM(Enum):
    # the source term schema
    NONE = 0
    MICRO = 1 # 修改分布函数
    MACRO = 2 # 修改宏观量
class FORCE_TERM(Enum):
    NONE = 0
    GUO = 1 # 外力项
    SHAN = 2 # 改变平衡速度
    MACRO = 3 # 处理外力项太大的刚性问题
class BC(Enum):
    # the boundary condition for general field
    periodic = 0
    zeroGradient = 1
    fixedValue = 2
class BC_FLOW(Enum):
    # the combined boundary condition for flow field => one condition will set two field : velocity and density
    periodic = 0
    inlet = 1
    outlet = 2
    wall = 3
    symmetric = 4
    inlet_flow = 5
class BC_MODEL(Enum):
    # the boundary condition schema
    NONE = 0
    NEE = 1
    NEBB = 2
    ES = 3