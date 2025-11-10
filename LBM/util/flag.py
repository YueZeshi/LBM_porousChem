from enum import Enum
class SPECIE_UNIT(Enum):
    MASS = 1
    MOLE = 2
class REACTION_TYPE(Enum):
    ARREHNIUS = 0
    LMH = 1
class RADIATION_MODEL(Enum):
    NONE = 0
    SURFACE_UNIFORM = 1
    REAL_RADIATION = 2
class PORO_MODEL(Enum):
    SPHERICAL = 0
    DARCY = 1
    DARCYFORCHHEIMER=2
class SOURCE_TERM(Enum):
    NONE = 0
    MICRO = 1 # 修改分布函数
    MACRO = 2 # 修改宏观量
class FORCE_TERM(Enum):
    NONE = 0
    GUO = 1 # 外力项
    SHAN = 2 # 改变平衡速度
    MACRO = 3 # 处理外力项太大的刚性问题
class BC(Enum):
    periodic = 0
    zeroGadient = 1
    fixedValue = 2
class BC_FLOW(Enum):
    periodic = 0
    inlet = 1
    outlet = 2
    wall = 3
    symmetric = 4
class BC_MODEL(Enum):
    NONE = 0
    NEE = 1
    NEBB = 2
    EQUILIBRIUM = 3