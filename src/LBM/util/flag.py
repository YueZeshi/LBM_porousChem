"""模型开关与配置枚举集合。

统一管理各物理模型/数值模型的枚举类型，避免魔法数：
- 黏度/导热/热扩散/扩散系数/热容等本构模型
- 多孔介质、辐射、源项/外力项、流体状态方程
- 通用与流场边界条件
"""
from enum import Enum
from pickle import NONE
class UNIT(Enum):
    """物理量单位类型。"""
    MASS = 1
    MOLE = 2
    NONE = 0
class VISCOSITY_MODEL(Enum):
    """黏度模型选择。"""
    NONE = 0
    CONSTANT = 1
    SUTHERLAND = 2
    MIXTURE = 3

class PORO_MODEL(Enum):
    """多孔介质阻力模型。"""
    # the porosity model
    SPHERICAL = 0 # ergun
    DARCY = 1 # linear
    DARCYFORCHHEIMER=2 # non-linear
    DARCY_HIGH = 3 # correction (not correct)
    ERGUN = 4

class CONDUCTIVITY_MODEL(Enum):
    """导热模型。"""
    NONE = 0
    CONSTANT = 1
    POLYNOMIAL = 2
    MIXTURE = 3

class THERMO_MODEL(Enum):
    """热容/热力学性质模型。"""
    NONE = 0
    CONSTANT = 1
    POLYNOMIAL = 2
    NASA7 = 3
    MIXTURE = 4

class THERMAL_DIFF_MODEL(Enum):
    """热扩散模型。"""
    NONE = 0
    CONSTANT = 1
    DERIVED = 2
    PRANDTL = 3

class DIFF_MODEL(Enum):
    """物种扩散模型。"""
    NONE = 0
    CONSTANT = 1
    SCHMIDT = 2

class REACTION_TYPE(Enum):
    """反应速率类型（预留）。"""
    ARREHNIUS = 0 # power law
    LMH = 1 # langemuir--

class REACTION_ENERGY_TYPE(Enum):
    """反应能量类型（预留）。"""
    CONSTANT = 0
    DIFFERENCE = 1

class FLUID_STATE_EQUATION(Enum):
    """流体状态方程类型。"""
    IDEAL_GAS = 0 # compressible model
    INCOMPRESSIBLE = 1

class RADIATION_MODEL(Enum):
    """辐射模型。"""
    # radiation model
    NONE = 0 # no radiation
    SURFACE_UNIFORM = 1 # uniform environmental temperature
    REAL_RADIATION = 2 # baked radiation
    P1_APPROACH = 3 # not implemented

class BC(Enum):
    """通用标量场边界条件类型。"""
    # the boundary condition for general field
    periodic = 0
    zeroGradient = 1
    fixedValue = 2
class BC_FLOW(Enum):
    """流场复合边界（联合速度与密度的常见组合）。"""
    # the combined boundary condition for flow field => one condition will set two field : velocity and density
    periodic = 0
    inlet = 1
    outlet = 2
    wall = 3
    symmetric = 4
    inlet_flow = 5
class BC_MODEL(Enum):
    """边界条件实现方案（算法级）。"""
    # the boundary condition schema
    NONE = 0
    NEE = 1
    NEBB = 2
    ES = 3