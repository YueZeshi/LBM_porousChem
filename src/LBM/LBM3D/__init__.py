import taichi as ti

from ._info import LBM3D_INFO
from ._core import LBM3D_BASE
from ._evolution import LBM3D_EVOLUTION
from ._io import LBM3D_INPUT,LBM3D_OUTPUT
from ._initialize import LBM3D_INITIALIZATION
from ._boundary import LBM3D_BOUNDARY
from ..util.flag import *
@ti.data_oriented
class LBM3DSolver(LBM3D_EVOLUTION,LBM3D_BOUNDARY,LBM3D_INPUT,LBM3D_OUTPUT,LBM3D_INITIALIZATION,LBM3D_INFO,LBM3D_BASE):
    """三维 LBM 求解器组合类。

    汇集演化模块、边界条件、输入/输出与初始化功能，提供一个
    统一的 3D LBM 使用入口。典型流程为：
    1) 构造求解器（配置网格/时间步长与功能开关）
    2) 初始化物理场与边界条件
    3) 时间步推进与周期性导出/记录

    Notes
    -----
    - 该类通过多继承组合不同模块，便于用户在同一对象上调用
      I/O、演化和边界相关方法。
    - 维度信息与坐标轴在 `LBM3D_BASE` 中统一管理。
    """
    def __init__(self, X, Y, Z,dx = 0.001,dt = 0.001,name="default_LBM",isThermal = False,isChemical = False,isPoro = False,isRadiation = False):
        """构造三维 LBM 求解器。

        Parameters
        ----------
        X, Y, Z : int | list | numpy.ndarray
            网格轴定义。可为轴长度（int）或一维坐标数组/列表。
            - 传入整数时，将根据长度生成等距网格。
            - 传入坐标数组时，按给定坐标构造结构化网格。
        dx : float, default 0.001
            空间步长（物理单位）。
        dt : float, default 0.001
            时间步长（物理单位）。
        name : str, default "default_LBM"
            求解器名称，用于输出文件命名。
        isThermal : bool, default False
            是否启用温度/热传导模块。
        isChemical : bool, default False
            是否启用化学反应/物种输运模块。
        isPoro : bool, default False
            是否启用多孔介质阻力模型（Darcy/Forchheimer）。
        isRadiation : bool, default False
            是否启用辐射相关模型。

        Notes
        -----
        - 若启用相应模块，请在输入侧接口中设置对应的物性、初值
          与边界条件；未启用的模块相关方法将不产生效果或返回占位值。
        """
        super().__init__(X, Y, Z,dx,dt,name,isThermal,isChemical,isPoro,isRadiation)
    def __str__(self):
        """返回求解器的简要描述字符串。"""
        return f"LBM3DSolver: {self.name}, Size: {self.nx} x {self.ny} x {self.nz}, dx: {self.dx}, dt: {self.dt}, Thermal: {self.TEMPERATURE}, Chemical: {self.CHEMISTRY}, Poro: {self.PORO}, Radiation: {self.RADIATION}"
    def __repr__(self):
        """同 `__str__`，用于交互式环境中的对象展示。"""
        return self.__str__()