边界条件（Boundary Conditions）
============================

本节汇总 LBM2D/3D 及相关模块支持的所有边界条件类型、物理含义与用法示例。


流体边界类型
------------

.. autoclass:: LBM.util.flag.BC
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: LBM.util.flag.BC_FLOW
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: LBM.util.flag.BC_MODEL
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


热/物质/多物理场边界
--------------------

.. autoclass:: LBM.util.flag.THERMAL_DIFF_MODEL
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: LBM.util.flag.DIFF_MODEL
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: LBM.util.flag.SPECIE_UNIT
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


边界设置方法
------------

2D:
.. automodule:: LBM.LBM2D._io
   :members: set_BC, set_BCs, set_BC_value, set_BC_flux
   :no-index:

3D:
.. automodule:: LBM.LBM3D._io
   :members: set_BC, set_BCs, set_BC_value, set_BC_flux
   :no-index:


示例
----

.. code-block:: python

   # 2D 设置入口速度边界
   input.set_BC('left', BC_FLOW.inlet, value=0.1)
   # 3D 设置温度固定边界
   input.set_BC('zmax', BC.fixedValue, value=300)

更多边界类型和参数说明请参考 API 文档和各模块 docstring。
