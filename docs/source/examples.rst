Tutorial examples
=================

Algorithm
---------
Structure
---------
lbm 2D
++++++
lbm 3D
++++++
cantera tool
+++++++++++++
In order to support chemkin file input, thanks to the open-sourced kinetic software Cantera, we could convert the chemkin file into yaml which is more readable. 
This lbm software could read this yaml to set the thermodynamics data, transport data and reactions.

The cantera could calculate the thermodynamical and transport properties of the mixtures in an implicit way. The coefficients described in the yaml file is not visible in ct.Solution class.
We should analyse the yaml file by ourselves. So there is no need to include cantera package in the project which is mainly used for zero D and one D simulaition of chemical reactions.

**yaml strcture**

refer to https://cantera.org/stable/userguide/input-tutorial.html

1. description

2. generator

3. input-files

4. cantera-version

5. date

6. units

 - length
 - time
 - quantity
 - activation-energy

7.phases : list of phases 

 - name 
 - themo : 

  + for gas : ideal-gas/Redlich-Kwong/Peng-Robinson/plasma
  + for surface and interface : ideal-surface/coverage-dependent-surface/edge
  + for single-species phase : fixed-stoichiometry/electron-cloud/pure-fluid/liquid-water-IAPWS95
  + for ideal solid and liquid : ideal-molar-solution/ideal-condensed/ideal-solution-VPSS/lattice/compound-lattice
  + for non-ideal solid and liquid solution : binary-solution-tabulated/Debye-Huckel
 - elements 
 - species 
 - kinetics : gas
 - transport : mixture-averaged
 - state

8.species : list of species

 - name 
 - composition 
 - equation-of-state

  + model: constant-volume/density-temperature-polynomial/molar-volume-temperature-polynomial/Peng-Robinson/Redlich-Kwong/HKFT/liquid-water-IAPWS95

 - thermo

  + model : NASA7/NASA9/Shomate/constant-cp/piecewise-Gibbs
  + temperature-ranges : two ranges
  + data : depend on the model
  + note 注释

 - transport

  + model 
  + geometry 
  + well-depth 
  + diameter 
  Examples
  ========

  本节系统介绍本项目自带的典型算例，涵盖二维/三维流体动力学、热/质量传递、多物理场耦合、复杂几何、化学反应等多种场景。

  2D 典型算例
  -----------

  - **泊肃叶流（Poiseuille Flow）**：验证层流管道流动的速度分布。
  - **顶盖驱动腔流（Lid-driven Cavity）**：经典方腔内驱动流动，考察涡结构。
  - **圆柱绕流（Cylinder Flow）**：二维圆柱后涡街、雷诺数效应。
  - **热扩散（Thermal Diffusion）**：温度场与流场耦合，热对流扩散。
  - **催化反应（Catalysis）**：壁面反应、物质传递与流体动力学耦合。
  - **热解反应（Pyrolysis）**：多组分反应动力学与流动、传热耦合。

  3D 典型算例
  -----------

  - **3D 方腔驱动流（3D Lid-driven Cavity）**：三维腔体内驱动流动。
  - **3D 圆柱绕流**：空间涡结构与雷诺数效应。
  - **3D 热扩散/热对流**：三维温度场与流场耦合。
  - **3D 热解/多物理场反应**：复杂反应体系三维模拟。

  GEO 几何算例
  -------------

  - **GEO 2D/3D**：支持复杂几何导入（如 STL），可用于多孔介质、催化剂载体等结构。

  Cantera 工具与化学反应
  ----------------------

  本项目集成 Cantera 工具，支持 Chemkin 文件到 YAML 的转换，便于读取热力学、输运与反应动力学参数。
  算例涵盖：

  - **气体/多相反应**：通过 YAML 配置多组分、复杂反应体系。
  - **热解/催化反应**：与 LBM 流场/温度场耦合。

  详细 YAML 结构与用法请参考 Cantera 官方文档：
  https://cantera.org/stable/userguide/input-tutorial.html

  验证与扩展
  ----------

  所有算例均可用于算法验证、性能测试与物理建模扩展。具体输入文件、参数与后处理脚本请见 examples 目录及各子文件夹。

  如需自定义新算例，可参考上述模板进行扩展。
