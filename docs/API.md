# API 参考

LBM 项目的 Python API 文档。本文档涵盖核心类、枚举和配置结构。

---

## LBM3D_BASE（核心基类）

`LBM3D_BASE` 是 3D LBM 求解器的核心基类，管理物理场、格子参数和模块开关。

**文件**：`src/LBM/LBM3D/_core.py`

### 构造函数

```python
LBM3D_BASE(X, Y, Z, dx=0.001, dt=0.001, name="LBM",
           isThermal=False, isChemical=False, isPoro=False, isRadiation=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `X` | float | — | 物理域 X 方向长度 (m) |
| `Y` | float | — | 物理域 Y 方向长度 (m) |
| `Z` | float | — | 物理域 Z 方向长度 (m) |
| `dx` | float | 0.001 | 空间步长 (m) |
| `dt` | float | 0.001 | 时间步长 (s) |
| `name` | str | "LBM" | 求解器名称，用于输出文件 |
| `isThermal` | bool | False | 启用温度/热传导模块 |
| `isChemical` | bool | False | 启用化学反应/物种输运模块 |
| `isPoro` | bool | False | 启用多孔介质模型 |
| `isRadiation` | bool | False | 启用辐射模型（需同时启用 isThermal） |

### 主要属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `nx, ny, nz` | int | 各方向格子数 |
| `dx` | float | 空间步长 |
| `dt` | float | 时间步长 |
| `rho` | ti.field(float, (nx,ny,nz)) | 流体密度场 |
| `v` | ti.Vector.field(3,float, (nx,ny,nz)) | 流体速度场 |
| `solid` | ti.field(float, (nx,ny,nz)) | 固体区域标识（0=纯流体, 1=纯固体） |
| `rhos` | ti.field(float, (nx,ny,nz)) | 固体密度场 |
| `f` | ti.Vector.field(19,float, (nx,ny,nz)) | D3Q19 分布函数 |
| `F` | ti.Vector.field(19,float, (nx,ny,nz)) | 后碰撞分布函数 |
| `viscosity_model` | VISCOSITY_MODEL | 粘度模型（CONSTANT / SUTHERLAND / MIXTURE） |
| `boundary_condition_model` | BC_MODEL | 边界条件方案（NEE / NEBB / ES） |
| `EOS` | FLUID_STATE_EQUATION | 状态方程（IDEAL_GAS / INCOMPRESSIBLE） |
| `poro_model` | PORO_MODEL | 多孔介质模型 |
| `TF` | TemperatureFluid | 流体温度场（需 isThermal） |
| `TS` | TemperatureSolid | 固体温度场（需 isThermal） |
| `species` | list[Specie] | 化学物种列表（需 isChemical） |
| `reactions` | Reactions | 化学反应管理（需 isChemical） |
| `tLattice` | int | 当前时间步数 |
| `bc` | list[BC_FLOW] | 6 个面的流场边界条件类型 |
| `even_step` | ti.i32 | ET 单数组奇偶步标记（0=偶, 1=奇） |

### 关键字段说明

| 字段 | 维度 | 说明 |
|------|------|------|
| `e19` | 19×3 | D3Q19 离散速度方向向量 |
| `w19` | 19 | D3Q19 权重系数 |
| `LR` | 19 | 对称方向索引映射 |
| `e7` | 7×3 | D3Q7 离散速度方向向量（标量输运） |
| `w7` | 7 | D3Q7 权重系数 |
| `ext_f` | 3 | 外部体积力 |
| `bc` | 6 | 6 个面的复合边界条件 (BC_FLOW) |
| `bc_v` | 6 | 6 个面的速度边界类型 (BC) |
| `bc_rho` | 6 | 6 个面的密度边界类型 (BC) |
| `v_BC` | 6×3 | 6 个面的指定速度值 |
| `rho_BC` | 6 | 6 个面的指定密度值 |
| `flow_BC` | 6 | 6 个面的指定质量流量值 |

---

## LBM2D_BASE（2D核心基类）

`LBM2D_BASE` 是 2D LBM 求解器的核心基类，管理 D2Q9 格子、物理场和碰撞模型。

**文件**：`src/LBM/LBM2D/_core.py`

### 构造函数

```python
LBM2D_BASE(X, Y, Z, dx=0.001, dt=0.001, name="LBM",
           isThermal=False, isChemical=False, isPoro=False,
           isRadiation=False,
           collision_model=COLLISION_MODEL.BGK)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `X` | float | — | 物理域 X 方向长度 (m) |
| `Y` | float | — | 物理域 Y 方向长度 (m) |
| `Z` | float | — | 物理域 Z 方向长度 (m) — 二维中代表"厚度" |
| `dx` | float | 0.001 | 空间步长 (m) |
| `dt` | float | 0.001 | 时间步长 (s) |
| `name` | str | "LBM" | 求解器名称 |
| `isThermal` | bool | False | 启用温度/热传导模块 |
| `isChemical` | bool | False | 启用化学反应/物种输运模块 |
| `isPoro` | bool | False | 启用多孔介质模型 |
| `isRadiation` | bool | False | 启用辐射模型 |
| `collision_model` | COLLISION_MODEL | BGK | 碰撞模型（BGK/MRT） |

### 关键字段

| 字段 | 维度 | 说明 |
|------|------|------|
| `f` | ti.Vector.field(9) | D2Q9 分布函数 |
| `F` | ti.Vector.field(9) | 后碰撞分布函数 |
| `rho` | ti.field(float) | 宏观密度场 |
| `v` | ti.Vector.field(3) | 宏观速度场 |
| `solid` | ti.field(float) | 固体标志场 |
| `e9` | 9×2 | D2Q9 离散速度方向向量 |
| `w9` | 9 | D2Q9 权重系数 |
| `LR` | 9 | 对称方向索引映射 |
| `e5` | 5×2 | D2Q5 离散速度方向向量（标量输运） |
| `w5` | 5 | D2Q5 权重系数 |
| `collision_model` | COLLISION_MODEL | 当前碰撞模型 |
| `s_mrt` | 9 | MRT 松弛率（仅 MRT 模式） |
| `tau_mrt` | 9 | MRT 松弛时间（仅 MRT 模式） |
| `s_mrt_q5` | 5 | D2Q5 MRT 松弛率（仅 MRT 模式） |
| `tau_mrt_q5` | 5 | D2Q5 MRT 松弛时间（仅 MRT 模式） |

### 相关方法

| 方法 | 说明 |
|------|------|
| `setup_mrt_rates(tau, tau_bulk, s_magic)` | 设置 MRT 松弛率（仅 MRT 模式可用） |

---

## LBM3DSolver（组合求解器类）

`LBM3DSolver` 通过多继承组合了所有功能模块，是用户使用 3D 求解器的统一入口。

**文件**：`src/LBM/LBM3D/__init__.py`

```python
class LBM3DSolver(LBM3D_EVOLUTION, LBM3D_BOUNDARY,
                  LBM3D_INPUT, LBM3D_OUTPUT,
                  LBM3D_INITIALIZATION, LBM3D_INFO,
                  LBM3D_BASE):
```

### 主要方法

| 方法 | 说明 |
|------|------|
| `step()` | 执行一个时间步（边界→演化→同步） |
| `init_simulation()` | 初始化仿真（根据宏观量初始化分布函数） |
| `export_VTK_pyvista()` | 导出当前时刻 VTK 数据 |
| `description()` | 返回求解器多行描述字符串 |
| `log_info()` | 返回当前时刻日志信息 |
| `check_valid()` | 检查求解器配置有效性 |

---

## Specie（化学物种）

**文件**：`src/LBM/LBM3D/_chemical.py`

```python
Specie(name, lb3d, FIX=False, Mmass=1.0, unit=UNIT.MOLE)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | — | 物种名称 |
| `lb3d` | LBM3D_BASE | — | 所属 LBM 实例 |
| `FIX` | bool | False | True=固相物种，False=流相物种 |
| `Mmass` | float | 1.0 | 摩尔质量 (g/mol) |
| `unit` | UNIT | UNIT.MOLE | 热物性单位 (MOLE / MASS) |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `molemass` | float | 摩尔质量 (kg/mol) |
| `viscosity_type` | VISCOSITY_MODEL | 粘度模型（仅流相） |
| `visco` | float | 常值粘度 |
| `coefSutherland` | list[2] | Sutherland 系数 [As, Ts] |
| `diff_model` | DIFF_MODEL | 扩散模型（CONSTANT / SCHMIDT） |
| `diff` | float | 常值扩散系数 |
| `isInert` | bool | 是否为惰性组分（不参与扩散） |
| `thermo_model` | THERMO_MODEL | 热力学模型 |
| `enthalpy` | float | 常值焓 |
| `capa` | float | 常值热容 |
| `NASAcoef` | list[2,7] | NASA7 多项式系数 |
| `capa_poly` | list[5] | 多项式热容系数 |
| `cond_model` | CONDUCTIVITY_MODEL | 热导率模型 |
| `cond` | float | 常值热导率 |

---

## Reaction（化学反应）

**文件**：`src/LBM/LBM3D/_chemical.py`

单个化学反应由反应方程式和 Arrhenius 速率常数定义。

```python
Reaction(equation, A, Ea, b, name="", fixDH=False, deltaH=0.0, unit=UNIT.NONE)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `equation` | str | 反应方程式，如 `"wood <=> tar + syngas"` |
| `A` | float | Arrhenius 指前因子 |
| `Ea` | float | 活化能 (J/mol) |
| `b` | float | 温度指数 |
| `name` | str | 反应名称 |
| `fixDH` | bool | 是否使用固定反应热 |
| `deltaH` | float | 固定反应热 (J/kg 或 J/mol) |
| `unit` | UNIT | 反应速率单位 (MASS / MOLE / NONE) |

### 速率公式

\[
k = A \cdot T^b \cdot \exp\left(-\frac{E_a}{RT}\right)
\]

产物分布根据反应方程式中各组分的化学计量系数自动计算。

---

## Reactions（反应集合）

**文件**：`src/LBM/LBM3D/_chemical.py`

管理所有化学反应的集合类，负责计算各反应对物种质量分数的贡献。

```python
Reactions(lb3d)  # 自动读取 lb3d.species 和 lb3d.reactions 列表
```

---

## TemperatureFluid（流体温度场）

**文件**：`src/LBM/LBM3D/_thermal.py`

```python
TemperatureFluid(name, lb3d, FIX=False)
```

继承自 `ScalarField`，使用 D3Q7 格子求解流体温度输运方程。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `thermal_diff_model` | THERMAL_DIFF_MODEL | 热扩散模型（CONSTANT / PRANDTL / DERIVED） |
| `Pr` | float | 普朗特数（默认 0.71） |
| `thermal_diff` | float | 常值热扩散系数 |
| `conductivity_model` | CONDUCTIVITY_MODEL | 热导率模型 |
| `cond` | float | 常值热导率 |
| `cond_poly` | list[5] | 多项式热导率系数 |
| `capacity_model` | THERMO_MODEL | 热容模型 |
| `cm` | float | 常值质量热容 |
| `cm_poly` | list[5] | 多项式热容系数 |
| `Trange` | list[3] | NASA7 温度范围 [Tmin, Tmid, Tmax] |
| `NASA_coef` | (2,7) | NASA7 多项式系数 |

### 热扩散模型说明

| 模型 | 说明 |
|------|------|
| CONSTANT | 使用固定热扩散系数 |
| PRANDTL | 通过运动粘度 / Pr 计算：`α = ν / Pr` |
| DERIVED | 通过热导率、密度、热容计算：`α = k / (ρ·cp)` |

---

## TemperatureSolid（固体温度场）

**文件**：`src/LBM/LBM3D/_thermal.py`

```python
TemperatureSolid(name, lb3d, FIX=False, isRadiation=False)
```

继承自 `ScalarField`，使用 D3Q7 格子求解固体温度输运方程。支持辐射表面属性。

### 辐射相关属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `radiation_surface` | ti.field | 辐射表面标记（固体边界格点） |
| `emissivity` | ti.field | 表面发射率 |
| `radiation_model` | RADIATION_MODEL | 辐射模型类型 |
| `Tambient` | float | 环境温度 (K) |

---

## ScalarField（标量场基类）

**文件**：`src/LBM/LBM3D/_scalarField.py`

所有标量场（温度、物种）的抽象基类，提供 D3Q7 格子玻尔兹曼标量输运的通用接口。

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | str | 标量场名称 |
| `LBM` | LBM3D_BASE | 所属求解器引用 |
| `FIX` | bool | 是否为固定相 |
| `S` | ti.field(float, (nx,ny,nz)) | 标量值场 |
| `g` | ti.Vector.field(7,float, (nx,ny,nz)) | D3Q7 分布函数 |
| `nx, ny, nz` | int | 网格维度 |

---

## 枚举类型

**文件**：`src/LBM/util/flag.py`

### BC_FLOW —— 流场复合边界条件

| 值 | 说明 |
|----|------|
| `periodic` (0) | 周期边界 |
| `inlet` (1) | 速度入口（指定速度 + 零梯度密度） |
| `outlet` (2) | 压力出口（指定密度 + 零梯度速度） |
| `wall` (3) | 无滑移壁面（反弹边界） |
| `symmetric` (4) | 对称边界 |
| `inlet_flow` (5) | 质量流量入口（指定质量流量） |

### BC —— 通用标量场边界条件

| 值 | 说明 |
|----|------|
| `periodic` (0) | 周期边界 |
| `zeroGradient` (1) | 零梯度（Neumann） |
| `fixedValue` (2) | 固定值（Dirichlet） |

### BC_MODEL —— 边界条件算法方案

| 值 | 说明 |
|----|------|
| `NONE` (0) | 无 |
| `NEE` (1) | 非平衡外推（Non-Equilibrium Extrapolation） |
| `NEBB` (2) | 非平衡反弹（Non-Equilibrium Bounce-Back） |
| `ES` (3) | 平衡态方案（Equilibrium Scheme） |

### PORO_MODEL —— 多孔介质模型

| 值 | 说明 |
|----|------|
| `SPHERICAL` (0) | 球堆积模型（基于渗透率 Carman-Kozeny） |
| `DARCY` (1) | 线性 Darcy 阻力 |
| `DARCYFORCHHEIMER` (2) | Darcy-Forchheimer 非线性阻力 |
| `DARCY_HIGH` (3) | Darcy 修正（不推荐） |
| `ERGUN` (4) | Ergun 方程 |

### VISCOSITY_MODEL —— 粘度模型

| 值 | 说明 |
|----|------|
| `CONSTANT` (1) | 常值粘度 |
| `SUTHERLAND` (2) | Sutherland 公式：μ = As·T¹·⁵/(T+Ts) |
| `MIXTURE` (3) | 混合物粘度（按质量分数加权） |

### THERMO_MODEL —— 热物性模型

| 值 | 说明 |
|----|------|
| `CONSTANT` (1) | 常值 |
| `POLYNOMIAL` (2) | 多项式（最多 5 阶） |
| `NASA7` (3) | NASA7 多项式（双温度区间） |
| `MIXTURE` (4) | 混合物（按组分质量分数加权） |

### THERMAL_DIFF_MODEL —— 热扩散模型

| 值 | 说明 |
|----|------|
| `CONSTANT` (1) | 常值热扩散系数 |
| `DERIVED` (2) | 由热导率+热容推导 |
| `PRANDTL` (3) | 通过 Prandtl 数推导 |

### CONDUCTIVITY_MODEL —— 热导率模型

| 值 | 说明 |
|----|------|
| `CONSTANT` (1) | 常值 |
| `POLYNOMIAL` (2) | 多项式（最多 5 阶） |
| `MIXTURE` (3) | 混合物（按组分质量分数加权） |

### DIFF_MODEL —— 物种扩散模型

| 值 | 说明 |
|----|------|
| `CONSTANT` (1) | 常值扩散系数 |
| `SCHMIDT` (2) | 通过 Schmidt 数推导 |

### FLUID_STATE_EQUATION —— 流体状态方程

| 值 | 说明 |
|----|------|
| `IDEAL_GAS` (0) | 理想气体（可压缩）：ρ = p/RT |
| `INCOMPRESSIBLE` (1) | 不可压缩 |

### RADIATION_MODEL —— 辐射模型

| 值 | 说明 |
|----|------|
| `NONE` (0) | 无辐射 |
| `SURFACE_UNIFORM` (1) | 表面均匀环境温度辐射 |
| `REAL_RADIATION` (2) | 烘焙辐射 |
| `P1_APPROACH` (3) | P1 近似（未实现） |

### UNIT —— 物理量单位

| 值 | 说明 |
|----|------|
| `MASS` (1) | 质量基（per kg） |
| `MOLE` (2) | 摩尔基（per mol） |
| `NONE` (0) | 无单位 |

### COLLISION_MODEL —— 碰撞模型

| 值 | 说明 |
|----|------|
| `BGK` (0) | 单松弛时间 Bhatnagar-Gross-Krook 模型 |
| `MRT` (1) | 多松弛时间模型（需调用 setup_mrt_rates 设置松弛率） |

---

## 配置文件 YAML 结构

配置文件是 YAML 格式，驱动整个仿真流程。顶层区块：

| 区块 | 必需 | 说明 |
|------|------|------|
| `basic` | 是 | 基本信息（名称、维度、计算架构） |
| `module` | 是 | 物理模块开关 |
| `flowProperties` | 是 | 流体属性（粘度模型） |
| `thermalPropertiesFluid` | 否 | 流体热物性（需 temperature=true） |
| `thermalPropertiesSolid` | 否 | 固体热物性（需 temperature=true） |
| `radiationProperties` | 否 | 辐射参数（需 radiation=true） |
| `chemicalProperties` | 否 | 化学机理（需 chemistry=true） |
| `geometry` | 否 | 几何区域定义 |
| `boundaryCondition` | 是 | 边界条件（6 面 + 各物理场） |
| `spaceControl` | 是 | 空间离散（几何尺寸 + DX） |
| `timeControl` | 是 | 时间控制（起止时间 + DT） |
| `initialCondition` | 是 | 初始条件 |
| `outputControl` | 是 | 输出控制（VTK/日志/快照） |

详细配置说明请参阅 [CONFIG.md](CONFIG.md)。

---

## 应用入口

**文件**：`src/LBM/app.py`

```python
application(config: ruamel.yaml.comments.CommentedMap, logger: logging.Logger)
```

按顺序执行以下步骤：

1. `load_basic_config` — 加载维度/架构/名称
2. `load_space_config` — 加载几何尺寸与 DX
3. `load_time_config` — 加载 DT
4. `load_module_config` — 加载物理模块开关
5. `initialize_lbm_solver` — 构造 LBM 求解器实例
6. `load_flow_properties` — 加载粘度属性
7. `load_thermal_properties` — 加载热物性
8. `load_radiation_properties` — 加载辐射参数
9. `load_chemical_properties` — 加载化学机理
10. `load_boundary_conditions` — 加载边界条件
11. `load_geometry` — 加载几何区域
12. `load_initial_conditions` — 加载初始条件
13. `load_time_and_output_control` — 加载输出控制
14. 运行主仿真循环
