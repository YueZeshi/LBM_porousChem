# 配置文件详解 (CONFIG.md)

基于 `examples/3D/pyrolysis_Sphere_Park/config.yaml` 的完整字段说明。

---

## 1. basic —— 基本信息

```yaml
basic:
  name: pyrolysis_test       # 仿真案例名称（用于输出文件命名）
  author: Sylvain            # 作者（可选）
  date: 2025-12-27           # 日期（可选）
  dimension: 3               # 维度：2 或 3
  arch: gpu                  # 计算后端：cpu / gpu / vulkan / cuda
  type: configuration        # 文件类型：configuration（配置文件）/ snapshot（快照恢复）
  collisionModel: BGK        # 碰撞模型：BGK（默认）/ MRT
```

| 字段 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `name` | str | 是 | — | 仿真案例名称 |
| `author` | str | 否 | — | 作者名 |
| `date` | str | 否 | — | 日期 |
| `dimension` | int | 是 | — | 2=二维, 3=三维 |
| `arch` | str | 是 | cpu | 计算架构：cpu/gpu/vulkan/cuda |
| `type` | str | 否 | configuration | 文件类型 |
| `collisionModel` | str | 否 | BGK | 碰撞模型：`BGK` 或 `MRT`。BGK 为单松弛时间模型，MRT 为多松弛时间模型（更高数值稳定性） |

---

## 2. module —— 物理模块开关

```yaml
module:
  flow: true                 # 流体求解（必须启用）
  porous_media: true         # 多孔介质模型
  temperature: true          # 温度/热传导
  chemistry: true            # 化学反应/物种输运
  radiation: true            # 辐射传热（需要 temperature=true）
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `flow` | bool | 是 | 流体求解模块（目前必须为 true） |
| `porous_media` | bool | 否 | 多孔介质阻力模型 |
| `temperature` | bool | 否 | 温度输运（流体+固体双温度场） |
| `chemistry` | bool | 否 | 化学反应 + 多组分输运 |
| `radiation` | bool | 否 | 辐射传热（需同时启用 temperature） |

---

## 3. flowProperties —— 流体属性

```yaml
flowProperties:
  viscosity:
    type: sutherland         # 粘度模型类型
    As: 1.6e-6               # Sutherland 系数 As
    Ts: 170                  # Sutherland 温度 Ts (K)
```

### viscosity

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | 粘度模型：`constant` / `sutherland` / `mixture` |

**当 type = constant 时：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `value` | float | 运动粘度 (m²/s) |

**当 type = sutherland 时：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `As` | float | Sutherland 系数 As (kg/(m·s·K^0.5)) |
| `Ts` | float | Sutherland 温度 Ts (K) |

公式：μ = As × T^1.5 / (T + Ts)

**当 type = mixture 时：**

无需额外参数，根据各物种粘度按质量分数混合。

---

## 4. thermalPropertiesFluid —— 流体热物性

```yaml
thermalPropertiesFluid:
  delay: 10                  # 温度场延迟启动步数
  normalize:
    Trange: [500, 1000]      # 温度归一化范围 [T_min, T_max]
  thermalDiff:
    type: derived            # 热扩散模型
  conductivity:
    type: mixture            # 热导率模型
  capacity:
    type: mixture            # 热容模型
```

### 4.1 delay

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `delay` | int | 0 | 温度场延迟启动步数。在流场建立稳定之前延迟温度场更新，避免数值不稳定 |

### 4.2 normalize

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `Trange` | list[2] | 否 | [T_min, T_max]，物理温度映射到 [0,1] 范围。用于无量纲化 |

### 4.3 thermalDiff

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | 热扩散模型：`constant` / `Prandtl` / `derived` |

**当 type = constant：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `value` | float | 热扩散系数 (m²/s) |

**当 type = Prandtl：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `Pr` | float | 普朗特数（默认 0.71）：α = ν / Pr |

**当 type = derived：**

由热导率和热容推导：α = k / (ρ·cp)。无需额外参数。

### 4.4 conductivity（热导率）

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | 热导率模型：`constant` / `polynomial` / `mixture` |

**当 type = constant：** `value` (float) — 热导率 (W/(m·K))

**当 type = polynomial：** `data` (list[5]) — 5 阶多项式系数 [c0,c1,c2,c3,c4]

**当 type = mixture：** 根据各物种热导率按质量分数混合。无需额外参数。

### 4.5 capacity（热容）

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | 热容模型：`constant` / `polynomial` / `NASA7` / `mixture` |

**当 type = constant：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `value` | float | 常值热容 |
| `unit` | str | `mole-based`（J/(mol·K)）或 `mass-based`（J/(kg·K)） |

**当 type = polynomial：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `data` | list[5] | 5 阶多项式系数 [c0,c1,c2,c3,c4]：cp = c0 + c1·T + c2·T² + c3·T³ + c4·T⁴ |

**当 type = NASA7：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `Trange` | list[3] | [Tmin, Tmid, Tmax] — 温度分段区间 |
| `data` | list[2,7] | NASA7 多项式系数（两个温度区间的 7 系数数组） |

**当 type = mixture：**

根据各物种热容按质量分数混合。无需额外参数。

---

## 5. thermalPropertiesSolid —— 固体热物性

结构与 `thermalPropertiesFluid` 相同：

```yaml
thermalPropertiesSolid:
  delay: 10
  normalize:
    Trange: [500, 1000]
  thermalDiff:
    type: derived            # 支持 constant / derived
  conductivity:
    type: mixture            # 支持 constant / polynomial / mixture
  capacity:
    type: mixture            # 支持 constant / polynomial / NASA7 / mixture
```

> 注意：固体热扩散模型仅支持 `constant` 和 `derived`（不支持 Prandtl）。

---

## 6. radiationProperties —— 辐射属性

```yaml
radiationProperties:
  type: mean_temp            # 辐射模型类型
  Tambient: 683              # 环境温度 (K)
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | 辐射模型：`mean_temp`（表面均匀环境温度）或 `none` |
| `Tambient` | float | 否 | 环境温度 (K)，默认 300 |

---

## 7. chemicalProperties —— 化学属性

```yaml
chemicalProperties:
  type: input                # 化学机理来源
  delay: 10                  # 化学反应延迟启动步数
  species:                   # 物种定义
    ...
  reactions:                 # 反应定义
    ...
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | 机理来源：`input`（YAML 内定义）或 `cantera`（加载 Cantera .yaml 文件） |
| `delay` | int | 否 | 化学反应延迟步数（默认 0） |
| `species` | dict | 仅 input | 物种定义（见 7.1） |
| `reactions` | list | 仅 input | 反应定义（见 7.2） |

**当 type = cantera 时：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `path` | str | Cantera YAML 机理文件路径 |

### 7.1 species —— 物种定义

每个物种是一个键值对，键名为物种名称。

```yaml
species:
  tar:                       # 物种名
    molemass: 28             # 摩尔质量 (g/mol)
    state: fluid             # 物态：fluid（气相）或 solid（固相）
    viscosity:               # 粘度（仅 fluid）
      type: constant
      value: 3e-5
    thermodynamic:           # 热力学属性
      type: polynomial
      data: [-110, 4.4, -1.57e-3, 0, 0]
      unit: mass-based
    diffusitivity:           # 扩散系数（仅 fluid）
      type: constant
      value: 1e-5
    conductivity:            # 热导率
      type: constant
      value: 0.0258
```

#### fluid 态物种字段

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `molemass` | float | 否 | 摩尔质量 (g/mol)。如果指定了 composition，则自动计算 |
| `composition` | dict | 否 | 元素组成，如 `{N: 2}` 表示 N₂。键名为元素符号（大写） |
| `state` | str | 是 | `fluid` |
| `viscosity` | dict | 是 | 粘度模型：`type` 支持 `constant` 或 `sutherland` |
| `thermodynamic` | dict | 是 | 热力学属性（见下方） |
| `diffusitivity` | dict | 是 | 扩散模型：`type` 支持 `constant` 或 `Schmidt` |
| `conductivity` | dict | 是 | 热导率模型：`type` 支持 `constant` 或 `polynomial` |

#### solid 态物种字段

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `molemass` | float | 否 | 摩尔质量 |
| `state` | str | 是 | `solid` |
| `thermodynamic` | dict | 是 | 热力学属性 |
| `conductivity` | dict | 是 | 热导率模型 |

#### 热力学属性 (thermodynamic)

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | 模型：`constant` / `polynomial` / `NASA7` |
| `unit` | str | 否 | 单位：`mole-based` 或 `mass-based`（仅 constant/polynomial） |

**constant 类型附加字段：**

| 字段 | 说明 |
|------|------|
| `enthalpy` | 生成焓 |
| `capacity` | 热容 |

**polynomial 类型附加字段：**

| 字段 | 说明 |
|------|------|
| `data` | 5 阶多项式系数 [c0,c1,c2,c3,c4] |

**NASA7 类型附加字段：**

| 字段 | 说明 |
|------|------|
| `Trange` | [Tmin, Tmid, Tmax] |
| `data` | 双区间 7 系数数组 |

#### 扩散属性 (diffusitivity)

仅 fluid 物种需要：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | str | `constant` 或 `Schmidt` |

**constant：** `value` (float) — 扩散系数

**Schmidt：** `Sc` (float) — Schmidt 数

---

**重要说明：**

- 如果未显式设置惰性组分（`isInert`），系统自动将最后一个 fluid 物种设为惰性组分。
- 惰性组分的质量分数由归一化约束自动计算，不参与扩散方程求解。
- 如果指定了 `composition` 字段（如 `{N: 2}`），则摩尔质量自动从元素摩尔质量表计算。

### 7.2 reactions —— 反应定义

```yaml
reactions:
  - name: tar formation               # 反应名称
    equation: wood <=> tar            # 反应方程式
    rate-constant: {A: 1.08e10, b: 0.0, Ea: 148000.0}  # Arrhenius 参数
    deltaH: 8000                      # 固定反应热
    unit: mass-based                  # 单位制
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | str | 是 | 反应名称（用于日志） |
| `equation` | str | 是 | 反应方程式。格式：`反应物 <=> 产物`。支持多物种：`A + B <=> C + D` |
| `rate-constant` | dict | 是 | Arrhenius 速率常数 |
| `rate-constant.A` | float | 是 | 指前因子 |
| `rate-constant.b` | float | 是 | 温度指数 |
| `rate-constant.Ea` | float | 是 | 活化能 (J/mol) |
| `deltaH` | float | 否 | 固定反应热。如果不指定，则根据生成焓自动计算 |
| `unit` | str | 否 | `mass-based`（质量基）或 `mole-based`（摩尔基） |

**反应速率公式：** k = A × T^b × exp(-Ea/RT)

> **注意：** 反应方程式中的物种名必须与 species 中定义的名称完全一致。`<=>` 符号表示不可逆反应（产物不参与逆反应）。

---

## 8. geometry —— 几何区域定义

```yaml
geometry:
  zone1:                     # 区域名称
    type: sphere             # 几何类型
    center: [0.05, 0.05, 0.05]  # 中心坐标 (m)
    radius: 0.02             # 半径 (m)
```

### 支持的几何类型

| 类型 | 必需字段 | 可选字段 | 说明 |
|------|----------|----------|------|
| `sphere` | `center`, `radius` | — | 球体 |
| `cylinder` | `center`, `radius`, `height` | `axis`（默认 [0,0,1]） | 圆柱体 |
| `box` | `center`, `size` | `rotate` | 长方体 |
| `cone` | `center`, `radius`, `height` | `axis` | 圆锥体 |
| `stl` | `path` | `scale`, `translate`, `rotate` | STL 文件导入 |

**坐标说明：**

- 所有坐标单位为米 (m)
- 2D 仿真中 center 为 [x, y]，会自动补 0
- zone 名称可自定义，用于后续初始条件和固体区域引用

**STL 导入说明：**

- `path`：STL 文件路径（绝对或相对）
- `scale`：缩放因子或 [sx,sy,sz]；如果是单个数字，三方向等比缩放
- `translate`：[dx,dy,dz] 平移向量 (m)
- `rotate`：[rx,ry,rz] 旋转角度 (度)

---

## 9. boundaryCondition —— 边界条件

```yaml
boundaryCondition:
  left:                      # 面名称
    flow:                    # 流场边界条件
      type: inlet
      velocity: [0.01, 0, 0]
    thermalFluid:            # 流体温度边界条件（需 temperature=true）
      type: fixedValue
      value: 683
    thermalSolid:            # 固体温度边界条件（需 temperature=true）
      type: none
    chemical:                # 化学物种边界条件（需 chemistry=true）
      type: fixedValue
      N2: 1.0                # 各物种的边界值
```

### 面名称

| 2D 名称 | 3D 名称 | 对应方向 |
|---------|---------|----------|
| `left` | `left` | X-（左） |
| `right` | `right` | X+（右） |
| `down` | `down` | Y-（下） |
| `up` | `up` | Y+（上） |
| — | `forward` | Z-（前） |
| — | `back` | Z+（后） |

### flow —— 流场边界条件

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | `inlet` / `outlet` / `wall` / `inlet_flowrate` |

**inlet（速度入口）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `velocity` | list[3] | 入口速度向量 [vx,vy,vz] (m/s) |
| `rho` | float | 入口密度（可选，用于压力计算） |

**outlet（压力出口）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `rho` | float | 出口密度 |

**wall（壁面）：**

无额外参数。

**inlet_flowrate（质量流量入口）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `flowrate` | float | 质量流量 (kg/s) |
| `rho` | float | 入口密度（可选） |

### thermalFluid / thermalSolid —— 温度边界条件

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | `fixedValue` / `zeroGradient` / `periodic` / `none` |
| `value` | float | 仅 fixedValue | 边界面温度 (K) |

- `none`：仅用于 thermalSolid，表示该面不与固体区域接触，跳过边界处理

### chemical —— 化学边界条件

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | str | 是 | `fixedValue` / `zeroGradient` / `periodic` |

**当 type = fixedValue 时：**

需要为每个参与输运的物种指定边界值（质量分数）：

```yaml
chemical:
  type: fixedValue
  tar: 0.0                  # 物种名: 边界质量分数
  syngas: 0.0
  N2: 1.0
```

---

## 10. spaceControl —— 空间控制

```yaml
spaceControl:
  geometry: [0.1, 0.1, 0.1]  # 物理域尺寸 [X, Y, Z] (m)
  DX: 0.001                   # 空间步长 (m)
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `geometry` | list | 是 | 物理域尺寸 (m)。2D 为 [X,Y]，3D 为 [X,Y,Z] |
| `DX` | float | 是 | 格子空间步长 (m)。格子数 = geometry / DX |

---

## 11. timeControl —— 时间控制

```yaml
timeControl:
  startTime: 0               # 起始时间 (s)
  endTime: 1000              # 结束时间 (s)
  DT: 0.001                  # 时间步长 (s)
```

| 字段 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `startTime` | float | 否 | 0 | 仿真起始物理时间 (s) |
| `endTime` | float | 是 | — | 仿真结束物理时间 (s) |
| `DT` | float | 是 | — | 时间步长 (s) |

---

## 12. initialCondition —— 初始条件

```yaml
initialCondition:
  solid:                     # 固体区域定义
    solid1:
      zone: zone1            # 引用的几何区域
      type: poro             # 固体类型
      rho: 400               # 固相密度 (kg/m³)
      porosity: 0.4          # 孔隙率（仅 poro）
      porousModel: darcy     # 多孔介质模型（仅 poro）
      darcy: 2.5e9           # Darcy 系数（仅 darcy）
      exchangeSurface: 1000  # 热交换比表面积 (m²/m³)
      exchangeCoef: 10       # 热交换系数 (W/(m²·K))
      emissivity: 0.6        # 表面发射率（仅 radiation=true）
  flow:                      # 流动初始条件
    rho: 1                   # 初始密度
    velocity: [0, 0, 0]      # 初始速度
  thermal:                   # 温度初始条件
    T: 200                   # 统一初始温度 (K)
    TS: 300                  # 固体初始温度 (K)（可选，覆盖 T）
    TF: 300                  # 流体初始温度 (K)（可选，覆盖 T）
  chemical:                  # 化学初始条件
    N2:
      zone: ALL              # 区域：ALL / zone 名称 / [zone1, zone2]
      value: 1.0             # 初始质量分数
    wood:
      zone: zone1
      value: 630             # 初始质量分数（可 > 1，为绝对质量）
```

### solid —— 固体区域

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `zone` | str/list | 是 | 引用的几何区域名、`ALL` 或名称列表 |
| `type` | str | 是 | `poro`（多孔）/ `concrete`（密实固体）/ `substract`（挖空） |
| `rho` | float | 否 | 固相表观密度 (kg/m³) |
| `porosity` | float | 仅 poro | 孔隙率 (0~1) |
| `porousModel` | str | 仅 poro | 多孔介质模型：`darcy` / `darcyforchheimer` / `ergun` |
| `darcy` | float | 仅 darcy | Darcy 阻力系数 |
| `forchheimer` | float | 仅 darcyforchheimer | Forchheimer 阻力系数 |
| `exchangeSurface` | float | 仅 poro+thermal | 热交换比表面积 (m²/m³) |
| `exchangeCoef` | float | 仅 poro+thermal | 热交换系数 (W/(m²·K)) |
| `emissivity` | float | 仅 poro+radiation | 表面发射率 (0~1) |

### flow —— 流动初始条件

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rho` | float | 1.0 | 初始密度 |
| `velocity` | list | [0,0,0] | 初始速度向量 |

### thermal —— 温度初始条件

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `T` | float | — | 统一初始温度 (K) |
| `TF` | float | T 的值 | 流体初始温度（覆盖 T） |
| `TS` | float | T 的值 | 固体初始温度（覆盖 T） |

### chemical —— 化学初始条件

每个条目以物种名为键：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `zone` | str/list | 是 | `ALL` / zone 名称 / 名称列表 |
| `value` | float | 是 | 该区域内的初始质量分数 |

> 注意：solid 物种可以设置 `value` > 1，表示绝对质量而非质量分数。

---

## 13. outputControl —— 输出控制

```yaml
outputControl:
  vtk:
    interval: 10              # VTK 输出间隔 (s)
    path: vtk                 # 输出文件夹相对路径
    LOD: 1                    # 细节层次（1=完整）
    clear: true               # 启动时清空历史输出
  log:
    interval: 1               # 日志输出间隔 (s)
    LOD: 1                    # 日志细节层次
  snapshot:
    interval: 100             # 快照保存间隔 (s)
    path: snapshot            # 快照文件夹路径
```

### vtk

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interval` | float | 100 | VTK 输出间隔 (s)。实际步数 = interval / DT |
| `path` | str | "output" | 输出文件夹 |
| `LOD` | int | 1 | 输出细节层次（目前仅支持 1） |
| `clear` | bool | false | 是否在启动时清空历史输出 |

输出文件格式：`{name}_{步数}.vts` + `{name}.pvd`（ParaView 集合文件）

### log

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interval` | float | 1 | 日志输出间隔 (s) |
| `LOD` | int | 1 | 日志细节层次 |

日志信息包括：执行时间、累计时间、MLUPS（每秒百万格子更新数）、各物理量最值。

### snapshot

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interval` | float | 100 | 快照保存间隔 (s) |
| `path` | str | "snapshot" | 快照文件夹 |

快照保存完整的物理场状态，可用于断点续算。

---

## 14. debugSetting —— 调试设置（可选）

```yaml
debugSetting:
  interval: 10               # 合法性检查间隔 (s)
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interval` | int | 10 | 合法性检查间隔 (s)。仅在 verbose=2 (debug) 时生效 |

---

## 最小配置模板

最简单的二维 lid-driven cavity 配置：

```yaml
basic:
  name: cavity
  dimension: 2
  arch: gpu

module:
  flow: true
  porous_media: false
  temperature: false
  chemistry: false
  radiation: false

flowProperties:
  viscosity:
    type: constant
    value: 1e-5

spaceControl:
  geometry: [0.1, 0.1]
  DX: 0.002

timeControl:
  startTime: 0
  endTime: 10
  DT: 0.001

boundaryCondition:
  left:   { flow: { type: wall } }
  right:  { flow: { type: wall } }
  down:   { flow: { type: wall } }
  up:     { flow: { type: wall, velocity: [0.1, 0] } }

initialCondition:
  flow:
    rho: 1.0
    velocity: [0, 0]

outputControl:
  vtk:      { interval: 0.5, path: vtk, clear: true }
  log:      { interval: 0.1 }
  snapshot: { interval: 5, path: snapshot }
```
