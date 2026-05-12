# Benchmark 配置参考

> **创建日期**: 2026-05-11
> **用途**: 记录所有 benchmark 算例的正确配置规范，降低后续配置错误概率

---

## 目录

1. [算例概览](#1-算例概览)
2. [配置文件规范](#2-配置文件规范)
3. [各算例详情](#3-各算例详情)
   - [01 — Poiseuille 流动](#01--poiseuille-流动)
   - [02 — Lid-Driven Cavity](#02--lid-driven-cavity)
   - [03 — Thermal Channel](#03--thermal-channel)
   - [04 — Porous Flow](#04--porous-flow)
   - [05 — Pyrolysis](#05--pyrolysis)
4. [已知陷阱](#4-已知陷阱)
5. [历史错误记录](#5-历史错误记录)

---

## 1. 算例概览

| # | 名称 | 模块 | 碰撞模型 | 状态 |
|---|------|------|----------|------|
| 01 | Poiseuille 流动 | flow | BGK / MRT | ✅ 已验证 |
| 02 | Lid-Driven Cavity | flow | BGK / MRT | ✅ 已验证 |
| 03 | Thermal Channel | flow + temperature | BGK | ✅ 已验证 |
| 04 | Porous Flow | flow + porous_media | BGK | ✅ 已验证 |
| 05 | Pyrolysis | flow + temperature + chemistry + porous_media | BGK | ✅ 已验证 |

所有算例均位于 `examples/benchmark/` 目录下。

---

## 2. 配置文件规范

### 2.1 通用结构

```yaml
basic:              # 基本信息（必需）
  name: <算例名>
  dimension: <2|3>
  arch: <cpu|gpu>
  collisionModel: <BGK|MRT>

module:             # 启用模块（必需）
  flow: <true|false>
  chemistry: ...
  temperature: ...
  porous_media: ...
  radiation: ...

flowProperties:     # 流体属性（flow=true 时必需）
  viscosity: { type: constant, value: <数值> }

spaceControl:       # 空间离散（必需）
  geometry: [<长>, <宽>]        # 2D: [Lx, Ly]
  DX: <网格尺寸>

timeControl:        # 时间离散（必需）
  startTime: 0
  endTime: <结束时间>
  DT: <时间步长>

boundaryCondition:  # 边界条件（必需）
  left/right/down/up:           # 2D 四边
    flow: { type: <类型>, ... }

initialCondition:   # 初始条件（必需）
  flow: { rho: <密度>, velocity: [<vx>, <vy>, <vz>] }

outputControl:      # 输出控制（可选）
  vtk: { interval: <间隔>, path: vtk, clear: true }
  log: { interval: <间隔> }
  snapshot: { interval: <步数>, path: snapshot }
```

### 2.2 支持边界条件类型

| `type` 值 | 描述 | 必需参数 | 支持场景 |
|-----------|------|---------|---------|
| `inlet` | 入口（速度边界） | `velocity: [vx, vy, vz]` | flow, thermalFluid, chemical |
| `outlet` | 出口（压力边界） | `rho: <密度>` | flow |
| `inlet_flowrate` | 流量入口 | `flowrate`, `area`, `rho` | flow |
| `wall` | 壁面（无滑移） | (可选 `velocity`) | flow |

> ⚠️ **不支持的 BC 类型**: `movingWall`、`NEE`、`NEBB` 等——BC 类型名必须在上述4种之内。
> ⚠️ **BC 模型**: 代码默认使用 NEE（非平衡外推），无需在 config 中指定。`NEBB` 仅枚举存在但未启用。

### 2.3 其他规范

| 字段 | 正确格式 | 错误示例 |
|------|---------|---------|
| 速度值 | 3 个浮点的列表 `[vx, vy, vz]` | `[vx, vy]` 或 `vx` |
| 化学反应方程 | `=` 连接 | `=>`、`→`、`<=>` |
| 化学类型 | `type: input` | `type: homogeneous` |
| 边界命名 | `left/right/down/up` | `top/bottom`、`north/south` |

---

## 3. 各算例详情

### 01 — Poiseuille 流动

**物理**: 二维平面通道内压力驱动流动，验证抛物线速度分布。

- `config_bgk.yaml` — BGK 碰撞（已验证 ✅）
- `config_mrt.yaml` — MRT 碰撞（已验证 ✅）

**关键参数**:
| 参数 | BGK | MRT |
|------|-----|-----|
| 粘度 `ν` | 0.002 | 0.002 |
| 几何 | [0.1, 0.02] | [0.1, 0.02] |
| DX | 1e-3 | 1e-3 |
| DT | 1e-3 | 1e-3 |
| 入口速度 | [0.01, 0, 0] | [0.01, 0, 0] |

**边界**: 左→inlet, 右→outlet(rho=1), 上下→wall

### 02 — Lid-Driven Cavity

**物理**: 顶盖驱动方腔流动，验证涡旋结构。

- `config_bgk.yaml` — BGK 碰撞（已验证 ✅）
- `config_mrt.yaml` — MRT 碰撞（已验证 ✅）

**关键参数**:
| 参数 | BGK | MRT |
|------|-----|-----|
| 几何 | [0.1, 0.1] | [0.1, 0.1] |
| 粘度 `ν` | 0.002 | 0.002 |

**边界**:
| 边 | 类型 | 参数 |
|----|------|------|
| 上(up) | `inlet` | velocity: [0.01, 0, 0] |
| 下/左/右 | `wall` | — |

### 03 — Thermal Channel

**物理**: 入口流体加热，下壁高温，上壁低温，验证热输运。

- `config.yaml` — BGK 碰撞，含温度耦合（已验证 ✅）

**关键参数**: 几何 [0.1, 0.05], ν=0.002, Thermal conductivity=0.05, 热膨胀系数=0.1

**边界**:
| 边 | flow | thermalFluid |
|----|------|-------------|
| 左 | inlet [0.01,0,0] | fixedValue 0.2 |
| 右 | outlet rho=1 | zeroGradient |
| 下 | wall | fixedValue 0.1 |
| 上 | wall | fixedValue 0.0 |

### 04 — Porous Flow

**物理**: 多孔介质达西流动，验证压力梯度下的渗流。

- `config.yaml` — BGK 碰撞，含 porous_media（已验证 ✅）

**关键参数**: 几何 [0.2, 0.05], ν=0.002, 孔隙率=0.5, 渗透率=1e-6

**边界**: 左→inlet [0.005,0,0], 右→outlet rho=1, 上下→wall

### 05 — Pyrolysis

**物理**: 木柴热解，固体木材→焦油+焦炭+气体，验证耦合化学反应。

- `config.yaml` — BGK 碰撞，flow+porous+temperature+chemistry（已验证 ✅）

**关键参数**: 几何 [0.05, 0.05], ν=0.002, DT=5e-5(需更小时间步), 孔隙率=0.4

**化学反应**: `Wood = 0.5 Tar + 0.35 Char + 0.15 Gas`, A=1e6, Ea=10000

**化学属性**:
```yaml
chemicalProperties:
  type: input          # 必须为 input（不可用 homogeneous）
  species: ...
  reactions:
    - equation: "Wood = 0.5 Tar + 0.35 Char + 0.15 Gas"  # 用 = 号
```

**边界**:
| 边 | flow | thermalFluid | chemical |
|----|------|-------------|----------|
| 左 | inlet [0.005,0,0] | fixedValue 0.0 | fixedValue, N2:1.0 |
| 右 | outlet rho=1 | zeroGradient | zeroGradient |
| 下 | wall | fixedValue 0.1 | — |
| 上 | wall | fixedValue 0.0 | — |

---

## 4. 已知陷阱

### ❌ 陷阱1: 不支持的 BC 类型名

`app.py` 的 `load_boundary_conditions()` (L425-443) 仅识别 `inlet` / `outlet` / `inlet_flowrate` / `wall` 四种。
使用 `movingWall`、`NEE`、`NEBB` 等名称时不会被任何 elif 捕获，边界**静默退化为 periodic**。

**解决**: 顶盖驱动腔用 `inlet` 代替 `movingWall`。

### ❌ 陷阱2: 化学反应方程分隔符

`_chemical.py` 的 `Reaction.parse()` 使用 `=` 作为分隔符解析反应物/生成物。
使用 `=>`、`→` 等符号会**解析失败**。

**解决**: 使用 `A = B + C` 格式。

### ❌ 陷阱3: `chemicalProperties.type` 错误

原 benchmark `05_pyrolysis/config.yaml` 写成了 `type: homogeneous`，
但 `app.py` 的 `load_chemical_properties()` 仅处理 `type: input` (对应手动定义物种和反应)。
`homogeneous` 分支未实现解析用户配置的逻辑。

**解决**: 使用 `type: input`。

### ❌ 陷阱4: 速度字段需3分量

即使 2D 算例，速度字段也必须提供 3 个浮点数 `[vx, vy, vz]`。
代码内部 `_core.py` 的 `vel` 是 `ti.Vector.field(3, ...)`。

**解决**: 写 `[0.01, 0, 0]` 而非 `[0.01, 0]`。

### ⚠️ 注意事项: MRT 模式

MRT 碰撞模型需要可变松弛率支持（s7/s8 由 tau 实时计算），
已通过 `_core.py` 和 `_evolution.py` 的源码修改解决。
MRT config 和 BGK config 在 BC 配置上无差别。

---

## 5. 历史错误记录

| 日期 | 错误描述 | 影响文件 | 修复方式 |
|------|---------|---------|---------|
| 2026-05-11 | config_mrt.yaml 使用 `movingWall` 而非 `inlet` | `config_mrt.yaml` | 改为 `type: inlet` |
| 2026-05-11 | 试图 patch `_initialize.py` 添加速度BC初始化——根本原因是 config 写错 | `_initialize.py`(回退) | 改为修 config |
| 2026-05-11 | pyrolysis config 反应方程用 `=>` | `config.yaml` | 改为 `=` |
| 2026-05-11 | pyrolysis config 化学类型用 `homogeneous` | `config.yaml` | 改为 `type: input` |
| 2026-05-11 | 考虑引入 NEBB 配置——代码默认NEE无需干预 | (无影响) | 确认默认NEE后放弃 |