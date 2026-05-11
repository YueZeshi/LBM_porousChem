# LBM —— 基于 Taichi 的多物理场格子玻尔兹曼仿真框架

LBM 是一个基于 [Taichi](https://www.taichi-lang.org/) 的格子玻尔兹曼方法（Lattice Boltzmann Method）多物理场仿真框架，专注于**生物质热解模拟**，支持流体流动、热传导、多组分化学反应、多孔介质和辐射传热的耦合求解。

---

## 核心功能

| 功能模块 | 描述 |
|----------|------|
| **流体求解** | D2Q9（2D）/ D3Q19（3D），支持 BGK（单松弛）/ **MRT（多松弛）** 碰撞模型，Sutherland 变粘度、理想气体状态方程 |
| **热输运** | D2Q5/D3Q7 标量输运模型，流体/固体双温度场，支持常值/多项式/NASA7/混合物热物性 |
| **化学反应** | 多步有限速率 Arrhenius 动力学，支持质量基/摩尔基反应，可加载 Cantera 机理文件 |
| **多孔介质** | 球堆积(Spherical/Ergun)、Darcy、Darcy-Forchheimer 阻力模型 |
| **辐射传热** | 表面均匀温度辐射模型 |
| **边界条件** | NEE（非平衡外推）、NEBB（非平衡反弹）、周期/入口/出口/壁面/对称 |
| **可视化** | 内置 VTK/PVD 输出 + ParaView 启动器 + 内建 PyVista 浏览器 |
| **AI 加速** | SRLBM 模块（基于 PyTorch 的深度学习代理模型，实验性） |

---

## 技术栈

- **Python 3.12**
- **Taichi** —— GPU 加速并行计算后端（支持 CUDA/Vulkan/CPU）
- **Cantera** —— 化学热力学数据库（NASA7 多项式 + 机理文件）
- **PyTorch** —— SRLBM 深度学习代理模型
- **PyVista / VTK** —— 可视化与数据导出
- **cadquery / meshio / rtree** —— 几何建模与 STL 处理
- **click** —— CLI 命令行入口
- **ruamel.yaml** —— YAML 配置解析

---

## 安装

### 环境要求

- Python 3.12（必须）
- 推荐使用 `uv` 管理虚拟环境

### 从源码安装

```bash
cd /mnt/d/yzs/lbm
pip install -e .
```

这将注册以下 CLI 命令：

| 命令 | 作用 |
|------|------|
| `lbm` | 运行仿真 |
| `paraLbm` | 使用系统 ParaView 打开结果 |
| `pvLbm` | 使用内置 PyVista 浏览器查看结果 |
| `ck2yaml` | Cantera CK 格式转 YAML |
| `cti2yaml` | Cantera CTI 格式转 YAML |
| `ctml2yaml` | Cantera CTML 格式转 YAML |

### 依赖列表

```
click, numpy, taichi, matplotlib, pyevtk, pytest, ruamel-yaml,
cantera, torch, torchvision, rtree, pyvista, scipy, cadquery,
vtk, meshio, scipy, ipykernel, typing-extensions
```

---

## 快速开始

### 1. 编写配置文件 `config.yaml`

最小二维修正示例（lid-driven cavity）：

```yaml
basic:
  name: lid_driven
  dimension: 2
  arch: gpu
  collisionModel: BGK

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
  DX: 0.001

timeControl:
  startTime: 0
  endTime: 10
  DT: 0.001

boundaryCondition:
  left:
    flow: { type: wall }
  right:
    flow: { type: wall }
  down:
    flow: { type: wall }
  up:
    flow: { type: wall, velocity: [0.01, 0] }

initialCondition:
  flow:
    rho: 1.0
    velocity: [0, 0]

outputControl:
  vtk: { interval: 1, path: vtk, clear: true }
  log: { interval: 1 }
  snapshot: { interval: 10, path: snapshot }
```

### 2. 运行仿真

```bash
lbm -c config.yaml -v 1
```

### 3. 查看结果

```bash
# 使用内置 PyVista 浏览器
pvLbm -d vtk

# 或使用系统 ParaView
paraLbm -d vtk
```

---

## CLI 命令说明

### `lbm` —— 运行仿真

```
lbm -c config.yaml [-v 0|1|2] [--clear]
```

| 选项 | 说明 |
|------|------|
| `-c, --config` | YAML 配置文件路径（默认 `config.yaml`） |
| `-v, --verbose` | 日志级别：0=warning, 1=info, 2=debug |
| `--clear` | 清除历史仿真结果文件夹 |

### `paraLbm` —— ParaView 可视化

```
paraLbm -d <数据文件夹>
```

自动查找文件夹中的 `.pvd` 文件，使用系统安装的 ParaView 打开。

### `pvLbm` —— 内置可视化

```
pvLbm -d <数据文件夹>
```

使用内建的 PyVista 浏览器直接查看结果，无需安装 ParaView。

---

## 目录结构

```
lbm/
├── src/
│   ├── cli.py                    # CLI 命令行入口
│   ├── LBM/
│   │   ├── app.py                # 应用编排（配置文件加载 → 求解器运行）
│   │   ├── config.yaml           # ParaView 路径配置
│   │   ├── LBM2D/                # 2D LBM 求解器
│   │   │   ├── _core.py          # 核心基类（D2Q9 格子）
│   │   │   ├── _evolution.py     # 碰撞/流式传输
│   │   │   ├── _boundary.py      # 边界条件
│   │   │   ├── _thermal.py       # 热输运
│   │   │   ├── _chemical.py      # 化学反应
│   │   │   ├── _scalarField.py   # 标量场基类
│   │   │   ├── _io.py            # 输入/输出
│   │   │   ├── _info.py          # 信息/日志
│   │   │   ├── _initialize.py    # 初始化
│   │   │   └── __init__.py       # LBM2DSolver 组合类
│   │   ├── LBM3D/                # 3D LBM 求解器（结构同 LBM2D）
│   │   ├── util/
│   │   │   ├── flag.py           # 枚举/标志定义
│   │   │   ├── constant.py       # 物理常数 + 元素摩尔质量
│   │   │   └── math.py           # 数学工具
│   │   └── GEO/
│   │       └── STL.py            # STL 几何生成器
│   ├── SRLBM/                    # 深度学习代理模型（实验性）
│   ├── cantera_tool/             # Cantera 格式转换工具
│   └── visualization_tool/
│       ├── PVD.py                # PVD/VTK 导出器
│       └── PvViewer.py           # PyVista 内建浏览器
├── examples/
│   ├── 2D/                       # 2D 仿真示例
│   └── 3D/                       # 3D 仿真示例（含完整热解案例）
├── tests/                        # 单元测试
├── docs/                         # 文档
│   ├── API.md                    # API 参考
│   └── CONFIG.md                 # 配置文件详解
├── data/                         # Cantera 机理文件
├── pyproject.toml                # 项目配置
└── README_zh.md                  # 本文件
```

---

## 引用

如果您在研究中使用了本项目，请引用相关论文：

> 待补充 —— 请联系项目作者获取引用信息。

**作者**：Sylvain (ZeshiYUE)  
**邮箱**：609190542@qq.com  
**许可证**：MIT
