"""实用工具集合（util）。

包含：
- `constant`：常用物理常数与元素摩尔质量表。
- `flag`：枚举型配置（模型/边界/源项/外力等）。
- `math`：几何与数值小工具（如向量到欧拉角）。
- `path`：项目根路径定位工具。
"""

# 可选导出，便于快捷导入
from . import constant, flag, math

__all__ = [
	"constant",
	"flag",
	"math",
]