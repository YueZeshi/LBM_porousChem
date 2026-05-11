"""
test_mrt.py — LBM 2D MRT 碰撞模型测试
验证: (1) MRT初始化 (2) setup_mrt_rates设置 (3) 与BGK比较流场结果
采用 lid-driven cavity 算例
"""
import time, sys, os
import numpy as np
import taichi as ti
sys.path.insert(0, r'D:\yzs\lbm\src')
from LBM.LBM2D import LBM2DSolver as LB2D, COLLISION_MODEL
from LBM.util.flag import *

ARCH = os.getenv("ARCH","cpu").lower()
if ARCH=="gpu":
    ti.init(arch=ti.gpu, kernel_profiler=True, print_ir=False)
else:
    ti.init(arch=ti.cpu, kernel_profiler=False, print_ir=False)

def run_test(collision, name_suffix):
    """运行lid-driven cavity测试"""
    X, Y = 200, 200
    DX = 1.0; DT = 1.0

    lb2d = LB2D(X, Y, 1, dx=DX, dt=DT,
                isPoro=False, isChemical=False,
                isThermal=False, isRadiation=False,
                collision_model=collision)

    lb2d.boundary_condition_model = BC_MODEL.NEE
    lb2d.EOS = FLUID_STATE_EQUATION.IDEAL_GAS
    lb2d.set_viscosity(0.05)

    # lid-driven cavity: top wall moving, others stationary
    lb2d.set_BCs([BC_FLOW.wall, BC_FLOW.wall, BC_FLOW.wall, BC_FLOW.wall])
    lb2d.set_v_BCs_value([[0,0,0], [0,0,0], [0.1,0,0], [0,0,0]])
    lb2d.set_rho_BCs_value([1.0, 1.0, 1.0, 1.0])
    lb2d.init_field(lb2d.rho, 1.0)

    # 如果是MRT，设置松弛率（通过tau）
    if collision == COLLISION_MODEL.MRT:
        lb2d.setup_mrt_rates(tau=3*0.05+0.5)

    lb2d.init_simulation()
    print(f"\n{lb2d.description()}")

    total_iter = 1000
    measures = []

    for it in range(total_iter):
        if it % 100 == 0:
            max_v = lb2d.get_max_v()
            measures.append(max_v)
            print(f"  iter={it}, max_v={max_v:.6f}")
        lb2d.step()

    print(f"  Final max_v={measures[-1]:.6f}")
    return measures, lb2d

def main():
    print("=" * 60)
    print("LBM 2D MRT 碰撞模型测试")
    print("=" * 60)

    # 测试1: BGK (基准)
    print("\n--- 测试1: BGK 碰撞 ---")
    m_bgk, lb_bgk = run_test(COLLISION_MODEL.BGK, "BGK")

    # 测试2: MRT (使用setup_mrt_rates)
    print("\n--- 测试2: MRT 碰撞 ---")
    m_mrt, lb_mrt = run_test(COLLISION_MODEL.MRT, "MRT")

    # 汇总
    print("\n" + "=" * 60)
    print("结果汇总 (max_v 演化):")
    print(f"  BGK:         {m_bgk[-1]:.6f}")
    print(f"  MRT:         {m_mrt[-1]:.6f}")
    print("=" * 60)

    # 保存数据
    out_dir = r'D:\yzs\lbm\examples\2D\test_mrt_results'
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, 'BGK.txt'), m_bgk)
    np.savetxt(os.path.join(out_dir, 'MRT.txt'), m_mrt)
    print(f"\n数据已保存至: {out_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())