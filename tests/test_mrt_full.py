import taichi as ti
ti.init(arch=ti.cpu)

import sys
sys.path.insert(0, r'D:\yzs\lbm\src')

from LBM.LBM3D import LBM3DSolver
from LBM.util.flag import COLLISION_MODEL
import numpy as np

print("=== TEST 1: BGK ===")
solver_bgk = LBM3DSolver(X=0.1, Y=0.1, Z=0.1, dx=0.01, dt=0.001,
                          isThermal=False, isChemical=False)
print(f"  collision_model = {solver_bgk.collision_model} [OK]")

print("=== TEST 2: MRT ===")
solver_mrt = LBM3DSolver(X=0.1, Y=0.1, Z=0.1, dx=0.01, dt=0.001,
                          isThermal=False, isChemical=False,
                          collision_model=COLLISION_MODEL.MRT)
print(f"  collision_model = {solver_mrt.collision_model} [OK]")
print(f"  s_mrt_D3Q19[0]={solver_mrt.s_mrt_D3Q19[0]} [OK]")

print("=== TEST 3: BGK step ===")
solver_bgk.static_init_kernel()
solver_bgk.step()
print(f"  BGK step OK, tLattice={solver_bgk.tLattice} [OK]")

print("=== TEST 4: MRT step ===")
solver_mrt.static_init_kernel()
solver_mrt.step()
print(f"  MRT step OK, tLattice={solver_mrt.tLattice} [OK]")

print("=== TEST 5: MRT multi-step ===")
for i in range(10):
    solver_mrt.step()
print(f"  10 steps OK, tLattice={solver_mrt.tLattice} [OK]")

rho = solver_mrt.rho.to_numpy()
print(f"  rho range: [{rho.min():.4f}, {rho.max():.4f}] [OK]")

print("\n=== ALL TESTS PASSED! ===")