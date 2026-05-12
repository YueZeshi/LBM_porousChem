#!/usr/bin/env python3
"""
LBM Benchmark Suite -- Multi-Physics Algorithm Validation

Usage:
    python run_benchmark.py [--cases 1,2,3] [--check-only]

Cases:
  1. Poiseuille flow (BGK vs MRT)       -- pure flow, analytical
  2. Lid-driven cavity (BGK vs MRT)     -- pure flow, recirculation
  3. Thermal channel flow               -- flow + thermal coupling
  4. Porous media flow                  -- flow + porous medium
  5. Pyrolysis (simplified)             -- flow + porous + thermal + chemistry
"""

import sys, os, time, subprocess, argparse, shutil
from pathlib import Path

SRC_DIR = Path(r'D:\yzs\lbm\src')
BENCH_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BENCH_DIR / 'results'

CASES = {
    1: {'name': 'Poiseuille (BGK)',       'cfg': '01_poiseuille/config_bgk.yaml'},
    2: {'name': 'Poiseuille (MRT)',        'cfg': '01_poiseuille/config_mrt.yaml'},
    3: {'name': 'Lid-driven cavity (BGK)', 'cfg': '02_lid_driven_cavity/config_bgk.yaml'},
    4: {'name': 'Lid-driven cavity (MRT)', 'cfg': '02_lid_driven_cavity/config_mrt.yaml'},
    5: {'name': 'Thermal channel',         'cfg': '03_thermal_channel/config.yaml'},
    6: {'name': 'Porous media flow',       'cfg': '04_porous_flow/config.yaml'},
    7: {'name': 'Pyrolysis',               'cfg': '05_pyrolysis/config.yaml'},
}

def run_case(cfg_rel_path, timeout=600):
    cfg_path = BENCH_DIR / cfg_rel_path
    case_name = cfg_path.parent.name
    log_dir = RESULTS_DIR / case_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'output.log'
    
    # Clean old vtk
    vtk_dir = cfg_path.parent / 'vtk'
    if vtk_dir.exists():
        shutil.rmtree(vtk_dir)
    
    cmd = [sys.executable, '-m', 'cli', 'run', '-c', str(cfg_path), '-v', '0']
    
    print(f"  Running: {case_name} ...", end=' ', flush=True)
    t0 = time.time()
    
    with open(log_file, 'w') as f:
        proc = subprocess.run(
            cmd, cwd=SRC_DIR,
            stdout=f, stderr=subprocess.STDOUT,
            timeout=timeout
        )
    
    elapsed = time.time() - t0
    status = 'OK' if proc.returncode == 0 else f'FAIL(rc={proc.returncode})'
    print(f"{status} [{elapsed:.1f}s]")
    
    vtk_files = list(cfg_path.parent.glob('vtk/*.vtk')) if vtk_dir.exists() else []
    return {
        'name': case_name,
        'rc': proc.returncode,
        'elapsed': elapsed,
        'vtk_count': len(vtk_files),
        'log': str(log_file)
    }

def main():
    parser = argparse.ArgumentParser(description='LBM Benchmark Suite')
    parser.add_argument('--cases', '-c', type=str, help='e.g. 1,2,3')
    parser.add_argument('--timeout', '-t', type=int, default=600)
    parser.add_argument('--check-only', action='store_true', help='Only verify config syntax')
    args = parser.parse_args()
    
    if args.cases:
        selected = [int(x) for x in args.cases.split(',')]
        to_run = {k: CASES[k] for k in selected if k in CASES}
    else:
        to_run = CASES
    
    print(f"LBM Benchmark Suite -- {len(to_run)} cases")
    print("=" * 60)
    
    if args.check_only:
        print("Config syntax check mode")
        for case_id, info in sorted(to_run.items()):
            cfg_path = BENCH_DIR / info['cfg']
            try:
                import yaml
                with open(cfg_path) as f:
                    yaml.safe_load(f)
                print(f"  [{case_id}] {info['name']:30s} OK YAML OK")
            except Exception as e:
                print(f"  [{case_id}] {info['name']:30s} FAIL {e}")
        return 0
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = []
    for case_id, info in sorted(to_run.items()):
        results.append(run_case(info['cfg'], args.timeout))
    
    print("
" + "=" * 60)
    print("SUMMARY")
    print("-" * 60)
    ok = sum(1 for r in results if r['rc'] == 0)
    print(f"Passed: {ok}/{len(results)}")
    for r in results:
        status = '[OK]' if r['rc'] == 0 else '[FAIL]'
        print(f"  {status} {r['name']:30s} [{r['elapsed']:.1f}s] vtk:{r['vtk_count']}")
    
    return 0 if ok == len(results) else 1

if __name__ == '__main__':
    sys.exit(main())
