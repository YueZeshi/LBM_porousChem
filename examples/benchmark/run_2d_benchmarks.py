#!/usr/bin/env python3
"""Run 2D LBM benchmarks from WSL environment (CPU mode)."""
import sys, os, time, shutil
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / 'src'
BENCH_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

CASES = [
    ('Poiseuille BGK',     '01_poiseuille/config_bgk.yaml',     'poiseuille_bgk'),
    ('Poiseuille MRT',     '01_poiseuille/config_mrt.yaml',     'poiseuille_mrt'),
    ('Lid-driven BGK',     '02_lid_driven_cavity/config_bgk.yaml','cavity_bgk'),
    ('Lid-driven MRT',     '02_lid_driven_cavity/config_mrt.yaml','cavity_mrt'),
    ('Thermal Channel',    '03_thermal_channel/config.yaml',     'thermal'),
    ('Porous Flow',        '04_porous_flow/config.yaml',         'porous'),
    ('Pyrolysis',          '05_pyrolysis/config.yaml',           'pyrolysis'),
]

def run_case(name, rel_path, short_name, timeout=600):
    from ruamel.yaml import YAML
    import logging

    cfg_path = BENCH_DIR / rel_path

    # Clean old vtk
    vtk_dir = cfg_path.parent / 'vtk'
    if vtk_dir.exists():
        shutil.rmtree(vtk_dir)

    # Setup logger
    logger = logging.getLogger(f"LBM_{short_name}")
    logger.setLevel(logging.WARNING)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)

    # Load config
    yaml = YAML()
    with open(cfg_path, 'r', encoding='utf8') as f:
        config = yaml.load(f)

    # Force CPU mode for WSL
    config['basic']['arch'] = 'cpu'

    from LBM.app import application
    print(f"  Running: {name} ...", end=' ', flush=True)
    t0 = time.time()
    try:
        application(config, logger)
        elapsed = time.time() - t0
        vtk_files = list((cfg_path.parent / 'vtk').glob('*.vt*')) if (cfg_path.parent / 'vtk').exists() else []
        print(f"OK [{elapsed:.1f}s] vtk:{len(vtk_files)}")
        return {'name': name, 'status': 'OK', 'elapsed': elapsed, 'vtk_count': len(vtk_files)}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAIL [{elapsed:.1f}s] {e}")
        return {'name': name, 'status': 'FAIL', 'elapsed': elapsed, 'error': str(e)}

def main():
    print("=" * 60)
    print("LBM 2D Benchmark Suite (WSL CPU)")
    print("=" * 60)

    results = []
    for name, rel_path, short_name in CASES:
        cfg_path = BENCH_DIR / rel_path
        if not cfg_path.exists():
            print(f"  SKIP: {name} (config not found)")
            continue
        results.append(run_case(name, rel_path, short_name))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("-" * 60)
    ok = sum(1 for r in results if r['status'] == 'OK')
    print(f"Passed: {ok}/{len(results)}")
    for r in results:
        marker = '[OK]' if r['status'] == 'OK' else '[FAIL]'
        print(f"  {marker} {r['name']:25s} [{r['elapsed']:.1f}s]", end='')
        if 'vtk_count' in r:
            print(f"  vtk:{r['vtk_count']}", end='')
        if 'error' in r:
            print(f"  {r['error']}", end='')
        print()

if __name__ == '__main__':
    main()
