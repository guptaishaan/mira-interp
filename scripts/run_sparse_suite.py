#!/usr/bin/env python3
"""Run three registered sparse seeds on the two explicitly authorized GPUs."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/data2/ishaangp/mira-interp/features/sparse_development_v1')
REPORT = ROOT / 'results/feature_development_v1/suite_status.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if REPORT.exists() or BASE.exists():
        raise ValueError('Preserve existing sparse-suite artifacts')
    protocol_path = ROOT / 'configs/feature_development_v1.json'
    protocol = json.loads(protocol_path.read_text()); protocol_hash = sha(protocol_path)
    geometry_path = ROOT / 'results/geometry_development_v1/geometry_audit.json'
    pilot_path = ROOT / 'results/feature_development_v1/synthetic_pilot.json'
    geometry = json.loads(geometry_path.read_text()); pilot = json.loads(pilot_path.read_text())
    if geometry['status'] != 'passed' or pilot['status'] != 'passed' or pilot['protocol_sha256'] != protocol_hash or pilot['geometry_audit_sha256'] != sha(geometry_path):
        raise ValueError('Geometry and full-width feasibility prerequisites must pass')
    # Authorization is restricted to6/7; an empty other GPU is never selected.
    uuids = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'], text=True)
    allowed = {uuid.strip() for line in uuids.splitlines() for index, uuid in [line.split(',')] if int(index) in (6,7)}
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
    if len(allowed) != 2 or any(line.split(',')[0].strip() in allowed for line in processes.splitlines() if ',' in line):
        raise ValueError('One of the two authorized GPUs is occupied')
    BASE.mkdir(parents=True)
    report = dict(status='running', protocol_sha256=protocol_hash, geometry_audit_sha256=sha(geometry_path),
                  synthetic_pilot_sha256=sha(pilot_path), assignments={'7':[0], '6':[1,2]}, reports=[],
                  expected_models=30, script_sha256=sha(Path(__file__)))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + '\n'); started = time.monotonic()

    def worker(gpu, seeds):
        records = []
        for seed in seeds:
            directory = BASE / f'seed{seed}'; log = REPORT.parent / f'seed{seed}.log'
            command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/analyze_feature_geometry.py'),
                       '--protocol', str(protocol_path), '--input', protocol['input_path'],
                       '--selected-probe', protocol['selected_probe_path'], '--gate', protocol['gate_path'],
                       '--output-dir', str(directory), '--stage', 'sparse', '--width', '3072',
                       '--active-budgets', '32', '64', '--steps', '1000', '--seed', str(seed),
                       '--device', 'cuda', '--temporal-weight', '.1', '--temporal-shuffle-control']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMPY_MADVISE_HUGEPAGE='0', PYTHONUNBUFFERED='1')
            with log.open('x') as stream:
                exit_code = subprocess.call(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            (REPORT.parent / f'seed{seed}.exit').write_text(str(exit_code)+'\n')
            result_path = directory / 'report.json'
            if exit_code != 0 or not result_path.exists():
                raise RuntimeError(f'Seed{seed} failed; inspect{log}')
            result = json.loads(result_path.read_text())
            if result['status'] != 'passed_development_feature_analysis' or len(result['sparse']) != 10:
                raise ValueError(f'Seed{seed} is incomplete')
            record = dict(seed=seed, gpu=gpu, path=str(result_path), sha256=sha(result_path), models=10)
            records.append(record); print(json.dumps(record), flush=True)
        return records

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, 7, [0]), pool.submit(worker, 6, [1,2])]
            for future in concurrent.futures.as_completed(futures):
                report['reports'].extend(future.result())
                REPORT.write_text(json.dumps(report, indent=2) + '\n')
        if sha(protocol_path) != protocol_hash:
            raise ValueError('Protocol changed during sparse training')
        report.update(status='passed', elapsed_seconds=time.monotonic()-started, completed_models=sum(r['models'] for r in report['reports']))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        REPORT.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
