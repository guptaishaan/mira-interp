#!/usr/bin/env python3
"""Wait for the existing study's audits, then export each phase once on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def completed_audit(study, phase, audit_path):
    """A filename alone is insufficient: the owner must have finished the stage."""
    require(study['status'] != 'failed', 'Registered study failed before this export')
    stages = [row for row in study['completed_stages']
              if row['stage'] == phase + '_measurement_audit']
    if not stages:
        return False
    require(len(stages) == 1 and Path(stages[0]['path']).resolve() == audit_path.resolve()
            and stages[0]['sha256'] == sha(audit_path), 'Completed audit provenance differs')
    audit = read(audit_path)
    require(audit['status'] == 'passed_generated_measurement_audit'
            and audit['phase'] == phase and audit['all_summary_metrics_independently_recomputed'],
            'Full independent measurement audit is required')
    return True


def stop_owned_export(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def interrupted(signum, frame):
    raise KeyboardInterrupt('Export watcher received signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--timeout-hours', type=float, default=6)
    args = parser.parse_args()
    require(1 <= args.poll_seconds <= 60 and 0 < args.timeout_hours <= 24, 'Invalid wait budget')
    generation = ROOT / 'results/rollout_steering_v2'
    evaluation = ROOT / 'results/generated_evaluation_v2'
    review_path = ROOT / 'results/generated_exporter_review.json'
    review = read(review_path)
    require(review['status'] == 'passed_synthetic_generated_export_review', 'Exporter review must pass')
    fixed = {ROOT / name: digest for name, digest in review['code_sha256'].items()}
    fixed[review_path] = sha(review_path)
    require(Path(__file__).resolve() in fixed and ROOT / 'scripts/export_generated_measurements.py' in fixed,
            'Review must bind both export scripts')
    output = evaluation / 'export_watcher'
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / 'status.json'
    started = time.monotonic()
    state = {'status': 'waiting', 'pid': os.getpid(), 'started_unix': time.time(),
             'review_sha256': sha(review_path), 'completed_exports': [], 'phase': 'selection',
             'gpu_used': False, 'audit_or_model_execution_performed': False}

    def save():
        state['elapsed_seconds'] = round(time.monotonic() - started, 3)
        state['heartbeat_unix'] = time.time()
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temporary.replace(status_path)

    def unchanged():
        require(all(sha(path) == digest for path, digest in fixed.items()), 'Reviewed export code changed')
        for row in state['completed_exports']:
            require(sha(row['path']) == row['sha256'], 'Completed export report changed')

    save()
    signal.signal(signal.SIGTERM, interrupted)
    process = None
    try:
        for phase in ('selection', 'confirmation'):
            state.update(phase=phase, status='waiting');save()
            audit_path = evaluation / (phase + '_audit.json')
            while True:
                unchanged()
                require(time.monotonic() - started < args.timeout_hours * 3600, 'Export wait budget expired')
                study = read(generation / 'study_status.json')
                if completed_audit(study, phase, audit_path):
                    break
                save();time.sleep(args.poll_seconds)
            state['status'] = 'exporting';save()
            command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/export_generated_measurements.py'),
                       '--phase', phase, '--evaluation-dir', str(evaluation), '--generation-dir', str(generation)]
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'NUMPY_MADVISE_HUGEPAGE': '0',
                           'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OMP_NUM_THREADS': '8'}
            remaining = args.timeout_hours * 3600 - (time.monotonic() - started)
            require(remaining > 0, 'Export execution budget expired')
            with (output / (phase + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                state['export_pid'] = process.pid;save()
                require(process.wait(timeout=remaining) == 0, 'Exporter failed; inspect its phase log')
            path = evaluation / (phase + '_figures/export.json')
            report = read(path)
            require(report['status'] == 'passed_audited_generated_measurement_export'
                    and report['phase'] == phase and report['measurement_audit_sha256'] == sha(audit_path),
                    'Export did not finish with its matching gate')
            state['completed_exports'].append({'phase': phase, 'path': str(path), 'sha256': sha(path)})
            unchanged();save()
        state.update(status='passed_both_phase_exports_pending_visual_inspection', phase=None);save()
    except BaseException as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            stop_owned_export(process)
        except BaseException as cleanup_error:
            state['cleanup_error'] = str(cleanup_error)
        state.update(status='failed', error_type=type(error).__name__, error=str(error));save()
        raise


if __name__ == '__main__':
    main()
