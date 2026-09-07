#!/usr/bin/env python3
"""Wait for an owner-completed phase audit, then package its full outputs on CPU."""
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
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ready(study, phase, path):
    rows = [r for r in study['completed_stages'] if r['stage'] == phase + '_measurement_audit']
    if rows:
        require(len(rows) == 1 and Path(rows[0]['path']).resolve() == path.resolve()
                and rows[0]['sha256'] == sha(path), 'Owner-completed audit binding differs')
        audit = read(path)
        require(audit['status'] == 'passed_generated_measurement_audit'
                and audit['phase'] == phase and audit['all_summary_metrics_independently_recomputed'],
                'Successful complete measurement audit required')
        return rows[0]['sha256']
    require(study['status'] != 'failed', 'Study failed before requested phase audit completed')
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['selection', 'confirmation'], default='selection')
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--timeout-hours', type=float, default=6)
    p.add_argument('--state-dir', type=Path, default=ROOT / 'results/generated_publication_waiter_v2')
    args = p.parse_args()
    require(1 <= args.poll_seconds <= 60 and 0 < args.timeout_hours <= 24, 'Invalid wait budget')
    phase = args.phase
    review_path = ROOT / 'results/generated_publication_code_review.json'
    review = read(review_path)
    require(review['status'] == 'passed' and not review['remaining_material_findings'], 'Review must pass')
    package = ROOT / 'scripts/package_generated_outputs.py'
    helper = ROOT / 'src/mira_interp/array_storage.py'
    fixed = {path: review['code_sha256'][str(path.relative_to(ROOT))] for path in (package, helper)}
    fixed[review_path] = sha(review_path)
    fixed[Path(__file__).resolve()] = sha(__file__)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / (phase + '.json')
    report_path = ROOT / f'results/generated_publication_{phase}_complete.json'
    log_path = args.state_dir / (phase + '_package.log')
    require(not state_path.exists() and not report_path.exists(), 'Prior waiter/report must be preserved')
    command = [str(ROOT / '.venv/bin/python'), str(package), '--phase', phase]
    started = time.monotonic()
    state = dict(status='waiting', phase=phase, pid=os.getpid(), started_unix=time.time(),
                 fixed_bindings={str(k): v for k, v in fixed.items()}, command=command,
                 log_path=str(log_path), gpu_used=False, upload_performed=False)

    def save():
        state.update(elapsed_seconds=time.monotonic() - started, heartbeat_unix=time.time())
        tmp = state_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        tmp.replace(state_path)

    def unchanged():
        require(all(sha(k) == v for k, v in fixed.items()), 'Frozen publication code/review changed')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Publication watcher received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    process = None
    save()
    try:
        audit_path = ROOT / f'results/generated_evaluation_v2/{phase}_audit.json'
        while True:
            unchanged()
            require(time.monotonic() - started < args.timeout_hours * 3600, 'Wait budget expired')
            accepted_audit_sha = ready(read(ROOT / 'results/rollout_steering_v2/study_status.json'), phase, audit_path)
            if accepted_audit_sha:
                break
            save()
            time.sleep(args.poll_seconds)
        state.update(status='packaging', measurement_audit_sha256=accepted_audit_sha);save()
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'NUMPY_MADVISE_HUGEPAGE': '0',
                       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        with log_path.open('x') as log:
            unchanged()
            require(sha(audit_path) == accepted_audit_sha, 'Accepted measurement audit changed before packaging')
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state['package_pid'] = process.pid;save()
            code = process.wait(timeout=max(1, args.timeout_hours * 3600 - (time.monotonic() - started)))
        state['exit_code'] = code
        require(code == 0, 'Packager failed; inspect log')
        report = read(report_path)
        require(report['status'] == 'passed_generated_artifact_package' and report['phase'] == phase
                and report['expected_records'] == report['packaged_records'] == {'selection': 420, 'confirmation': 924}[phase]
                and report['measurement_audit_sha256'] == accepted_audit_sha == sha(audit_path)
                and report['all_members_readback_verified'] and report['all_exported_arrays_exact_source_slices']
                and report['all_registered_conditions_included'] and report['private_sources_unchanged']
                and not report['source_context_actions_simulator_labels_included'], 'Incomplete package report')
        unchanged()
        state.update(status='passed_generated_publication_waiter', report={'path': str(report_path), 'sha256': sha(report_path)})
        save()
    except BaseException as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL);process.wait(timeout=10)
            except ProcessLookupError:
                pass
        state.update(status='failed', error_type=type(error).__name__, error=str(error));save()
        raise


if __name__ == '__main__':
    main()
