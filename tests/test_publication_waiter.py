import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('publication_waiter', Path(__file__).resolve().parents[1] / 'scripts/watch_generated_publication.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_owner_completion_required(tmp_path):
    path = tmp_path / 'selection_audit.json'
    path.write_text(json.dumps({'status': 'passed_generated_measurement_audit', 'phase': 'selection',
                                'all_summary_metrics_independently_recomputed': True}))
    study = {'status': 'running', 'completed_stages': []}
    assert not module.ready(study, 'selection', path)
    study['completed_stages'] = [{'stage': 'selection_measurement_audit', 'path': str(path), 'sha256': module.sha(path)}]
    accepted_sha = module.ready(study, 'selection', path)
    assert accepted_sha == study['completed_stages'][0]['sha256'] == module.sha(path)
    # Failure in a later phase does not erase completed, unchanged selection evidence.
    study['status'] = 'failed'
    assert module.ready(study, 'selection', path)
    path.write_text(path.read_text() + '\n')
    assert accepted_sha != module.sha(path)
    with pytest.raises(RuntimeError, match='binding differs'):
        module.ready(study, 'selection', path)


def test_failed_owner_stops_wait(tmp_path):
    with pytest.raises(RuntimeError, match='Study failed'):
        module.ready({'status': 'failed', 'completed_stages': []}, 'selection', tmp_path / 'missing.json')
