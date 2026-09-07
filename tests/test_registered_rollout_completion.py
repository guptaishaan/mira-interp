"""Final completion must reject partial stages, stale metadata and remote substitutions."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('final_rollout_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_registered_rollout_completion.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def study_fixture(root):
    fixed = write(root / 'fixed.json', {'fixed': True})
    study = {'status': 'passed_generation_and_measurement_pending_publication', 'current_stage': None,
             'physical_control_established': False, 'fixed_bindings': {str(fixed): audit.sha(fixed)}, 'completed_stages': []}
    for phase in audit.PHASES:
        for name, status in audit.STAGES:
            folder = 'rollout_steering_v2' if name.startswith('generation') else 'generated_evaluation_v2'
            path = root / 'results' / folder / (phase + ('_audit' if name.endswith('audit') else '') + '.json')
            write(path, {'status': status})
            study['completed_stages'].append({'stage': phase + '_' + name, 'path': str(path), 'sha256': audit.sha(path)})
    return study


def test_complete_eight_stage_order_and_changed_prerequisite(tmp_path):
    study = study_fixture(tmp_path)
    assert len(audit.complete_study(study, tmp_path, audit.Bindings())) == 8
    write(tmp_path / 'fixed.json', {'fixed': False})
    with pytest.raises(ValueError, match='Hash differs'):
        audit.complete_study(study, tmp_path, audit.Bindings())


@pytest.mark.parametrize('mutation', ['running', 'missing', 'duplicate', 'reordered', 'bad_hash', 'bad_stage_status', 'physical_claim'])
def test_refuse_incomplete_or_inconsistent_study(tmp_path, mutation):
    study = study_fixture(tmp_path)
    if mutation == 'running': study['status'] = 'running'
    if mutation == 'missing': study['completed_stages'].pop()
    if mutation == 'duplicate': study['completed_stages'][-1] = study['completed_stages'][0]
    if mutation == 'reordered': study['completed_stages'].reverse()
    if mutation == 'bad_hash': study['completed_stages'][0]['sha256'] = '0' * 64
    if mutation == 'physical_claim': study['physical_control_established'] = True
    if mutation == 'bad_stage_status':
        row = study['completed_stages'][0]; write(Path(row['path']), {'status': 'running'}); row['sha256'] = audit.sha(row['path'])
    with pytest.raises(ValueError):
        audit.complete_study(study, tmp_path, audit.Bindings())


def test_running_study_never_calls_github_or_writes_report(tmp_path):
    write(tmp_path / 'results/rollout_steering_v2/study_status.json', {'status': 'running'})
    requests = []
    with pytest.raises(ValueError, match='incomplete'):
        audit.audit(tmp_path, {}, api=lambda path: requests.append(path))
    assert not requests
    assert not (tmp_path / 'completion.json').exists()


def remote_fixture():
    commit = 'a' * 40
    tag = 'generated-selection-v2-2026-09-07'
    url = 'https://github.com/' + audit.REPO + '/releases/tag/' + tag
    asset = {'id': 9, 'name': 'part.tar', 'size': 37, 'digest': 'sha256:' + 'b' * 64,
             'state': 'uploaded', 'browser_download_url': 'https://github.com/' + audit.REPO + '/releases/download/' + tag + '/part.tar'}
    release = {'id': 12, 'html_url': url, 'tag_name': tag, 'target_commitish': commit, 'draft': False, 'body': 'Notes\n'}
    responses = {'releases/tags/' + tag: release, 'git/ref/tags/' + tag: {'object': {'type': 'tag', 'sha': 'c' * 40}},
                 'git/tags/' + 'c' * 40: {'object': {'type': 'commit', 'sha': commit}},
                 'releases/12/assets?per_page=100&page=1': [asset]}
    pub = {'release_url': url, 'release_id': 12, 'target_commit': commit, 'assets': [copy.deepcopy(asset)]}
    specs = [{'name': 'part.tar', 'bytes': 37, 'sha256': 'b' * 64}]
    return pub, specs, responses


def test_live_remote_set_and_annotated_tag():
    pub, specs, responses = remote_fixture()
    result = audit.remote_publication(pub, specs, 'Notes', lambda p: responses[p])
    assert result['actual_tag_commit_checked'] and result['all_remote_sizes_and_digests_current']


@pytest.mark.parametrize('mutation', ['digest', 'size', 'missing', 'extra', 'duplicate', 'tag_commit', 'notes', 'draft', 'asset_identity'])
def test_remote_substitution_or_incomplete_upload_fails(mutation):
    pub, specs, responses = remote_fixture()
    rows = responses['releases/12/assets?per_page=100&page=1']
    release = responses['releases/tags/generated-selection-v2-2026-09-07']
    if mutation == 'digest': rows[0]['digest'] = None
    if mutation == 'size': rows[0]['size'] += 1
    if mutation == 'missing': rows.clear()
    if mutation == 'extra': rows.append({**rows[0], 'name': 'extra.tar'})
    if mutation == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    if mutation == 'tag_commit': responses['git/tags/' + 'c' * 40]['object']['sha'] = 'd' * 40
    if mutation == 'notes': release['body'] = 'Changed notes'
    if mutation == 'draft': release['draft'] = True
    if mutation == 'asset_identity': rows[0]['id'] += 1
    with pytest.raises(ValueError):
        audit.remote_publication(pub, specs, 'Notes', lambda p: responses[p])


def test_new_metadata_hashes_detect_mid_audit_mutation(tmp_path):
    path = write(tmp_path / 'source.json', {'value': 1})
    bindings = audit.Bindings(); bindings.read(path)
    write(path, {'value': 2})
    with pytest.raises(ValueError, match='during audit'):
        bindings.unchanged()


def package_fixture(root, monkeypatch):
    monkeypatch.setitem(audit.PHASES, 'selection', (2, 1))
    bindings = audit.Bindings()
    source = write(root / 'source.json', {'fixed': True})
    hashes = {'generation_registration': audit.sha(source), 'evaluation_protocol': audit.sha(source),
              'generation_manifest': audit.sha(source), 'generation_audit': audit.sha(source),
              'evaluation_manifest': audit.sha(source), 'measurement_audit': audit.sha(source)}
    for name in ['scripts/package_generated_outputs.py', 'src/mira_interp/array_storage.py']:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# fixture\n')
    native = {str(i): dict(record_id=str(i), generated_sha256=str(i) * 64, generated_artifact_path=str(root / f'private{i}.npz'),
                           match_id='m', clip_id='c', seed=i, intervention_type='baseline', dose=0, baseline_record_id=str(i), split='selection') for i in range(2)}
    records = [dict(record_id=k, source_path=v['generated_artifact_path'], source_sha256=v['generated_sha256'],
                    **{f: v[f] for f in ['match_id', 'clip_id', 'seed', 'intervention_type', 'dose', 'baseline_record_id', 'split']}) for k, v in native.items()]
    tar = root / 'part.tar'; tar.write_bytes(b'fake tar bytes')
    part = dict(status='passed_generated_artifact_part', completed=True, readback_verified=True, phase='selection', part_index=1,
                path=str(tar), filename=tar.name, bytes=tar.stat().st_size, sha256=audit.sha(tar), record_count=2)
    src = {k: dict(path=str(source), sha256=v) for k, v in hashes.items()}
    members = [{'kind': 'generated_future_npz', 'member': f'tensor{i}.npz', 'sha256': str(i) * 64} for i in range(2)]
    payload = {'schema_version': 1, 'phase': 'selection', 'part_index': 1, 'source_bindings': src,
               'members': members, 'observed_context_actions_simulator_labels_included': False}
    part['manifest_sha256'] = audit.hashlib.sha256((json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
    part['member_count'] = 3
    sidecar = write(root / 'part.tar.json', {**part, 'members': members})
    part['report'] = dict(path=str(sidecar), sha256=audit.sha(sidecar))
    manifest = write(root / 'manifest.json', dict(parts=[part], source_bindings=src, phase='selection',
                     all_registered_conditions_included=True, observed_context_actions_simulator_labels_included=False, records=records, members=members))
    package = dict(status='passed_generated_artifact_package', phase='selection', expected_records=2, packaged_records=2,
                   source_context_actions_simulator_labels_included=False, parts=[part], source_bindings=src,
                   archive_manifest=dict(path=str(manifest), sha256=audit.sha(manifest)),
                   script_sha256=audit.sha(root / 'scripts/package_generated_outputs.py'), array_storage_sha256=audit.sha(root / 'src/mira_interp/array_storage.py'))
    for key in ['all_exported_arrays_exact_source_slices', 'all_members_readback_verified', 'all_registered_conditions_included',
                'private_sources_unchanged', 'packaging_code_hashes_captured_at_start_and_unchanged']: package[key] = True
    for key, sourcekey in [('registration_sha256', 'generation_registration'), ('protocol_sha256', 'evaluation_protocol'),
                          ('generation_manifest_sha256', 'generation_manifest'), ('generation_audit_sha256', 'generation_audit'),
                          ('evaluation_sha256', 'evaluation_manifest'), ('measurement_audit_sha256', 'measurement_audit')]: package[key] = hashes[sourcekey]
    pp = write(root / 'results/generated_publication_selection_complete.json', package)
    waiter = write(root / 'waiter.json', dict(status='passed_generated_publication_waiter', phase='selection',
                   measurement_audit_sha256=hashes['measurement_audit'], report=dict(path=str(pp), sha256=audit.sha(pp))))
    review = dict(status='passed_generated_publication_independent_rehash', phase='selection', total_records=2,
                  all_2_registered_records_accounted=True, package_report=dict(path=str(pp), sha256=audit.sha(pp)),
                  archive_manifest=package['archive_manifest'], waiter_report=dict(path=str(waiter), sha256=audit.sha(waiter)),
                  part_count=1, total_bytes=part['bytes'], parts=[copy.deepcopy(part)])
    for key in ['all_archive_sha256_independently_recomputed', 'all_sizes_and_1_8GB_limits_checked', 'all_part_sidecar_hashes_checked',
                'archive_manifest_hash_checked', 'passed_package_and_waiter_gates_checked', 'packager_attests_all_members_readback_and_exact_future_slices',
                'packager_attests_no_source_context_actions_simulator_labels']: review[key] = True
    rp = write(root / 'results/generated_publication_selection_independent_review.json', review)
    return hashes, native, bindings, pp, rp, tar


def test_small_metadata_checked_and_part_rehash_explicit(tmp_path, monkeypatch):
    hashes, native, bindings, _, _, tar = package_fixture(tmp_path, monkeypatch)
    result = audit.package_and_review('selection', tmp_path, hashes, native, bindings, False)
    assert len(result[2]) == 4 and tar.resolve() not in bindings.hashes
    audit.package_and_review('selection', tmp_path, hashes, native, bindings, True)
    assert tar.resolve() in bindings.hashes


@pytest.mark.parametrize('mutation', ['missing_record', 'stale_review', 'size', 'source_labels', 'empty_part_review', 'missing_part_digest'])
def test_package_and_independent_review_cannot_be_bypassed(tmp_path, monkeypatch, mutation):
    hashes, native, bindings, pp, rp, tar = package_fixture(tmp_path, monkeypatch)
    if mutation == 'missing_record': native.pop('1')
    if mutation == 'stale_review':
        r = json.loads(rp.read_text()); r['package_report']['sha256'] = 'f' * 64; write(rp, r)
    if mutation == 'size': tar.write_bytes(b'changed file size')
    if mutation in ['empty_part_review', 'missing_part_digest']:
        r = json.loads(rp.read_text())
        if mutation == 'empty_part_review': r['parts'][0] = {}
        else: del r['parts'][0]['sha256']
        write(rp, r)
    if mutation == 'source_labels':
        p = json.loads(pp.read_text()); p['source_context_actions_simulator_labels_included'] = True; write(pp, p)
    with pytest.raises(ValueError):
        audit.package_and_review('selection', tmp_path, hashes, native, bindings, False)
