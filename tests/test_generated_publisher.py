"""Refuse incomplete packages and mismatched remote assets before publication."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('generated_publisher', SCRIPTS / 'publish_generated_package.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


def fixture_package(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path
    source = write('source.json', {'science': 'fixed'})
    archive = tmp_path / 'part0001.tar'
    archive.write_bytes(b'fixture archive bytes')
    part = dict(status='passed_generated_artifact_part', phase='pilot', completed=True,
                readback_verified=True, part_index=1, filename=archive.name, path=str(archive),
                bytes=archive.stat().st_size, sha256=pub.sha(archive), record_count=2)
    sidecar = write('part0001.tar.json', part)
    part['report'] = dict(path=str(sidecar), sha256=pub.sha(sidecar))
    bindings = {'source': dict(path=str(source), sha256=pub.sha(source))}
    manifest = write('archive_manifest.json', dict(parts=[part], phase='pilot',
                                                   records=[{'record_id':'a'}, {'record_id':'b'}],
                                                   source_bindings=bindings))
    report = dict(status='passed_generated_artifact_package', phase='pilot',
                  all_exported_arrays_exact_source_slices=True, all_members_readback_verified=True,
                  all_registered_conditions_included=True, private_sources_unchanged=True,
                  source_context_actions_simulator_labels_included=False,
                  packaged_records=2, expected_records=2, parts=[part], source_bindings=bindings,
                  archive_manifest=dict(path=str(manifest), sha256=pub.sha(manifest)))
    return write('package.json', report)


def test_complete_package_and_local_corruption(tmp_path):
    path = fixture_package(tmp_path)
    source, assets, digest = pub.validate_package(path)
    assert len(assets) == 4 and digest == pub.sha(path)
    (tmp_path / 'part0001.tar').write_bytes(b'corrupt archive bytes')
    with pytest.raises(ValueError, match='Local asset hash differs'):
        pub.validate_package(path)


@pytest.mark.parametrize('key,value', [
    ('all_members_readback_verified', False),
    ('all_exported_arrays_exact_source_slices', False),
    ('source_context_actions_simulator_labels_included', True),
    ('packaged_records', 1),
])
def test_reject_incomplete_or_source_payload_package(tmp_path, key, value):
    path = fixture_package(tmp_path)
    report = json.loads(path.read_text()); report[key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): pub.validate_package(path)


def test_reject_changed_prerequisite(tmp_path):
    path = fixture_package(tmp_path)
    (tmp_path / 'source.json').write_text('{"science":"changed"}')
    with pytest.raises(ValueError, match='frozen source changed'):
        pub.validate_package(path)


def test_reject_sidecar_disagreement(tmp_path):
    path = fixture_package(tmp_path)
    report = json.loads(path.read_text())
    manifest_path = Path(report['archive_manifest']['path'])
    manifest = json.loads(manifest_path.read_text())
    # Rebind the modified sidecar: semantic agreement must still fail.
    sidecar = tmp_path / 'part0001.tar.json'
    content = json.loads(sidecar.read_text()); content['readback_verified'] = False
    sidecar.write_text(json.dumps(content))
    report['parts'][0]['report']['sha256'] = pub.sha(sidecar)
    manifest['parts'] = report['parts']; manifest_path.write_text(json.dumps(manifest))
    report['archive_manifest']['sha256'] = pub.sha(manifest_path)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='sidecar disagrees'):
        pub.validate_package(path)


def test_remote_conflicts_and_incomplete_uploads_rejected():
    asset = dict(id=7, name='part.tar', size=33, digest='sha256:abc', state='uploaded',
                 browser_download_url='https://example.invalid/part.tar')
    spec = dict(name='part.tar', bytes=33, sha256='abc')
    assert pub.verify_remote(asset, spec) == asset
    for key, value in [('name', 'wrong.tar'), ('size', 34), ('digest', 'sha256:wrong'),
                       ('digest', None), ('state', 'starter')]:
        wrong = copy.deepcopy(asset); wrong[key] = value
        with pytest.raises(ValueError): pub.verify_remote(wrong, spec)


def test_tag_peeling_checks_actual_commit_and_rejects_noncommit(monkeypatch):
    objects = {'git/tags/outer': {'object': {'type':'tag', 'sha':'inner'}},
               'git/tags/inner': {'object': {'type':'commit', 'sha':'actual'}}}
    monkeypatch.setattr(pub, 'api', lambda path: objects[path])
    assert pub.peel_tag_object({'type':'commit','sha':'direct'}) == 'direct'
    assert pub.peel_tag_object({'type':'tag','sha':'outer'}) == 'actual'
    with pytest.raises(ValueError): pub.peel_tag_object({'type':'tree','sha':'wrong'})
    objects['git/tags/inner']['object'] = {'type':'tag','sha':'outer'}
    with pytest.raises(ValueError, match='cyclic'):
        pub.peel_tag_object({'type':'tag','sha':'outer'})
