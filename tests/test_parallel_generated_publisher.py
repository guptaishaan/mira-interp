"""Exercise actual local subprocess concurrency, cleanup and verified resume."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('parallel_publisher', SCRIPTS / 'publish_generated_package_parallel.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


@pytest.fixture
def fake_network(tmp_path, monkeypatch):
    executable = tmp_path / 'fake-gh'
    executable.write_text('#!' + sys.executable + '\n' + r'''
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['TASK_FAKE_UPLOAD_ROOT'])
if sys.argv[1:2]==['api'] or sys.argv[1:3]==['release','create']:
    (root/'metadata_pid').write_text(str(os.getpid()));time.sleep(30);sys.exit(0)
source=Path(sys.argv[4])
name=source.name; cfg=json.loads(source.read_text())
def event(kind):
    with (root/'events.jsonl').open('a') as f:
        f.write(json.dumps({'kind':kind,'name':name,'pid':os.getpid()})+'\n')
def stopped(signum,frame):
    event('stop');sys.exit(128+signum)
signal.signal(signal.SIGTERM,stopped)
event('start');time.sleep(cfg['delay'])
if cfg.get('orphan'):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    (root/'orphan_pid').write_text(str(child.pid));event('fail');sys.exit(7)
failed=root/('failed_once_'+name)
if cfg.get('fail_once') and not failed.exists():
    failed.touch();event('fail');sys.exit(7)
digest=hashlib.sha256(source.read_bytes()).hexdigest()
asset={'id':int(digest[:7],16),'name':name,'size':source.stat().st_size,
       'digest':'sha256:'+digest,'state':'uploaded','browser_download_url':'https://example.invalid/'+name}
temporary=root/'remote'/(name+'.tmp');temporary.write_text(json.dumps(asset))
temporary.replace(root/'remote'/(name+'.json'));event('end')
''')
    executable.chmod(0o755)
    (tmp_path / 'remote').mkdir()
    monkeypatch.setenv('TASK_FAKE_UPLOAD_ROOT', str(tmp_path))
    monkeypatch.setattr(pub, 'GH', str(executable))
    def specs(configs):
        result = []
        for index, config in enumerate(configs):
            p = tmp_path / f'asset{index}.tar';p.write_text(json.dumps(config))
            result.append(pub.original.asset_spec(p))
        return result
    def fetch():
        return {'assets': [json.loads(p.read_text()) for p in (tmp_path / 'remote').glob('*.json')]}
    def events():
        p = tmp_path / 'events.jsonl'
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    return tmp_path, specs, fetch, events


def run(assets, fetch, progress=lambda verified, active: None, timeout=10):
    return pub.upload_missing(assets, 'fixture', 3, fetch, lambda: None, progress, time.monotonic() + timeout)


def assert_owned_processes_ended(events):
    for row in events:
        if row['kind'] == 'start':
            with pytest.raises(ProcessLookupError):
                os.kill(row['pid'], 0)


def test_real_upload_concurrency_is_three_and_assets_are_distinct(fake_network):
    _, specs, fetch, events = fake_network
    assets = specs([{'delay': .25}] * 7)
    observed = []
    verified = run(assets, fetch, lambda done, active: observed.append(len(active)))
    active, maximum = set(), 0
    for row in events():
        if row['kind'] == 'start':
            assert row['name'] not in active
            active.add(row['name']);maximum = max(maximum, len(active))
        else:
            active.remove(row['name'])
    assert maximum == max(observed) == 3 and not active
    assert len(verified) == 7 and {a['name'] for a in verified} == {a['name'] for a in assets}
    assert len([r for r in events() if r['kind'] == 'start']) == 7
    assert_owned_processes_ended(events())


def test_verified_resume_does_not_reupload_and_conflict_precedes_mutation(fake_network):
    root, specs, fetch, events = fake_network
    assets = specs([{'delay': .1}] * 4)
    run(assets, fetch)
    before = events()
    assert len(run(assets, fetch)) == 4 and events() == before
    # Missing first asset must not launch if a later existing asset conflicts.
    (root / 'remote/asset0.tar.json').unlink()
    p = root / 'remote/asset3.tar.json';value = json.loads(p.read_text());value['digest'] = 'sha256:wrong'
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='digest differs'):
        run(assets, fetch)
    assert events() == before


def test_failure_stops_only_owned_uploads_then_unchanged_request_resumes(fake_network):
    _, specs, fetch, events = fake_network
    assets = specs([{'delay': .3, 'fail_once': True}, {'delay': 1}, {'delay': 1}, {'delay': .1}])
    with pytest.raises(ValueError, match='Upload failed.*exit 7'):
        run(assets, fetch)
    before = events()
    assert {r['name'] for r in before if r['kind'] == 'start'} == {'asset0.tar', 'asset1.tar', 'asset2.tar'}
    assert len([r for r in before if r['kind'] == 'stop']) == 2
    assert_owned_processes_ended(before)
    assert len(run(assets, fetch)) == 4
    assert_owned_processes_ended(events())


def test_timeout_cleans_active_uploads_without_launching_remaining(fake_network):
    _, specs, fetch, events = fake_network
    assets = specs([{'delay': 5}] * 4)
    with pytest.raises(ValueError, match='budget expired'):
        run(assets, fetch, timeout=.3)
    rows = events()
    assert len([r for r in rows if r['kind'] == 'start']) == 3
    assert len([r for r in rows if r['kind'] == 'stop']) == 3
    assert_owned_processes_ended(rows)


@pytest.mark.parametrize('field,value', [('tag_name', 'changed'), ('target_commitish', 'changed'),
                                        ('name', 'changed'), ('body', 'changed'), ('id', 8),
                                        ('draft', True),
                                        ('assets', [{'name': 'unexpected'}]),
                                        ('assets', [{'name': 'part'}, {'name': 'part'}])])
def test_release_headers_identity_and_asset_set_are_strict(field, value):
    release = dict(tag_name='tag', target_commitish='commit', name='title', body='notes', id=7, assets=[], draft=False)
    assert pub.check_release(release, 'tag', 'commit', 'title', 'notes', {'part'}, 7) == release
    release[field] = value
    with pytest.raises(ValueError):
        pub.check_release(release, 'tag', 'commit', 'title', 'notes', {'part'}, 7)


def test_reject_more_than_three_workers_before_network():
    with pytest.raises(ValueError, match='between one and three'):
        pub.upload_missing([], 'tag', 4, lambda: pytest.fail('must not call network'),
                           lambda: None, lambda *args: None, time.monotonic() + 1)


def test_failed_leader_cannot_leave_running_descendant(fake_network):
    root, specs, fetch, _ = fake_network
    assets = specs([{'delay': .1, 'orphan': True}])
    with pytest.raises(ValueError, match='Upload failed'):
        run(assets, fetch)
    pid = int((root / 'orphan_pid').read_text())
    status = Path('/proc') / str(pid) / 'status'
    assert not status.exists() or any(line.startswith('State:') and 'Z' in line for line in status.read_text().splitlines())


@pytest.mark.parametrize('arguments', [['api', 'repos/fixture'], ['release', 'create', 'fixture']])
def test_hanging_metadata_and_create_commands_are_bounded_and_cleaned(fake_network, arguments):
    root, _, _, _ = fake_network
    start = time.monotonic()
    with pytest.raises(pub.subprocess.TimeoutExpired):
        pub.GitHub(time.monotonic() + .2).run(arguments)
    assert time.monotonic() - start < 3
    pid = int((root / 'metadata_pid').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_bounded_tag_resolver_checks_annotation_chain_and_404(monkeypatch):
    github = pub.GitHub(time.monotonic() + 10)
    responses = {'repos/' + pub.REPO + '/git/ref/tags/tag': {'object': {'type': 'tag', 'sha': 'outer'}},
                 'repos/' + pub.REPO + '/git/tags/outer': {'object': {'type': 'commit', 'sha': 'target'}}}
    monkeypatch.setattr(github, 'run', lambda args: pub.subprocess.CompletedProcess(args, 0, json.dumps(responses[args[1]]), ''))
    assert github.tag_commit('tag') == 'target'
    responses['repos/' + pub.REPO + '/git/tags/outer']['object'] = {'type': 'tag', 'sha': 'outer'}
    with pytest.raises(ValueError, match='Cyclic'):
        github.tag_commit('tag')
    monkeypatch.setattr(github, 'run', lambda args: pub.subprocess.CompletedProcess(args, 1, '', '404'))
    assert github.tag_commit('missing') is None
    monkeypatch.setattr(github, 'run', lambda args: pub.subprocess.CompletedProcess(args, 1, '', 'authentication failure'))
    with pytest.raises(ValueError, match='Cannot resolve'):
        github.tag_commit('tag')


def test_different_resume_request_preserves_old_state_without_upload(tmp_path, monkeypatch):
    report = tmp_path / 'publication.json';status = tmp_path / 'publication_status.json'
    status.write_text(json.dumps({'request': {'different': 'request'}}));prior = status.read_bytes()
    notes = tmp_path / 'notes.md';notes.write_text('Notes')
    monkeypatch.setattr(pub.original, 'validate_package', lambda path: ({'phase': 'pilot'}, [], 'hash'))
    monkeypatch.setattr(pub.GitHub, 'api', lambda self, path: {'sha': 'a' * 40})
    monkeypatch.setattr(pub.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Must not upload or create a release'))
    monkeypatch.setattr(sys, 'argv', ['parallel', '--package-report', str(tmp_path / 'package.json'), '--notes-file', str(notes),
                                     '--report', str(report), '--target', 'a' * 40, '--tag', 'tag', '--title', 'Title'])
    previous = pub.signal.getsignal(pub.signal.SIGTERM)
    try:
        with pytest.raises(ValueError, match='Resumed publication inputs differ'):
            pub.main()
    finally:
        pub.signal.signal(pub.signal.SIGTERM, previous)
    assert status.read_bytes() == prior and not report.exists()
