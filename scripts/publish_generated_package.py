#!/usr/bin/env python3
"""Publish every part of an audited generated-output package, with remote hashes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from publish_audited_archive import GH, REPO, api, save, sha


def require(value, message):
    if not value:
        raise ValueError(message)


def asset_spec(path, digest=None, size=None):
    path = Path(path).resolve()
    actual_size, actual_hash = path.stat().st_size, sha(path)
    require(0 < actual_size <= 1_800_000_000, "Asset outside the publication size bound")
    require(digest is None or actual_hash == digest, "Local asset hash differs: " + path.name)
    require(size is None or actual_size == size, "Local asset size differs: " + path.name)
    return dict(path=str(path), name=path.name, bytes=actual_size, sha256=actual_hash)


def validate_package(report_path):
    report_path = Path(report_path).resolve()
    report_hash = sha(report_path)
    source = json.loads(report_path.read_text())
    require(source['status'] == 'passed_generated_artifact_package', 'Package did not pass')
    for key in ('all_exported_arrays_exact_source_slices', 'all_members_readback_verified',
                'all_registered_conditions_included', 'private_sources_unchanged'):
        require(source[key] is True, 'Incomplete package check: ' + key)
    require(source['source_context_actions_simulator_labels_included'] is False,
            'Publication must exclude source dataset payloads')
    expected = {'pilot': 2, 'selection': 420, 'confirmation': 924}[source['phase']]
    require(source['packaged_records'] == source['expected_records'] == expected,
            'Incomplete registered condition count')
    manifest_spec = asset_spec(source['archive_manifest']['path'], source['archive_manifest']['sha256'])
    manifest = json.loads(Path(manifest_spec['path']).read_text())
    require(manifest['parts'] == source['parts'] and manifest['phase'] == source['phase'],
            'Package and external manifest disagree')
    require(len(manifest['records']) == expected and source['parts'], 'Missing records or parts')
    require(len({record['record_id'] for record in manifest['records']}) == expected,
            'Duplicate manifest record IDs')
    require(sum(part['record_count'] for part in source['parts']) == expected,
            'Part record counts do not cover the phase')
    require(manifest['source_bindings'] == source['source_bindings'], 'Source bindings differ')
    for binding in source['source_bindings'].values():
        require(sha(binding['path']) == binding['sha256'], 'A frozen source changed')
    assets = []
    for index, part in enumerate(source['parts'], 1):
        require(part['status'] == 'passed_generated_artifact_part' and
                part['completed'] is True and part['readback_verified'] is True,
                'An archive part did not pass readback')
        require(part['phase'] == source['phase'] and part['part_index'] == index,
                'Phase or part order differs')
        require(Path(part['path']).name == part['filename'], 'Part filename differs')
        sidecar_spec = asset_spec(part['report']['path'], part['report']['sha256'])
        sidecar = json.loads(Path(sidecar_spec['path']).read_text())
        require(all(sidecar.get(key) == value for key, value in part.items() if key != 'report'),
                'Part sidecar disagrees with package')
        assets.extend([asset_spec(part['path'], part['sha256'], part['bytes']), sidecar_spec])
    assets.extend([manifest_spec, asset_spec(report_path, report_hash)])
    require(len({a['name'] for a in assets}) == len(assets), 'Duplicate release asset names')
    require(sha(report_path) == report_hash, 'Package report changed during validation')
    return source, assets, report_hash


def verify_remote(asset, spec):
    require(asset['name'] == spec['name'] and asset['state'] == 'uploaded' and
            asset['size'] == spec['bytes'] and asset.get('digest') == 'sha256:' + spec['sha256'],
            'Remote asset state, size or digest differs: ' + spec['name'])
    return {key: asset[key] for key in ('id', 'name', 'size', 'digest', 'state', 'browser_download_url')}


def peel_tag_object(obj):
    """Resolve an annotated tag chain to its actual commit, with a cycle bound."""
    for _ in range(16):
        if obj['type'] == 'commit':
            return obj['sha']
        require(obj['type'] == 'tag', 'Release tag does not point to a commit')
        obj = api('git/tags/' + obj['sha'])['object']
    raise ValueError('Release tag has an excessive or cyclic annotation chain')


def tag_commit(tag):
    probe = subprocess.run([GH, 'api', 'repos/' + REPO + '/git/ref/tags/' + tag],
                           capture_output=True, text=True)
    if probe.returncode:
        require('404' in probe.stderr, 'Cannot resolve release tag')
        return None
    return peel_tag_object(json.loads(probe.stdout)['object'])


def main():
    code_bindings = {str(Path(__file__).resolve()): sha(__file__),
                     str(Path(__file__).with_name('publish_audited_archive.py').resolve()):
                     sha(Path(__file__).with_name('publish_audited_archive.py'))}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-report', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--notes-file', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), 'Completed publication reports are immutable')
    source, assets, report_hash = validate_package(args.package_report)
    notes_hash = sha(args.notes_file)
    require(re.fullmatch('[0-9a-f]{40}', args.target) and
            api('commits/' + args.target)['sha'] == args.target,
            'An exact already-published target commit is required')
    request = dict(tag=args.tag, target_commit=args.target, title=args.title,
                   package_report_sha256=report_hash, release_notes_sha256=notes_hash,
                   phase=source['phase'], assets=assets, code_bindings=code_bindings)
    status_path = args.report.with_name(args.report.stem + '_status.json')
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 request=request, verified_assets=[])
    if status_path.exists():
        old = json.loads(status_path.read_text())
        require(old['request'] == request, 'Resumed publication inputs differ')
    save(status_path, state)
    try:
        existing_commit = tag_commit(args.tag)
        require(existing_commit is None or existing_commit == args.target,
                'Existing tag resolves to another commit')
        probe = subprocess.run([GH, 'api', 'repos/' + REPO + '/releases/tags/' + args.tag],
                               capture_output=True, text=True)
        if probe.returncode:
            require('404' in probe.stderr, 'Cannot establish whether the release exists')
            subprocess.run([GH, 'release', 'create', args.tag, '--repo', REPO,
                            '--target', args.target, '--title', args.title,
                            '--notes-file', str(args.notes_file)], check=True)
            release = api('releases/tags/' + args.tag)
        else:
            release = json.loads(probe.stdout)
        require(release['target_commitish'] == args.target, 'Release points at another commit')
        require(tag_commit(args.tag) == args.target, 'Actual release tag resolves to another commit')
        require(release['name'] == args.title and
                release['body'].replace('\r\n', '\n').rstrip() ==
                args.notes_file.read_text().replace('\r\n', '\n').rstrip(),
                'Existing release title or notes differ')
        for spec in assets:
            require(sha(spec['path']) == spec['sha256'] and
                    Path(spec['path']).stat().st_size == spec['bytes'], 'Local upload source changed')
            matching = [a for a in release['assets'] if a['name'] == spec['name']]
            require(len(matching) <= 1, 'Duplicate remote asset names')
            if not matching:
                state.update(stage='uploading', current_asset=spec['name'])
                save(status_path, state)
                subprocess.run([GH, 'release', 'upload', args.tag, spec['path'], '--repo', REPO], check=True)
                release = api('releases/tags/' + args.tag)
                matching = [a for a in release['assets'] if a['name'] == spec['name']]
            require(len(matching) == 1, 'Uploaded asset is missing')
            state['verified_assets'].append(verify_remote(matching[0], spec))
            save(status_path, state)
            print(json.dumps(dict(verified_asset=spec['name'], bytes=spec['bytes'],
                                  completed=len(state['verified_assets']), total=len(assets))), flush=True)
        # Check the final remote set and all local inputs once more.
        release = api('releases/tags/' + args.tag)
        require(tag_commit(args.tag) == args.target, 'Actual release tag changed during publication')
        remote = {a['name']: a for a in release['assets']}
        require(set(remote) == {a['name'] for a in assets}, 'Release has unexpected or missing assets')
        verified = [verify_remote(remote[a['name']], a) for a in assets]
        _, final_assets, final_hash = validate_package(args.package_report)
        require(final_assets == assets and final_hash == report_hash and
                sha(args.notes_file) == notes_hash, 'Publication inputs changed during upload')
        require(all(sha(path) == digest for path, digest in code_bindings.items()),
                'Publication code changed during upload')
        result = dict(status='passed', verified_utc=datetime.now(timezone.utc).isoformat(),
                      phase=source['phase'], packaged_records=source['packaged_records'],
                      release_url=release['html_url'], release_id=release['id'],
                      target_commit=args.target, assets=verified,
                      all_remote_sizes_and_sha256_equal_local=True,
                      actual_release_tag_commit_verified=True,
                      package_report_sha256=report_hash, release_notes_sha256=notes_hash,
                      code_bindings=code_bindings)
        save(args.report, result)
        state.update(status='passed', stage='complete', publication_report_sha256=sha(args.report))
        save(status_path, state)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        save(status_path, state)
        raise


if __name__ == '__main__':
    main()
