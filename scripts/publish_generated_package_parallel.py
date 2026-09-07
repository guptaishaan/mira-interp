#!/usr/bin/env python3
"""Publish an audited package with at most three distinct concurrent uploads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

# Bind the imported implementation before loading it, then check it again.
INITIAL_CODE_BINDINGS = {
    str(Path(__file__).with_name(name).resolve()): hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    for name in ('publish_generated_package_parallel.py', 'publish_generated_package.py', 'publish_audited_archive.py')
}
import publish_generated_package as original

GH, REPO = original.GH, original.REPO
save, sha, require = original.save, original.sha, original.require


def check_code():
    require(all(sha(path) == digest for path, digest in INITIAL_CODE_BINDINGS.items()),
            'Publication code or imported helper changed')


check_code()


def normalized_body(value):
    return value.replace('\r\n', '\n').rstrip()


def check_release(release, tag, target, title, notes, expected_names, release_id=None):
    require(release['tag_name'] == tag and release['target_commitish'] == target,
            'Release tag or target differs')
    require(release['name'] == title and normalized_body(release['body']) == normalized_body(notes),
            'Release title or notes differ')
    require(release_id is None or release['id'] == release_id, 'Release identity changed')
    require(release['draft'] is False, 'Release must be publicly visible, not a draft')
    names = [a['name'] for a in release['assets']]
    require(len(names) == len(set(names)) and set(names) <= set(expected_names),
            'Unexpected or duplicate remote assets')
    return release


def remote_match(release, spec):
    rows = [a for a in release['assets'] if a['name'] == spec['name']]
    require(len(rows) <= 1, 'Duplicate remote asset names')
    return original.verify_remote(rows[0], spec) if rows else None


def stop_owned(process):
    """An exited leader can leave descendants; address the owned group as well."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    else:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.025)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=2)


def run_owned(command, deadline):
    """Bound API/create commands and clean their descendants on every exit."""
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'Command wall-clock budget expired before launch')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=remaining)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        stop_owned(process)


class GitHub:
    """Use bounded owned commands without changing the original helpers."""
    def __init__(self, deadline):
        self.deadline = deadline

    def run(self, arguments):
        return run_owned([GH, *arguments], min(self.deadline, time.monotonic() + 45))

    def api(self, endpoint):
        result = self.run(['api', 'repos/' + REPO + '/' + endpoint])
        require(result.returncode == 0, 'GitHub API request failed: ' + endpoint)
        return json.loads(result.stdout)

    def tag_commit(self, tag):
        result = self.run(['api', 'repos/' + REPO + '/git/ref/tags/' + tag])
        if result.returncode:
            require('404' in result.stderr, 'Cannot resolve release tag')
            return None
        obj, seen = json.loads(result.stdout)['object'], set()
        for _ in range(16):
            require(obj['sha'] not in seen, 'Cyclic release tag')
            seen.add(obj['sha'])
            if obj['type'] == 'commit':
                return obj['sha']
            require(obj['type'] == 'tag', 'Release tag does not point to a commit')
            obj = self.api('git/tags/' + obj['sha'])['object']
        raise ValueError('Release tag has excessive annotation depth')


def upload_missing(assets, tag, workers, fetch_release, unchanged, progress, deadline):
    """Verify all existing assets before mutation; never overwrite or delete one."""
    require(1 <= workers <= 3, 'Concurrent upload count must be between one and three')
    require(len({a['name'] for a in assets}) == len(assets), 'Duplicate local asset names')
    release = fetch_release()
    verified, pending, active = {}, [], {}
    for spec in assets:
        value = remote_match(release, spec)
        if value is None:
            pending.append(spec)
        else:
            verified[spec['name']] = value

    def update():
        progress([verified[a['name']] for a in assets if a['name'] in verified],
                 {name: process.pid for name, (process, _) in active.items()})

    update()
    try:
        while pending or active:
            require(time.monotonic() < deadline, 'Upload wall-clock budget expired')
            unchanged()
            for name, (process, spec) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                require(code == 0, f'Upload failed for {name}: exit {code}')
                value = remote_match(fetch_release(), spec)
                require(value is not None, 'Completed upload is missing remotely: ' + name)
                verified[name] = value
                stop_owned(process)
                del active[name]
                update()
            while pending and len(active) < workers:
                spec = pending.pop(0)
                # A previous attempt may have finished after initial inspection.
                value = remote_match(fetch_release(), spec)
                if value is not None:
                    verified[spec['name']] = value;update();continue
                unchanged()
                require(sha(spec['path']) == spec['sha256'] and Path(spec['path']).stat().st_size == spec['bytes'],
                        'Local upload source changed')
                require(time.monotonic() < deadline, 'Upload wall-clock budget expired before launch')
                process = subprocess.Popen([GH, 'release', 'upload', tag, spec['path'], '--repo', REPO],
                                           start_new_session=True)
                active[spec['name']] = (process, spec)
                update()
            if active:
                time.sleep(.1)
        return [verified[a['name']] for a in assets]
    finally:
        # These process groups were created here; unrelated uploads are untouched.
        errors = []
        for process, _ in active.values():
            try:
                stop_owned(process)
            except BaseException as error:
                errors.append(str(error))
        require(not errors, 'Owned upload cleanup failed: ' + '; '.join(errors))


def main():
    check_code()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package-report', 'notes-file', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('tag', 'target', 'title'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--timeout-hours', type=float, default=6)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Parallel publisher received signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    require(1 <= args.workers <= 3 and 0 < args.timeout_hours <= 24, 'Invalid upload resource budget')
    require(not args.report.exists(), 'Completed publication reports are immutable')
    started = time.monotonic()
    deadline = started + args.timeout_hours * 3600
    github = GitHub(deadline)
    source, assets, report_hash = original.validate_package(args.package_report)
    notes_hash, notes = sha(args.notes_file), args.notes_file.read_text()
    require(re.fullmatch('[0-9a-f]{40}', args.target) and github.api('commits/' + args.target)['sha'] == args.target,
            'An exact already-published target commit is required')
    request = dict(tag=args.tag, target_commit=args.target, title=args.title, phase=source['phase'],
                   package_report_sha256=report_hash, release_notes_sha256=notes_hash,
                   assets=assets, code_bindings=INITIAL_CODE_BINDINGS, workers=args.workers,
                   timeout_hours=args.timeout_hours)
    status_path = args.report.with_name(args.report.stem + '_status.json')
    if status_path.exists():
        require(json.loads(status_path.read_text())['request'] == request, 'Resumed publication inputs differ')
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), request=request,
                 verified_assets=[], active_uploads={})
    save(status_path, state)

    def unchanged():
        check_code()
        require(sha(args.package_report) == report_hash and sha(args.notes_file) == notes_hash,
                'Publication request changed')

    def progress(verified, active):
        state.update(stage='uploading', verified_assets=verified, active_uploads=active,
                     elapsed_seconds=time.monotonic() - started)
        save(status_path, state)
        print(json.dumps({'verified': len(verified), 'total': len(assets), 'active': list(active)}), flush=True)

    try:
        unchanged()
        existing_commit = github.tag_commit(args.tag)
        require(existing_commit is None or existing_commit == args.target, 'Existing tag resolves to another commit')
        probe = github.run(['api', 'repos/' + REPO + '/releases/tags/' + args.tag])
        if probe.returncode:
            require('404' in probe.stderr, 'Cannot establish whether the release exists')
            unchanged()
            created = github.run(['release', 'create', args.tag, '--repo', REPO, '--target', args.target,
                                  '--title', args.title, '--notes-file', str(args.notes_file)])
            require(created.returncode == 0, 'Release creation failed')
            release = github.api('releases/tags/' + args.tag)
        else:
            release = json.loads(probe.stdout)
        expected_names = {a['name'] for a in assets}
        check_release(release, args.tag, args.target, args.title, notes, expected_names)
        release_id = release['id']

        def fetch_release():
            require(github.tag_commit(args.tag) == args.target, 'Actual release tag changed')
            return check_release(github.api('releases/tags/' + args.tag), args.tag, args.target, args.title,
                                 notes, expected_names, release_id)

        upload_missing(assets, args.tag, args.workers, fetch_release, unchanged, progress,
                       deadline)
        release = fetch_release()
        remote = {a['name']: a for a in release['assets']}
        require(set(remote) == expected_names, 'Final release asset set differs')
        verified = [original.verify_remote(remote[a['name']], a) for a in assets]
        _, final_assets, final_hash = original.validate_package(args.package_report)
        require(final_assets == assets and final_hash == report_hash, 'Package changed during upload')
        unchanged()
        # Repeat the release header/tag check after the potentially long local rehash.
        release = fetch_release()
        require({a['name'] for a in release['assets']} == expected_names, 'Final release asset set changed')
        verified = [remote_match(release, spec) for spec in assets]
        unchanged()
        result = dict(status='passed', verified_utc=datetime.now(timezone.utc).isoformat(), phase=source['phase'],
                      packaged_records=source['packaged_records'], release_url=release['html_url'], release_id=release_id,
                      target_commit=args.target, assets=verified, all_remote_sizes_and_sha256_equal_local=True,
                      actual_release_tag_commit_verified=True, release_title_and_body_verified_before_and_after=True,
                      package_report_sha256=report_hash, release_notes_sha256=notes_hash,
                      code_bindings=INITIAL_CODE_BINDINGS, max_concurrent_uploads=args.workers,
                      elapsed_seconds=time.monotonic() - started)
        save(args.report, result)
        state.update(status='passed', stage='complete', active_uploads={}, verified_assets=verified,
                     publication_report_sha256=sha(args.report))
        save(status_path, state)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        state.update(status='failed', error_type=type(error).__name__, error=str(error), active_uploads={})
        save(status_path, state)
        raise


if __name__ == '__main__':
    main()
