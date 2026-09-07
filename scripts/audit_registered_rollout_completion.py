#!/usr/bin/env python3
"""Read-only final joins of the registered rollout study and its GitHub artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPO = 'guptaishaan/mira-interp'
GH = '/ccn2/u/ishaangp/bin/gh'
PHASES = {'selection': (420, 10), 'confirmation': (924, 22)}
STAGES = [('generation', 'passed_rollout_generation'),
          ('generation_audit', 'passed_rollout_generation_audit'),
          ('video_measurement', 'passed_frozen_generated_video_evaluation'),
          ('measurement_audit', 'passed_generated_measurement_audit')]
PATHS = ['probe_linear', 'affine_forward', 'quadratic_forward', 'random_norm', 'wrong_variable_ball_x']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Bindings:
    """Hash small metadata/derived predictions; never silently hash raw source arrays."""
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, f'Hash differs: {path}')
        require(path not in self.hashes or self.hashes[path] == digest, f'Input changed: {path}')
        self.hashes[path] = digest
        return digest

    def read(self, path, expected=None):
        self.bind(path, expected)
        return json.loads(Path(path).read_text())

    def check(self, spec, expected_path=None):
        if expected_path is not None:
            require(Path(spec['path']).resolve() == Path(expected_path).resolve(), 'Bound path differs')
        return self.bind(spec['path'], spec['sha256'])

    def unchanged(self):
        for path, digest in self.hashes.items():
            require(sha(path) == digest, f'Input changed during audit: {path}')


def github_api(endpoint):
    """GET only. No upload, release creation, or remote mutation is possible here."""
    result = subprocess.run([GH, 'api', '--method', 'GET', 'repos/' + REPO + '/' + endpoint],
                            capture_output=True, text=True, timeout=45)
    require(result.returncode == 0, 'Read-only GitHub API request failed: ' + endpoint)
    return json.loads(result.stdout)


def tag_commit(tag, api):
    obj = api('git/ref/tags/' + quote(tag, safe=''))['object']
    seen = set()
    for _ in range(16):
        require(obj['sha'] not in seen, 'Cyclic GitHub tag')
        seen.add(obj['sha'])
        if obj['type'] == 'commit':
            return obj['sha']
        require(obj['type'] == 'tag', 'Tag does not resolve to a commit')
        obj = api('git/tags/' + obj['sha'])['object']
    raise ValueError('Excessive annotated tag depth')


def remote_publication(publication, assets, notes, api):
    url = urlsplit(publication['release_url'])
    prefix = '/' + REPO + '/releases/tag/'
    require(url.scheme == 'https' and url.netloc == 'github.com' and url.path.startswith(prefix)
            and not url.query and not url.fragment, 'Unexpected release repository or URL')
    tag = url.path[len(prefix):]
    require(bool(tag) and re.fullmatch('[0-9a-f]{40}', publication['target_commit']), 'Invalid tag/target')
    release = api('releases/tags/' + quote(tag, safe=''))
    require(release['id'] == publication['release_id'] and release['html_url'] == publication['release_url']
            and release['tag_name'] == tag and release['target_commitish'] == publication['target_commit']
            and not release['draft'], 'Release identity, target or visibility differs')
    require(tag_commit(tag, api) == publication['target_commit'], 'Actual tag commit differs')
    require(release['body'].replace('\r\n', '\n').rstrip() == notes.replace('\r\n', '\n').rstrip(),
            'Release notes differ')
    actual = []
    for page in range(1, 101):
        chunk = api(f"releases/{release['id']}/assets?per_page=100&page={page}")
        actual.extend(chunk)
        if len(chunk) < 100:
            break
    else:
        raise ValueError('Unbounded release asset list')
    def index(rows):
        require(len({x['name'] for x in rows}) == len(rows), 'Duplicate release asset names')
        return {x['name']: x for x in rows}
    remote, saved, expected = index(actual), index(publication['assets']), index(assets)
    require(set(remote) == set(saved) == set(expected), 'Remote/saved/expected asset set differs')
    verified = []
    for name, spec in expected.items():
        row, old = remote[name], saved[name]
        require(row['state'] == 'uploaded' and row['size'] == spec['bytes']
                and row.get('digest') == 'sha256:' + spec['sha256'], 'Remote digest/size/state differs: ' + name)
        keys = ['id', 'name', 'size', 'digest', 'state', 'browser_download_url']
        require(all(row[k] == old[k] for k in keys), 'Remote asset changed since publication: ' + name)
        verified.append({k: row[k] for k in keys})
    require(tag_commit(tag, api) == publication['target_commit'], 'Tag changed during API checks')
    return {'release_url': release['html_url'], 'tag': tag, 'target_commit': publication['target_commit'],
            'assets': verified, 'all_remote_sizes_and_digests_current': True, 'actual_tag_commit_checked': True}


def complete_study(study, root, bindings):
    require(study['status'] == 'passed_generation_and_measurement_pending_publication'
            and study['current_stage'] is None and study['physical_control_established'] is False,
            'Study is incomplete or failed; no completion report will be written')
    expected = [phase + '_' + name for phase in PHASES for name, _ in STAGES]
    require([r['stage'] for r in study['completed_stages']] == expected, 'Exactly eight ordered completed stages required')
    require(bool(study['fixed_bindings']), 'Missing supervisor bindings')
    for path, digest in study['fixed_bindings'].items():
        bindings.bind(path, digest)
    reports = {}
    for row, (phase, name, status) in zip(study['completed_stages'],
            [(p, n, s) for p in PHASES for n, s in STAGES]):
        directory = 'rollout_steering_v2' if name.startswith('generation') else 'generated_evaluation_v2'
        filename = phase + ('_audit' if name.endswith('audit') else '') + '.json'
        path = root / 'results' / directory / filename
        bindings.check(row, path)
        report = bindings.read(path)
        require(report['status'] == status, 'A completed stage no longer has passed status')
        reports[phase, name] = report
    return reports


def check_phase(phase, root, reports, registration, protocol, bindings):
    count, matches = PHASES[phase]
    gen, ga, ev, ea = [reports[phase, name] for name, _ in STAGES]
    gd, ed = root / 'results/rollout_steering_v2', root / 'results/generated_evaluation_v2'
    hashes = {'generation_manifest': bindings.bind(gd / (phase + '.json')),
              'generation_audit': bindings.bind(gd / (phase + '_audit.json')),
              'evaluation_manifest': bindings.bind(ed / (phase + '.json')),
              'measurement_audit': bindings.bind(ed / (phase + '_audit.json')),
              'generation_registration': bindings.bind(gd / 'registration.json'),
              'evaluation_protocol': bindings.bind(root / 'configs/generated_evaluation_v1.json')}
    require(gen['n_records'] == ga['n_records'] == ev['records'] == ea['records_checked'] == count
            and gen['match_count'] == ga['n_matches'] == ev['matches'] == ea['matches'] == matches
            and gen['seed_count'] == ga['n_seeds'] == ev['paired_seeds_per_match'] == 2, 'Phase count differs')
    require(gen['registration_sha256'] == ga['registration_sha256'] == hashes['generation_registration']
            and ga['manifest_sha256'] == ev['generation_manifest_sha256'] == ea['generation_manifest_sha256'] == hashes['generation_manifest']
            and ev['generation_audit_sha256'] == ea['generation_audit_sha256'] == hashes['generation_audit']
            and ea['evaluation_sha256'] == hashes['evaluation_manifest']
            and ev['protocol_sha256'] == ea['protocol_sha256'] == hashes['evaluation_protocol'], 'Stage provenance join differs')
    for report in [ga, ev, ea]:
        require(report['phase'] == phase and report['physical_control_established'] is False, 'Phase or claim boundary differs')
    require(gen['physical_control_established'] is False and not ev['generated_video_accuracy_established']
            and not ea['generated_video_accuracy_established'] and not ev['any_context_pixels_changed'], 'Unsupported claim/context change')
    for key in ['complete_registered_grid', 'source_report_and_generated_hashes_verified', 'exact_action_conditioning_preserved',
                'preintervention_sampler_inputs_bitwise_equal', 'fixed_context_latents_unchanged',
                'all_stored_context_pixels_exact', 'all45_sampler_calls_validated', 'saved_tile_descriptor_lift_metrics_recomputed']:
        require(ga[key] is True, 'Generation audit did not pass: ' + key)
    for key in ['all_cache_predictions_exact', 'all_record_source_seed_joins_exact', 'all_absolute_ball_coordinates_recomputed',
                'frozen_calibration_scale_exact', 'all_summary_metrics_independently_recomputed']:
        require(ea[key] is True, 'Measurement audit did not pass: ' + key)
    bindings.bind(root / 'scripts/audit_rollout_generation.py', ga['script_sha256'])
    bindings.bind(root / 'scripts/audit_generated_measurements.py', ea['audit_script_sha256'])
    require(ea['checkpoint_sha256'] == protocol['checkpoint_sha256'], 'Evaluator checkpoint differs')
    require(ea['prediction_artifact_sha256'] == bindings.check(ev['prediction_artifact']), 'Prediction artifact differs')
    chosen = [x for x in registration['records'] if x['role'] == phase]
    require(len(chosen) == len({x['match_id'] for x in chosen}) == matches, 'Registration match cohort differs')
    expected = {(x['match_id'], x['clip_id'], seed, path, dose) for x in chosen
                for seed in registration['seeds'] for path, dose in registration['condition_grid']}
    identities = [(x['match_id'], x['clip_id'], x['seed'], x['intervention_type'], x['dose']) for x in gen['records']]
    require(len(identities) == len(set(identities)) == count and set(identities) == expected, 'Actual registered condition grid differs')
    native = {x['record_id']: x for x in gen['records']}
    require(len(native) == count, 'Duplicate generated record ID')
    for rows in [ga['records'], ea['cache_records']]:
        require(len(rows) == len({x['record_id'] for x in rows}) == count and {x['record_id'] for x in rows} == set(native),
                'Audit record IDs differ')
        require(all(x['generated_sha256'] == native[x['record_id']]['generated_sha256'] for x in rows), 'Generated source digest joins differ')
    summary = [(x['path'], x['time_index']) for x in ev['summary']]
    require(len(summary) == 20 and set(summary) == {(p, t) for p in PATHS for t in range(4)}, 'Missing or duplicated path/time summary')
    export_path = ed / (phase + '_figures/export.json')
    export = bindings.read(export_path)
    require(export['status'] == 'passed_audited_generated_measurement_export' and export['phase'] == phase, 'Export incomplete')
    for key, source in [('measurement_audit_sha256', 'measurement_audit'), ('evaluation_sha256', 'evaluation_manifest'),
                        ('protocol_sha256', 'evaluation_protocol'), ('generation_manifest_sha256', 'generation_manifest')]:
        require(export[key] == hashes[source], 'Export source differs: ' + key)
    require(export['prediction_artifact_sha256'] == ea['prediction_artifact_sha256'], 'Export predictions differ')
    require(export['all_five_paths_four_times_retained'] and export['requested_dose_grouping_unchanged']
            and not export['negative_cases_omitted'] and not export['undefined_ranks_imputed']
            and not export['physical_control_established'] and not export['model_or_probe_fitting_performed'], 'Incomplete/altered export')
    for spec in export['artifact_bindings']:
        bindings.check(spec)
    stems = ['dose_response_requested', 'dose_response_effective', 'contrasts_controls_nuisance', 'ordering_and_undefined']
    require(set(export['figures']) == {s + ext for s in stems for ext in ['.png', '.pdf']}, 'Exact four PNG/PDF figure pairs required')
    expected_csv = {'summary.csv': 20, 'per_pair_time.csv': matches * 40, 'per_pair_dose.csv': matches * 200,
                    'per_match_time.csv': matches * 20, 'per_match_dose.csv': matches * 100, 'dose_response.csv': 100}
    require(set(export['csv']) == set(expected_csv), 'Exact six exported tables required')
    for name, spec in {**export['figures'], **export['csv']}.items():
        path = export_path.parent / name
        bindings.bind(path, spec['sha256'])
        require(path.stat().st_size == spec['bytes'], 'Export size differs')
        if name in expected_csv:
            require(spec['rows'] == expected_csv[name], 'Export row count differs')
    return hashes, native, export_path


def package_and_review(phase, root, hashes, native, bindings, rehash_parts):
    count, _ = PHASES[phase]
    path = root / f'results/generated_publication_{phase}_complete.json'
    package = bindings.read(path)
    review_path = root / f'results/generated_publication_{phase}_independent_review.json'
    review = bindings.read(review_path)
    require(package['status'] == 'passed_generated_artifact_package' and package['phase'] == phase
            and package['expected_records'] == package['packaged_records'] == count, 'Incomplete phase package')
    for key in ['all_exported_arrays_exact_source_slices', 'all_members_readback_verified',
                'all_registered_conditions_included', 'private_sources_unchanged', 'packaging_code_hashes_captured_at_start_and_unchanged']:
        require(package[key] is True, 'Package guard failed: ' + key)
    require(package['source_context_actions_simulator_labels_included'] is False, 'Package includes gated source payloads')
    bindings.bind(root / 'scripts/package_generated_outputs.py', package['script_sha256'])
    bindings.bind(root / 'src/mira_interp/array_storage.py', package['array_storage_sha256'])
    require(set(package['source_bindings']) == set(hashes), 'Package provenance set differs')
    for key, digest in hashes.items():
        require(bindings.check(package['source_bindings'][key]) == digest, 'Package source digest differs')
    for key, source in [('registration_sha256', 'generation_registration'), ('protocol_sha256', 'evaluation_protocol'),
                        ('generation_manifest_sha256', 'generation_manifest'), ('generation_audit_sha256', 'generation_audit'),
                        ('evaluation_sha256', 'evaluation_manifest'), ('measurement_audit_sha256', 'measurement_audit')]:
        require(package[key] == hashes[source], 'Package cross-binding differs')
    manifest = bindings.read(package['archive_manifest']['path'], package['archive_manifest']['sha256'])
    require(manifest['parts'] == package['parts'] and manifest['source_bindings'] == package['source_bindings']
            and manifest['phase'] == phase and manifest['all_registered_conditions_included']
            and manifest['observed_context_actions_simulator_labels_included'] is False, 'Package manifest differs')
    rows = manifest['records']
    require(len(rows) == len({x['record_id'] for x in rows}) == count and {x['record_id'] for x in rows} == set(native),
            'Package omits or duplicates a registered condition')
    for row in rows:
        original = native[row['record_id']]
        require(row['source_sha256'] == original['generated_sha256']
                and Path(row['source_path']).resolve() == Path(original['generated_artifact_path']).resolve(), 'Package source join differs')
        for key in ['match_id', 'clip_id', 'seed', 'intervention_type', 'dose', 'baseline_record_id', 'split']:
            require(row[key] == original[key], 'Package condition metadata differs')
    require(review['status'] == 'passed_generated_publication_independent_rehash' and review['phase'] == phase
            and review['total_records'] == count and review[f'all_{count}_registered_records_accounted'] is True,
            'Independent package review incomplete')
    for key in ['all_archive_sha256_independently_recomputed', 'all_sizes_and_1_8GB_limits_checked',
                'all_part_sidecar_hashes_checked', 'archive_manifest_hash_checked', 'passed_package_and_waiter_gates_checked',
                'packager_attests_all_members_readback_and_exact_future_slices', 'packager_attests_no_source_context_actions_simulator_labels']:
        require(review[key] is True, 'Independent package guard failed: ' + key)
    bindings.check(review['package_report'], path)
    require(review['archive_manifest'] == package['archive_manifest'], 'Review archive manifest differs')
    waiter = bindings.read(review['waiter_report']['path'], review['waiter_report']['sha256'])
    require(waiter['status'] == 'passed_generated_publication_waiter' and waiter['phase'] == phase
            and waiter['measurement_audit_sha256'] == hashes['measurement_audit'], 'Package waiter incomplete')
    bindings.check(waiter['report'], path)
    require(review['part_count'] == len(review['parts']) == len(package['parts']) > 0
            and review['total_bytes'] == sum(x['bytes'] for x in package['parts'])
            and sum(x['record_count'] for x in package['parts']) == count, 'Package part coverage differs')
    specs, all_members = [], []
    for index, (part, reviewed) in enumerate(zip(package['parts'], review['parts']), 1):
        require(part['status'] == 'passed_generated_artifact_part' and part['completed'] and part['readback_verified']
                and part['phase'] == phase and part['part_index'] == index, 'Incomplete or unordered part')
        required_review_fields = {'filename', 'path', 'bytes', 'record_count', 'member_count',
                                  'sha256', 'manifest_sha256', 'report'}
        require(required_review_fields <= set(reviewed)
                and all(k in part and part[k] == v for k, v in reviewed.items()),
                'Independent part review differs or omits required fields')
        tar = Path(part['path'])
        require(tar.name == part['filename'] and tar.stat().st_size == part['bytes']
                and 0 < part['bytes'] <= 1_800_000_000, 'Local part size/name differs')
        if rehash_parts:
            bindings.bind(tar, part['sha256'])
        sidecar = bindings.read(part['report']['path'], part['report']['sha256'])
        require(set(sidecar) == (set(part) - {'report'}) | {'members'}
                and all(sidecar[k] == v for k, v in part.items() if k != 'report'), 'Part sidecar differs')
        members = sidecar['members']
        require(len(members) + 1 == part['member_count']
                and sum(x['kind'] == 'generated_future_npz' for x in members) == part['record_count'],
                'Part sidecar member counts differ')
        payload = {'schema_version': 1, 'phase': phase, 'part_index': index,
                   'source_bindings': package['source_bindings'],
                   'members': sorted(members, key=lambda x: x['member']),
                   'observed_context_actions_simulator_labels_included': False}
        encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        require(hashlib.sha256(encoded).hexdigest() == part['manifest_sha256'], 'Part manifest digest differs')
        all_members.extend(members)
        specs.append({'name': tar.name, 'bytes': part['bytes'], 'sha256': part['sha256']})
        sp = Path(part['report']['path'])
        specs.append({'name': sp.name, 'bytes': sp.stat().st_size, 'sha256': part['report']['sha256']})
    require(len({x['member'] for x in all_members}) == len(all_members)
            and sorted(all_members, key=lambda x: x['member']) == sorted(manifest['members'], key=lambda x: x['member']),
            'Cross-part member coverage differs')
    for p in [Path(package['archive_manifest']['path']), path]:
        specs.append({'name': p.name, 'bytes': p.stat().st_size, 'sha256': bindings.bind(p)})
    return path, review_path, specs, package['parts']


def audit(root, publication_paths, *, rehash_parts=False, api=github_api):
    root = Path(root).resolve()
    bindings = Bindings()
    bindings.bind(Path(__file__))  # Capture loaded-source identity before any source/API checks.
    study_path = root / 'results/rollout_steering_v2/study_status.json'
    study = bindings.read(study_path)
    reports = complete_study(study, root, bindings)  # Reject a running study before any API requests.
    registration = bindings.read(root / 'results/rollout_steering_v2/registration.json')
    protocol = bindings.read(root / 'configs/generated_evaluation_v1.json')
    for source in [registration, protocol]:
        for name, digest in source['code_sha256'].items():
            bindings.bind(root / name, digest)
    bindings.bind(root / 'scripts/run_registered_rollouts.py', study['script_sha256'])
    bindings.bind(root / 'results/rollout_steering_v2/resource_review.json', study['resource_review_sha256'])
    require(registration['paths'] == protocol['paths'] == PATHS
            and registration['seeds'] == [2026090701, 2026090702]
            and registration['n_conditions_per_pair'] == 21, 'Registered paths or paired seeds differ')
    phases, local_parts = {}, []
    for phase in PHASES:
        hashes, native, export_path = check_phase(phase, root, reports, registration, protocol, bindings)
        package_path, review_path, specs, parts = package_and_review(phase, root, hashes, native, bindings, rehash_parts)
        local_parts.extend(parts)
        pubpath = Path(publication_paths[phase])
        publication = bindings.read(pubpath)
        require(publication['status'] == 'passed' and publication['phase'] == phase
                and publication['packaged_records'] == PHASES[phase][0]
                and publication['package_report_sha256'] == bindings.bind(package_path)
                and publication['all_remote_sizes_and_sha256_equal_local'] is True
                and publication['actual_release_tag_commit_verified'] is True, 'Publication report incomplete')
        for path, digest in publication['code_bindings'].items():
            bindings.bind(path, digest)
        notes_path = root / f'docs/release_notes/generated-{phase}-v2-2026-09-07.md'
        bindings.bind(notes_path, publication['release_notes_sha256'])
        remote = remote_publication(publication, specs, notes_path.read_text(), api)
        phases[phase] = {'records': PHASES[phase][0], 'matches': PHASES[phase][1], 'stage_hashes': hashes,
                         'export': {'path': str(export_path), 'sha256': bindings.bind(export_path)},
                         'package': {'path': str(package_path), 'sha256': bindings.bind(package_path)},
                         'independent_package_review': {'path': str(review_path), 'sha256': bindings.bind(review_path)},
                         'publication': {'path': str(pubpath), 'sha256': bindings.bind(pubpath)}, 'remote': remote}
    presentation_path = root / 'results/progress/final_presentation_review.json'
    presentation = bindings.read(presentation_path)
    require(presentation['status'] == 'passed_publication_presentation_review' and not presentation['material_findings'],
            'Initial-progress presentation review has findings')
    readme_spec = [x for x in presentation['artifacts'] if x['path'] == 'README.md']
    require(len(readme_spec) == 1, 'Missing reviewed README binding')
    bindings.bind(root / 'README.md', readme_spec[0]['sha256'])
    readme = (root / 'README.md').read_text()
    require(len(readme.split()) == 113 and len(re.findall(r'!\[', readme)) == 2, 'Initial-progress README changed')
    progress = presentation['progress_artifacts']
    figures = bindings.read(root / 'results/progress/figure_manifest.json', progress['figure_manifest_sha256'])
    figure_review = bindings.read(root / 'results/progress/figure_review.json', progress['figure_review_sha256'])
    require(figures['status'] == 'passed_audited_initial_progress_export' and figure_review['status'] == 'passed'
            and figure_review['figure_manifest_sha256'] == progress['figure_manifest_sha256'], 'Progress figure review differs')
    output_specs = {x['path']: x for x in figures['outputs']}
    image_paths = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', readme)
    require(len(image_paths) == 2 and len(set(image_paths)) == 2, 'Two distinct README figure links required')
    for image_path in image_paths:
        require(image_path in output_specs, 'README figure absent from audited manifest')
        spec = output_specs[image_path]
        bindings.bind(root / image_path, spec['sha256'])
        require((root / image_path).stat().st_size == spec['bytes'], 'README image size differs')
    bindings.bind(Path(__file__))
    bindings.unchanged()
    for part in local_parts:
        require(Path(part['path']).stat().st_size == part['bytes'], 'Local archive size changed during audit')
    return {'status': 'passed_execution_and_publication_audit', 'checked_utc': datetime.now(timezone.utc).isoformat(),
            'study_status_sha256': bindings.hashes[study_path.resolve()], 'all_eight_ordered_stages_passed': True,
            'phases': phases, 'total_conditions': 1344, 'all_phase_export_hashes_current': True,
            'all_phase_package_and_independent_review_bindings_current': True,
            'all_remote_asset_sizes_digests_and_actual_tag_commits_current': True,
            'local_archive_bytes_rehashed_now': rehash_parts,
            'local_archive_bytes_evidence': 'Current sizes checked; exact bytes previously verified by packager, independent rehash reviewer and uploader. Remote GitHub SHA256 checked now.' if not rehash_parts else 'All local TAR bytes freshly rehashed in this audit as well as existing prior verification.',
            'array_decoding_or_metric_refitting_performed': False, 'new_model_execution': False, 'gpu_used': False,
            'remote_mutations_performed': False, 'physical_control_established': False,
            'generated_video_accuracy_established': False,
            'readme_scope': 'Initial decoding progress; 113 source-whitespace words and exactly two figures, unchanged from passed presentation review.',
            'visual_inspection_scope': 'Existing phase exports are hashed, but final visual inspection is a separate review; this script does not view or approve plots.',
            'claim_limit': 'Successful execution, audit joins and artifact publication do not imply successful physical steering.',
            'artifact_bindings': [{'path': str(p), 'sha256': h} for p, h in sorted(bindings.hashes.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--selection-publication', type=Path)
    parser.add_argument('--confirmation-publication', type=Path)
    parser.add_argument('--rehash-parts', action='store_true', help='Optional costly repeat SHA256 of all local TAR bytes')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), 'Completed audit report is immutable')
    paths = {p: getattr(args, p + '_publication') or args.root / f'results/generated_{p}_v2_publication.json' for p in PHASES}
    result = audit(args.root, paths, rehash_parts=args.rehash_parts)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'report': str(args.report), 'sha256': sha(args.report)}))


if __name__ == '__main__':
    main()
