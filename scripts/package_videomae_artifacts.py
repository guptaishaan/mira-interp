#!/usr/bin/env python3
"""Build a deterministic, label-free VideoMAE derived-artifact release; CPU only."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path('/data2/ishaangp/mira-interp/video_evaluator_v1')
REPORTS = ('videomae_access', 'videomae_model_load', 'videomae_pilot', 'videomae_capture',
           'videomae_capture_audit', 'videomae_analysis', 'videomae_mlp', 'videomae_finetune_pilot',
           'videomae_prefix', 'videomae_finetune_v1', 'videomae_finetune_v1_audit',
           'videomae_matched_input_calibration', 'videomae_selftest', 'videomae_training_selftest')
FEATURE_KEYS = {'X', 'X_codec_reconstruction', 'view_index', 'target_names'}
PREDICTION_KEYS = {'real_prediction', 'codec_prediction', 'mean_prediction', 'shuffled_prediction',
                   'real_MLP_prediction', 'real_ridge_prediction', 'codec_MLP_prediction',
                   'codec_ridge_prediction', 'match_ids', 'clip_ids', 'view_index', 'target_names'}
RIDGE_KEYS = {'alpha', 'x_mean', 'x_scale', 'y_mean', 'y_scale', 'coefficient', 'target_names'}
REMOVABLE = {'y', 'absolute_ball6', 'timestamps', 'source_frame_index'}
LIMIT = 2_000_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def deterministic_npz(arrays):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as z:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[key], allow_pickle=False)
            info = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            z.writestr(info, buffer.getvalue())
    return output.getvalue()


def references(value, report, pointer=''):
    """Inventory all absolute filesystem references, including excluded source inputs."""
    if isinstance(value, dict):
        for key, child in value.items():
            here = pointer + '/' + key
            if isinstance(child, str) and child.startswith(('/data2/', '/ccn2/')):
                expected = value.get('sha256') if key in {'path', 'artifact_path'} else None
                if key == 'source_artifact_path':
                    expected = value.get('source_sha256')
                yield {'report': report, 'pointer': here, 'path': child, 'declared_sha256': expected}
            else:
                yield from references(child, report, here)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from references(child, report, pointer + '/' + str(index))


def archive(path, entries):
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as tar:
        for name, data in sorted(entries.items()):
            require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts, 'Unsafe member')
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            tar.addfile(info, io.BytesIO(data))
    require(path.stat().st_size < LIMIT, 'Asset exceeds conservative 2GB limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/data2/ishaangp/mira-interp/publication/videomae_v1'))
    parser.add_argument('--report', type=Path, default=ROOT / 'results/videomae_artifact_release.json')
    args = parser.parse_args()
    require(not args.report.exists(), 'Release report is immutable; use a new destination')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / 'videomae-derived-v1.tar'
    require(not target.exists(), 'Archive destination exists')
    require(shutil.disk_usage(args.output_dir).free > 25 * 1024**3 + 2 * LIMIT, 'Insufficient disk reserve')
    entries, members, inventory, reports = {}, [], [], {}
    for name in REPORTS:
        path = ROOT / 'results' / (name + '.json')
        report = json.loads(path.read_text())
        require(str(report.get('status', '')).startswith('passed'), f'Incomplete source report: {name}')
        reports[name] = report
        inventory.extend(references(report, str(path.relative_to(ROOT))))
    require(reports['videomae_capture']['expected_clips'] == 336
            and len(reports['videomae_capture']['records']) == 336, 'Development feature coverage differs')
    require(reports['videomae_finetune_v1']['selected_step'] == 500, 'Frozen evaluator checkpoint differs')

    def add(name, data, source=None, **metadata):
        require(name not in entries, 'Duplicate release member')
        entries[name] = data
        members.append({'member': name, 'bytes': len(data), 'sha256': digest(data),
                        **({'source_path': str(source), 'source_sha256': sha(source)} if source else {}), **metadata})

    # The report tree contains aggregate metrics and provenance, not per-row simulator labels.
    paths = {ROOT / 'results' / (name + '.json') for name in REPORTS}
    for folder in ('videomae_exports', 'videomae_repair_exports', 'videomae_attempts'):
        paths.update(p for p in (ROOT / 'results' / folder).iterdir() if p.suffix in {'.json', '.csv', '.py'})
    paths.update((ROOT / 'figures').glob('videomae_*.png'))
    paths.update((ROOT / 'figures').glob('videomae_*.pdf'))
    paths.update((ROOT / 'scripts').glob('*video_evaluator*.py'))
    paths.update((ROOT / 'src/mira_interp').glob('video_evaluator*.py'))
    paths.update((ROOT / 'tests').glob('test_video_evaluator*.py'))
    paths.update(ROOT / p for p in ('scripts/download_videomae.py', 'scripts/package_videomae_artifacts.py',
        'src/mira_interp/probes.py', 'src/mira_interp/development.py', 'src/mira_interp/model_loading.py',
        'configs/videomae_partial_finetune_v1.json', 'docs/review/video_evaluator.md', 'pyproject.toml'))
    for path in sorted(paths):
        require(path.is_file(), f'Missing release source: {path}')
        add('repository/' + path.relative_to(ROOT).as_posix(), path.read_bytes(), path, kind='documentation_code_or_aggregate')

    copied = {}
    for ref in inventory:
        path = Path(ref['path'])
        # Directory and raw-input references are retained as metadata; never traverse/copy them.
        if not path.is_relative_to(PRIVATE):
            ref.update(decision='excluded', reason='source_dataset_or_upstream_asset_or_external_metadata', hash_verified=False)
            continue
        if 'prefix' in path.relative_to(PRIVATE).parts:
            ref.update(decision='excluded', reason='large_reproducible_prefix_cache_or_source_label_metadata', hash_verified=False)
            continue
        require(path.is_file(), f'Referenced derived file is missing: {path}')
        if str(path) in copied:
            ref.update(copied[str(path)])
            if ref['declared_sha256']:
                require(ref['declared_sha256'] == ref['verified_source_sha256'], 'Conflicting source hash')
            continue
        original_sha = sha(path)
        require(ref['declared_sha256'] is None or original_sha == ref['declared_sha256'], f'Source hash mismatch: {path}')
        name = 'derived/' + path.relative_to(PRIVATE).as_posix()
        metadata = {}
        if path.suffix == '.pt':
            import torch
            weights = torch.load(path, map_location='cpu', weights_only=True)
            require(weights and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in weights.values()),
                    'Checkpoint must contain finite tensor weights only')
            require(all(k.startswith(('blocks.', 'norm.', 'head.', 'layers.')) or k in {'x_mean','x_scale','y_mean','y_scale'}
                        for k in weights), 'Unexpected checkpoint payload')
            payload, kind = path.read_bytes(), 'unchanged_derived_checkpoint'
            metadata['tensor_shapes'] = {k: list(v.shape) for k, v in weights.items()}
        else:
            require(path.suffix == '.npz', 'Unexpected derived format')
            with np.load(path, allow_pickle=False) as z:
                keys = set(z.files)
                if path.name in {'main_ridge.npz', 'shuffled_ridge.npz'}:
                    require(keys == RIDGE_KEYS, 'Ridge schema changed')
                    allowed, kind = RIDGE_KEYS, 'unchanged_ridge_checkpoint'
                elif 'X' in keys:
                    allowed, kind = FEATURE_KEYS, 'sanitized_frozen_features'
                    require(z['X'].shape == (4, 9216), 'Frozen feature shape differs')
                else:
                    allowed, kind = PREDICTION_KEYS, 'sanitized_predictions'
                    require('y' in keys and z['y'].shape == (352, 12), 'Prediction row schema differs')
                require(not (keys - allowed - REMOVABLE), 'Unrecognized NPZ array; fail closed')
                arrays = {key: z[key] for key in sorted(keys & allowed)}
                require(arrays and all(not a.dtype.hasobject for a in arrays.values()), 'Invalid NPZ types')
                require(all(np.isfinite(a).all() for a in arrays.values() if a.dtype.kind in 'fci'), 'Nonfinite derived array')
                payload = path.read_bytes() if kind == 'unchanged_ridge_checkpoint' else deterministic_npz(arrays)
                metadata.update(removed_arrays=sorted(keys - allowed), arrays={k: {'shape': list(a.shape), 'dtype': str(a.dtype)} for k,a in arrays.items()})
                with np.load(io.BytesIO(payload), allow_pickle=False) as check:
                    require(set(check.files) == set(arrays) and all(np.array_equal(check[k], a) for k,a in arrays.items()),
                            'Sanitized NPZ readback differs')
        add(name, payload, path, kind=kind, **metadata)
        decision = {'decision': 'included', 'member': name, 'transformation': kind,
                    'verified_source_sha256': original_sha, 'hash_verified': True}
        ref.update(decision)
        copied[str(path)] = decision

    feature_members = [m for m in members if m.get('kind') == 'sanitized_frozen_features']
    require(len(feature_members) == 337, 'Expected 336 development plus one reserved pilot feature artifact')
    require(sum(m.get('kind') == 'unchanged_derived_checkpoint' for m in members) == 4, 'Missing saved checkpoint')
    source_model = Path('/data2/ishaangp/mira-interp/models/videomae-base/README.md')
    add('UPSTREAM_MODEL_CARD.md', source_model.read_bytes(), source_model, kind='upstream_attribution')
    note = '''# VideoMAE derived evaluator artifacts\n\nThis archive contains four saved derived neural checkpoints, two ridge checkpoints,\n337 frozen-feature artifacts (336 development clips plus one reserved pilot),\nlabel-free predictions, and the original aggregate reports, figures and code.\nSource video, actions, per-row simulator positions/velocities, and prefix caches\nare excluded. Checkpoint y_mean/y_scale are derived training statistics.\n\nThe feature/prediction files are sanitized copies, not the original hashed files.\nMANIFEST.json maps both hashes and lists omitted arrays. Original report hashes\ncontinue to identify private originals. Reattach labels only from authorized\nRocket Science data using the original clip/match order and source hashes; this\nlabel-free release alone cannot independently recompute supervised metrics.\n\nVideoMAE by Tong et al.: https://arxiv.org/abs/2203.12602\nOfficial code: https://github.com/MCG-NJU/VideoMAE\nUpstream model: https://huggingface.co/MCG-NJU/videomae-base\nPinned revision: dc740ceda42fce44faed2ea03c6d447db72f6af9\nUpstream model-card license: CC-BY-NC-4.0. The adapted encoder blocks derive\nfrom those weights; preserve attribution and the upstream noncommercial terms.\nThe unchanged upstream full model is not bundled. Retrieve the pinned original\nseparately and use the strict loader plus partial fine-tune tail checkpoint.\n\nThe selected evaluator was trained on31 discovery matches and selected on11\ndevelopment matches. It has substantial position error and weak velocity\nprediction. Codec reconstruction calibration does not certify accuracy on\ngenerated or intervened videos. This archive establishes reproducible derived\nartifacts; it does not establish causal physical control.\n\nRebuild from the same private source files and repo snapshot:\nNUMPY_MADVISE_HUGEPAGE=0 python scripts/package_videomae_artifacts.py \\\n  --output-dir /data2/your-new-output --report /data2/your-new-report.json\nSorted tar entries have zero timestamps and stable owner/mode metadata; NPZ\narrays use a fixed ZIP timestamp. The script verifies every included source\nhash, reopens each sanitized NPZ, builds the archive twice, and verifies every\narchive member by SHA256 without extracting paths.\n'''
    add('README.md', note.encode(), kind='release_documentation')
    manifest = {'schema_version': 1, 'scope': 'completed_development_videomae_derived_artifacts_only',
        'source_reports': {name: sha(ROOT/'results'/(name+'.json')) for name in REPORTS},
        'members': members, 'all_absolute_report_references': inventory,
        'raw_video_actions_physics_included': False, 'prefix_caches_included': False,
        'sanitized_feature_artifacts': len(feature_members), 'archive_metadata': {'mtime': 0, 'uid': 0, 'gid': 0, 'mode': '0644'},
        'supervised_metric_reproduction_requires_authorized_labels': True}
    entries['MANIFEST.json'] = json_bytes(manifest)
    archive(target, entries)
    second = target.with_suffix('.readback.tar')
    archive(second, entries)
    require(sha(target) == sha(second), 'Independent deterministic rebuild differs')
    second.unlink()
    with tarfile.open(target, 'r:') as tar:
        rows = tar.getmembers()
        require(len(rows) == len(entries) and len({m.name for m in rows}) == len(rows), 'Archive duplicate/missing member')
        for member in rows:
            require(member.isfile() and member.uid == member.gid == member.mtime == 0 and member.mode == 0o644, 'Unsafe or unstable member metadata')
            data = tar.extractfile(member).read()
            require(member.name in entries and digest(data) == digest(entries[member.name]), 'Archive readback hash mismatch')
    result = {'status': 'passed_derived_artifact_release_audit', 'script_sha256': sha(Path(__file__)),
        'archive': {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target), 'member_count': len(entries)},
        'manifest_sha256': digest(entries['MANIFEST.json']), 'source_report_sha256': manifest['source_reports'],
        'all_members_readback_verified': True, 'deterministic_second_build_identical': True,
        'all_included_derived_source_hashes_verified': True, 'sanitized_npz_arrays_exactly_preserved': True,
        'source_dataset_arrays_included': False, 'prefix_caches_included': False,
        'checkpoints': 4, 'ridge_checkpoints': 2, 'feature_artifacts': len(feature_members),
        'predictions': sum(m.get('kind') == 'sanitized_predictions' for m in members),
        'original_files_mutated': False, 'uploaded_to_github': False}
    args.report.write_bytes(json_bytes(result))
    (args.output_dir/'MANIFEST.json').write_bytes(entries['MANIFEST.json'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
