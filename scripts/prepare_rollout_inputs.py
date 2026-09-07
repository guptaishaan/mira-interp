#!/usr/bin/env python3
"""Extend metadata-chosen clips to16 observed+8 future frames; no model execution."""
import os
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mira_interp.clips import audit_physics_tracks
from mira_interp.development_capture import decode_float_video
from mira_interp.model_loading import sha256
from mira_interp.data import TARGET_NAMES

CONFIG = ROOT / 'configs/rollout_inputs_v1.json'
REPORT = ROOT / 'results/rollout_inputs_v1.json'
BASE = Path('/data2/ishaangp/mira-interp/rollout_inputs_v1')


class QualityExclusion(ValueError):
    """Expected source-physics quality rejection; integrity/code errors are fatal."""


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def register():
    if CONFIG.exists():
        raise ValueError('Preserve existing rollout input registration')
    candidates, exclusions, bindings = [], [], {}
    sources = [('pilot', 'data/qualified_pilot_clip_manifest.json', 'test'),
               ('selection', 'data/qualified_clip_manifest.json', 'test'),
               ('confirmation', 'data/qualified_fresh_confirmation_clip_manifest.json', 'dev')]
    for role, name, source_split in sources:
        path = ROOT / name; bindings[name] = sha256(path)
        rows = [r for r in json.loads(path.read_text())['records'] if r['role'] == role]
        for match in sorted({r['match_id'] for r in rows}):
            available = sorted([r for r in rows if r['match_id'] == match and
                                r['plan']['local_start'] + 24 <= min(v['decoded_frame_count'] for v in r['video_audit'])], key=lambda r: r['clip_id'])
            if not available:
                exclusions.append(dict(match_id=match, role=role, reason='No recorded candidate has24 frames left in its source chunk'))
                continue
            candidates.append(dict(role=role, source_split=source_split, source=available[0]))
    config = dict(status='registered_before_extended_input_preparation', registered_utc=datetime.now(timezone.utc).isoformat(),
                  rule='For each match use lexicographically first existing candidate with24 within-chunk frames; exclude failed extended source-quality audits without replacement',
                  observed_frames=16, generated_frames=8, fps=20, source_manifest_sha256=bindings,
                  candidates=candidates, metadata_exclusions=exclusions,
                  code_sha256={p: sha256(ROOT/p) for p in ['scripts/prepare_rollout_inputs.py','src/mira_interp/clips.py','src/mira_interp/development_capture.py','src/mira_interp/data.py']},
                  scope='Input preparation only; no model, evaluator or intervention outcomes consulted',
                  confirmation_note='Fresh matches have observational readout outcomes exposed, but have not trained or selected these edits. Do not call them wholly uninspected data.')
    save(CONFIG, config)
    print(json.dumps(dict(status=config['status'], candidates=len(candidates), exclusions=len(exclusions))))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--register', action='store_true'); args = p.parse_args()
    if args.register:
        register(); return
    if REPORT.exists():
        raise ValueError('Do not overwrite completed or failed preparation')
    config = json.loads(CONFIG.read_text()); config_hash = sha256(CONFIG)
    for group in ('source_manifest_sha256', 'code_sha256'):
        if any(sha256(ROOT/name) != digest for name, digest in config[group].items()):
            raise ValueError('Registered source or code changed')
    torch.set_num_threads(1)
    report = dict(status='running', registration_sha256=config_hash, records=[], exclusions=[], errors=[], expected_candidates=len(config['candidates']))
    save(REPORT, report); started = time.monotonic()

    def work(candidate):
        row = candidate['source']; old_path = Path(row['artifact_path'])
        if sha256(old_path) != row['sha256']:
            raise ValueError('Original prepared source changed')
        with np.load(old_path, allow_pickle=False) as old:
            old_arrays = {key: old[key] for key in old.files}
        shard = Path('/data2/ishaangp/mira-interp/rocket-science') / candidate['source_split'] / row['source_shard']
        key = f"{row['match_id']}_c{row['chunk_index']:05d}"
        start = row['plan']['local_start']; local = list(range(start, start+24))
        frames, pts, physics, actions, hashes = [], [], [], [], {}
        with tarfile.open(shard) as tar:
            meta = json.loads(tar.extractfile(f'{key}.meta.json').read())
            if meta['player_ids'] != row['player_ids'] or meta['teams'] != row['teams'] or meta['match_id'] != row['match_id']:
                raise ValueError('Source metadata changed')
            for view in range(4):
                blobs = {suffix: tar.extractfile(f'{key}.p{view}.{suffix}').read() for suffix in ('mp4','physics.jsonl','jsonl')}
                for suffix, blob in blobs.items():
                    name = f'p{view}.{suffix}'; hashes[name] = hashlib.sha256(blob).hexdigest()
                    if hashes[name] != row['source_component_sha256'][name]:
                        raise ValueError('Source component checksum mismatch')
                a, stamps = decode_float_video(blobs['mp4'], local, row['video_audit'][view]['decoded_frame_count'])
                frames.append(a); pts.append(stamps)
                lines = blobs['physics.jsonl'].splitlines(); acts = blobs['jsonl'].decode().splitlines()
                physics.append([json.loads(lines[i]) for i in local]); actions.append([acts[i] for i in local])
        try:
            y, action, quality = audit_physics_tracks(physics, actions, row['player_ids'], row['teams'])
        except ValueError as exc:
            raise QualityExclusion(str(exc)) from exc
        frames, pts = np.stack(frames), np.stack(pts)
        if not np.array_equal(y[:, :16], old_arrays['targets']) or not np.array_equal(action[:, :16], old_arrays['actions']):
            raise ValueError('Extended context labels/actions differ from original')
        if not np.array_equal(np.rint(frames[:, :16]).clip(0,255).astype(np.uint8), old_arrays['frames']) or not np.array_equal(pts[:, :16], old_arrays['video_pts']):
            raise ValueError('Extended context pixels/PTS differ from original')
        indices = np.arange(row['plan']['source_start_frame'], row['plan']['source_start_frame'] + 24)
        origins = np.array([v['calibrated_origin_seconds'] for v in row['timeline_calibration']['views']])
        timestamps = origins[:,None] + indices[None,:]/20
        if not np.array_equal(timestamps[:,:16], old_arrays['timestamps']):
            raise ValueError('Per-view clock mapping changed')
        out = BASE / candidate['role'] / (row['clip_id'] + '.npz'); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, frames=frames, actions=action, targets=y, source_frame_indices=indices,
                            timestamps=timestamps, video_pts=pts, player_ids=old_arrays['player_ids'], target_names=np.array(TARGET_NAMES))
        with np.load(out, allow_pickle=False) as back:
            if not np.array_equal(back['frames'], frames) or not np.array_equal(back['actions'], action) or not np.array_equal(back['targets'], y):
                raise ValueError('Extended artifact readback differs')
        return dict(match_id=row['match_id'], clip_id=row['clip_id'], role=candidate['role'], path=str(out), sha256=sha256(out),
                    parent_source_sha256=row['sha256'], source_component_sha256=hashes, source_shard_sha256=row['source_shard_sha256'],
                    physical_audit=quality, context16_source_join_exact=True, frames=24, observed_context_frames=16,
                    future_ground_truth_use='calibration/reference only; all eight future video frames will be hidden from generator',
                    own_view_time_alignment=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work, c) for c in config['candidates']]
        for candidate, future in zip(config['candidates'], futures):
            try:
                report['records'].append(future.result())
            except QualityExclusion as exc:
                report['exclusions'].append(dict(match_id=candidate['source']['match_id'], role=candidate['role'], reason=f'{type(exc).__name__}: {exc}'))
            except Exception as exc:
                report['errors'].append(dict(match_id=candidate['source']['match_id'], role=candidate['role'], reason=f'{type(exc).__name__}: {exc}'))
            save(REPORT, report)
            print(json.dumps(dict(passed=len(report['records']), excluded=len(report['exclusions']), expected=report['expected_candidates'])), flush=True)
    counts = {role: sum(r['role'] == role for r in report['records']) for role in ('pilot','selection','confirmation')}
    # Quality exclusions are prospective, but missing pilot or research roles block execution.
    complete = not report['errors'] and counts['pilot'] == 1 and counts['selection'] >= 2 and counts['confirmation'] >= 2
    if sha256(CONFIG) != config_hash:
        raise ValueError('Registration changed during preparation')
    report.update(status='passed_quality_qualified_rollout_inputs' if complete else 'failed', match_counts=counts,
                  elapsed_seconds=time.monotonic()-started, metadata_exclusions=config['metadata_exclusions'],
                  no_replacements_after_quality_exclusion=True, model_executed=False)
    save(REPORT, report)
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
