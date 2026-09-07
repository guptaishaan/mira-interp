#!/usr/bin/env python3
"""Re-decode exact old development windows without uint8 resize quantization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira_interp.development_capture import decode_float_video
from mira_interp.model_loading import sha256

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    manifest = ROOT/'data'/('qualified_pilot_clip_manifest.json' if args.pilot else 'qualified_clip_manifest.json')
    rows = json.loads(manifest.read_text())['records']
    rows = [r for r in rows if r['role'] in (['pilot'] if args.pilot else ['discovery', 'selection'])]
    if args.pilot:
        rows = rows[:1]
    out = Path('/data2/ishaangp/mira-interp/prepared_float_v3')
    report_path = ROOT/'results'/('development_prepare_pilot.json' if args.pilot else 'development_prepare.json')
    report = dict(status='running', scope='development only; no old confirmation', input_manifest_sha256=sha256(manifest),
                  code_sha256=sha256(Path(__file__)), helper_sha256=sha256(ROOT/'src/mira_interp/development_capture.py'),
                  expected_clips=len(rows), records=[], errors=[])
    save(report_path, report)
    start = time.monotonic()
    def work(row):
        if row['role'] == 'confirmation':
            raise ValueError('Confirmation forbidden')
        path = out/row['role']/(row['clip_id']+'.npz')
        sidecar = path.with_suffix('.json')
        if sidecar.exists():
            cached = json.loads(sidecar.read_text())
            if cached['parent_sha256'] != row['sha256'] or sha256(path) != cached['sha256']:
                raise ValueError('Cache integrity mismatch')
            return cached
        old_path = Path(row['artifact_path'])
        if sha256(old_path) != row['sha256']:
            raise ValueError('Prepared input changed')
        with np.load(old_path, allow_pickle=False) as old:
            kept = {k: old[k] for k in old.files if k != 'frames'}
            old_frames = old['frames']
        shard = Path('/data2/ishaangp/mira-interp/rocket-science/test')/row['source_shard']
        key = f"{row['match_id']}_c{row['chunk_index']:05d}"
        selected = list(range(row['plan']['local_start'], row['plan']['local_start']+16))
        frames, pts = [], []
        with tarfile.open(shard) as tar:
            for v in range(4):
                blob = tar.extractfile(f'{key}.p{v}.mp4').read()
                if hashlib.sha256(blob).hexdigest() != row['source_component_sha256'][f'p{v}.mp4']:
                    raise ValueError('Video source hash mismatch')
                arr, stamps = decode_float_video(blob, selected, row['video_audit'][v]['decoded_frame_count'])
                frames.append(arr)
                pts.append(stamps)
        frames = np.stack(frames)
        if not np.array_equal(np.asarray(pts), kept['video_pts']):
            raise ValueError('Source PTS changed')
        rounded = np.rint(frames).clip(0, 255).astype(np.uint8)
        if not np.array_equal(rounded, old_frames):
            raise ValueError('Fractional resize does not reproduce original rounded pixels')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frames=frames, **kept)
        record = dict(clip_id=row['clip_id'], match_id=row['match_id'], split=row['role'], artifact_path=str(path),
                      sha256=sha256(path), parent_sha256=row['sha256'], source_pts_exact=True,
                      round_to_previous_pixels_exact=True, max_resize_quantization=float(np.max(np.abs(frames-old_frames))),
                      source_component_sha256=row['source_component_sha256'], row_metadata_unchanged=True)
        save(sidecar, record)
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, future in zip(rows, [pool.submit(work, r) for r in rows]):
            try:
                record = future.result()
                report['records'].append(record)
                print(json.dumps(dict(completed=len(report['records']), expected=len(rows), clip_id=row['clip_id'])), flush=True)
            except Exception as e:
                report['errors'].append(dict(clip_id=row['clip_id'], type=type(e).__name__, message=str(e)))
            save(report_path, report)
    report.update(status='passed' if not report['errors'] and len(report['records']) == len(rows) else 'failed', elapsed_seconds=time.monotonic()-start)
    save(report_path, report)
    return 0 if report['status']=='passed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
