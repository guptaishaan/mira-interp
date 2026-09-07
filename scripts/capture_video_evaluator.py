#!/usr/bin/env python3
"""Frozen VideoMAE capture; reserved pilot precedes all336 development clips."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'external/mira/src'))
import numpy as np
import torch
from mira_interp.model_loading import sha256
from mira_interp.video_evaluator import load_encoder, encode_frames, endpoint_labels, load_roundtrip_codec, codec_roundtrip


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--output-dir', type=Path, default=Path('/data2/ishaangp/mira-interp/video_evaluator_v1'))
    p.add_argument('--assets', type=Path, default=Path('/data2/ishaangp/mira-interp/models/videomae-base'))
    p.add_argument('--mira-assets', type=Path, default=Path('/data2/ishaangp/mira-interp/assets'))
    args = p.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise ValueError('Capture requires the explicitly reserved visible GPU')
    device = 'cuda:0'
    source_report = ROOT/'results'/('development_prepare_pilot.json' if args.pilot else 'development_prepare.json')
    source = json.loads(source_report.read_text())
    expected = 1 if args.pilot else 336
    if source['status'] != 'passed' or len(source['records']) != expected or source['errors']:
        raise ValueError('Float-video preparation has not passed complete coverage')
    rows = source['records']; ids = [row['clip_id'] for row in rows]
    if len(set(ids)) != expected or any(row['split'] not in (['pilot'] if args.pilot else ['discovery', 'selection']) for row in rows):
        raise ValueError('Duplicate clips or forbidden partitions')
    counts = Counter((row['split'], row['match_id']) for row in rows)
    if not args.pilot and (set(counts.values()) != {8} or Counter(role for role, match in counts) != {'discovery': 31, 'selection': 11}):
        raise ValueError('Expected original31/11 development matches with eight clips each')
    codes = {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
             'src/mira_interp/video_evaluator.py': sha256(ROOT/'src/mira_interp/video_evaluator.py')}
    pilot_path = ROOT/'results/videomae_pilot.json'
    if not args.pilot:
        pilot = json.loads(pilot_path.read_text())
        if pilot['status'] != 'passed' or pilot['code_sha256'] != codes or not pilot['pilot_controls']['passed']:
            raise ValueError('Same-code reserved evaluator pilot must pass first')
    report_path = ROOT/'results'/('videomae_pilot.json' if args.pilot else 'videomae_capture.json')
    report = {'status': 'running', 'scope': 'reserved engineering pilot' if args.pilot else '31/11 development only',
              'input_report_sha256': sha256(source_report), 'code_sha256': codes, 'expected_clips': expected,
              'expected_view_rows': expected*4, 'records': [], 'gpu': torch.cuda.get_device_name(0),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'confirmation_used': False,
              'codec_calibration_partition': 'pilot' if args.pilot else 'selection only',
              'independent_generated_video_calibration_passed': False}
    save(report_path, report)
    started = time.monotonic()
    encoder, preprocess, report['encoder_loading'] = load_encoder(args.assets)
    codec, report['codec_loading'] = load_roundtrip_codec(args.mira_assets)
    encoder.to(device); codec.to(device)
    if any(parameter.dtype != torch.float32 for model in (encoder, codec) for parameter in model.parameters()):
        raise ValueError('Keep encoder and codec checkpoint parameters in FP32')
    save(report_path, report)
    for row in rows:
        path = args.output_dir/row['split']/(row['clip_id']+'.npz'); sidecar = path.with_suffix('.json')
        if sidecar.exists() and not args.pilot:
            cached = json.loads(sidecar.read_text())
            if cached['source_sha256'] != row['sha256'] or cached['code_sha256'] != codes or sha256(path) != cached['sha256']:
                raise ValueError('Cached evaluator capture integrity mismatch')
            report['records'].append(cached); save(report_path, report); continue
        input_path = Path(row['artifact_path'])
        if sha256(input_path) != row['sha256']:
            raise ValueError('Prepared float clip hash changed')
        with np.load(input_path, allow_pickle=False) as source_clip:
            frames = source_clip['frames']
            if frames.dtype != np.float32 or frames.shape != (4, 16, 3, 288, 512):
                raise ValueError('Unexpected source float video')
            features = encode_frames(encoder, frames, preprocess, device=device)
            labels = endpoint_labels(source_clip)  # labels joined after pixel-only encoding
        arrays = {'X': features, **labels}; calibration = None
        if row['split'] in ('pilot', 'selection'):
            reconstructed, calibration = codec_roundtrip(codec, frames, device=device)
            arrays['X_codec_reconstruction'] = encode_frames(encoder, reconstructed, preprocess, device=device)
        if args.pilot:
            repeated = encode_frames(encoder, frames, preprocess, device=device)
            repeat_error = float(np.max(np.abs(features-repeated)))
            if not np.allclose(features, repeated, rtol=1e-5, atol=1e-5):
                raise ValueError('Repeated frozen VideoMAE encoding is unstable')
            if features.shape != (4, 9216) or not np.isfinite(features).all() or labels['y'].shape != (4, 12):
                raise ValueError('Pilot feature/target schema failed')
            report['pilot_controls'] = {'passed': True, 'repeat_max_abs': repeat_error,
                'features_shape': list(features.shape), 'target_shape': list(labels['y'].shape),
                'endpoint_source_frames': labels['source_frame_index'].tolist(),
                'per_view_endpoint_timestamps': labels['timestamps'].tolist(),
                'original_state_target_time': 'last of16 source frames; last VideoMAE tubelet contains frames14 and15',
                'no_action_or_state_input_to_encoder': True, 'all_parameters_fp32': True,
                'encoder_autocast': False, 'codec_autocast': 'bfloat16', 'codec_reconstruction': calibration}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        record = {'clip_id': row['clip_id'], 'match_id': row['match_id'], 'split': row['split'],
                  'artifact_path': str(path), 'sha256': sha256(path), 'source_sha256': row['sha256'],
                  'source_artifact_path': str(input_path), 'code_sha256': codes,
                  'view_rows': 4, 'codec_calibration': calibration}
        save(sidecar, record); report['records'].append(record)
        report['elapsed_seconds'] = time.monotonic()-started
        save(report_path, report)
        print(json.dumps({'completed': len(report['records']), 'expected': expected, 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    if sha256(source_report) != report['input_report_sha256']:
        raise ValueError('Prepared data report changed during evaluator capture')
    report.update(status='passed', elapsed_seconds=time.monotonic()-started)
    save(report_path, report)
    print(json.dumps({'status': 'passed', 'report': str(report_path)}))


if __name__ == '__main__':
    main()
