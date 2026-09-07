#!/usr/bin/env python3
"""Independently audit complete evaluator captures, then fit development ridge."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from mira_interp.development import ALPHAS, ROLE_TARGET_NAMES, check_development_roles, metrics
from mira_interp.model_loading import sha256
from mira_interp.probes import match_weights, ridge_path, shuffle_match_labels
from mira_interp.video_evaluator import VIDEOMAE_WEIGHT_SHA


def audit_captures(capture_path):
    capture = json.loads(capture_path.read_text())
    if capture['status'] != 'passed' or capture['expected_clips'] != 336 or len(capture['records']) != 336:
        raise ValueError('Require all336 passed development captures')
    if capture['encoder_loading']['weight_sha256'] != VIDEOMAE_WEIGHT_SHA or capture['confirmation_used']:
        raise ValueError('Wrong encoder or forbidden confirmation data')
    for name, digest in capture['code_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Evaluator capture code changed')
    source_path = ROOT/'results/development_prepare.json'
    if sha256(source_path) != capture['input_report_sha256']:
        raise ValueError('Prepared data report changed')
    prepared = json.loads(source_path.read_text())
    source_rows = {row['clip_id']: row for row in prepared['records']}
    rows = sorted(capture['records'], key=lambda row: (row['match_id'], row['clip_id']))
    if len({row['clip_id'] for row in rows}) != 336 or {row['clip_id'] for row in rows} != set(source_rows):
        raise ValueError('Capture and prepared clip membership differ')
    counts = Counter((row['split'], row['match_id']) for row in rows)
    if set(counts.values()) != {8} or Counter(role for role, match in counts) != {'discovery': 31, 'selection': 11}:
        raise ValueError('Expected unchanged31/11 matches and eight clips per match')
    X, y, reconstruction, ids, roles, clips, ball = [], [], [], [], [], [], []
    for row in rows:
        original = source_rows[row['clip_id']]
        for key in ('match_id', 'split'):
            if row[key] != original[key]:
                raise ValueError('Source role or match changed')
        if row['source_sha256'] != original['sha256'] or row['code_sha256'] != capture['code_sha256']:
            raise ValueError('Capture source/code lineage changed')
        path = Path(row['artifact_path'])
        if sha256(path) != row['sha256']:
            raise ValueError('Capture artifact changed')
        with np.load(path, allow_pickle=False) as saved, np.load(original['artifact_path'], allow_pickle=False) as source:
            if saved['X'].shape != (4, 9216) or saved['X'].dtype != np.float32 or not np.isfinite(saved['X']).all():
                raise ValueError('Invalid full spatial VideoMAE descriptors')
            if list(saved['target_names']) != list(ROLE_TARGET_NAMES) or not np.array_equal(saved['view_index'], np.arange(4)):
                raise ValueError('Target ordering or view identities changed')
            # Independent explicit endpoint calculation, not capture's role_targets helper.
            state = source['targets'][:, 15].reshape(4, 5, 6)
            expected = np.stack([np.r_[state[v, v+1], state[v, 0]-state[v, v+1]] for v in range(4)])
            if not np.array_equal(saved['y'], expected) or not np.array_equal(saved['absolute_ball6'], state[:, 0]):
                raise ValueError('Endpoint entity labels do not match source physics')
            indices = source['source_frame_indices']
            expected_indices = np.repeat(indices[15], 4) if indices.ndim == 1 else indices[:, 15]
            if not np.array_equal(saved['source_frame_index'], expected_indices) or not np.array_equal(saved['timestamps'], source['timestamps'][:, 15]):
                raise ValueError('Endpoint timestamp/frame alignment mismatch')
            X.append(saved['X']); y.append(saved['y']); ball.append(saved['absolute_ball6'])
            if row['split'] == 'selection':
                value = saved['X_codec_reconstruction']
                if value.shape != (4, 9216) or not np.isfinite(value).all() or row['codec_calibration'] is None:
                    raise ValueError('Selection codec calibration is incomplete')
                reconstruction.append(value)
        ids.extend([row['match_id']]*4); roles.extend([row['split']]*4); clips.extend([row['clip_id']]*4)
    data = dict(X=np.concatenate(X), y=np.concatenate(y), X_codec_reconstruction=np.concatenate(reconstruction),
                absolute_ball6=np.concatenate(ball), match_ids=np.asarray(ids), split=np.asarray(roles), clip_ids=np.asarray(clips))
    check_development_roles(data['match_ids'], data['split'])
    audit = {'status': 'passed', 'capture_report_sha256': sha256(capture_path),
             'prepared_report_sha256': capture['input_report_sha256'], 'clips': 336, 'view_rows': 1344,
             'match_counts': {'discovery': 31, 'selection': 11}, 'source_labels_independently_rejoined': True,
             'selection_codec_rows': len(data['X_codec_reconstruction']), 'old_confirmation_used': False,
             'encoder_weight_sha256': VIDEOMAE_WEIGHT_SHA, 'audit_script_sha256': sha256(Path(__file__)),
             'codec_calibration': {'clip_count': len(reconstruction),
                 'mean_rgb_mse_0_255': float(np.mean([row['codec_calibration']['rgb_mse_0_255'] for row in rows if row['split'] == 'selection'])),
                 'maximum_clipped_fraction': max(max(row['codec_calibration']['clipped_fraction_by_view']) for row in rows if row['split'] == 'selection')},
             'codec_reconstruction_is_generated_rollout': False}
    return data, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-report', type=Path, default=ROOT/'results/videomae_capture.json')
    p.add_argument('--output-dir', type=Path, default=Path('/data2/ishaangp/mira-interp/video_evaluator_v1/analysis'))
    p.add_argument('--report', type=Path, default=ROOT/'results/videomae_analysis.json')
    args = p.parse_args()
    if args.output_dir.exists() or args.report.exists():
        raise ValueError('Use fresh result paths; prior analyses are not overwritten')
    data, audit = audit_captures(args.capture_report)
    (ROOT/'results/videomae_capture_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    x, y, ids, roles = (data[name] for name in ('X', 'y', 'match_ids', 'split'))
    train, select = roles == 'discovery', roles == 'selection'
    w = match_weights(ids[train])
    shuffled, mapping = shuffle_match_labels(y[train], ids[train], seed=20260907)
    models = ridge_path(x[train], np.c_[y[train], shuffled], w, alphas=ALPHAS)
    curve, predictions = [], []
    for model in models:
        prediction = model.predict(x[select]); predictions.append(prediction)
        curve.append({'alpha': model.alpha,
            'main': metrics(y[select], prediction[:, :12], ids[select], model.y_scale[:12]),
            'shuffled': metrics(y[select], prediction[:, 12:], ids[select], model.y_scale[12:])})
    choices = {variant: int(np.argmin([entry[variant]['normalized_mse'] for entry in curve])) for variant in ('main', 'shuffled')}
    fitted = models[choices['main']].outputs(np.arange(12))
    control = models[choices['shuffled']].outputs(np.arange(12, 24))
    real = fitted.predict(x[select]); codec = fitted.predict(data['X_codec_reconstruction'])
    mean = np.broadcast_to(w @ y[train], y[select].shape)
    args.output_dir.mkdir(parents=True)
    files = {}
    for name, model in [('main', fitted), ('shuffled', control)]:
        path = args.output_dir/f'{name}_ridge.npz'
        np.savez(path, alpha=model.alpha, x_mean=model.x_mean, x_scale=model.x_scale, y_mean=model.y_mean,
                 y_scale=model.y_scale, coefficient=model.coefficient, target_names=np.asarray(ROLE_TARGET_NAMES))
        files[name] = {'path': str(path), 'sha256': sha256(path), 'model_fingerprint': model.fingerprint()}
    drift = np.average(np.abs(codec-real), axis=0, weights=match_weights(ids[select]))
    prediction_path = args.output_dir/'selection_predictions.npz'
    np.savez_compressed(prediction_path, y=y[select], real_prediction=real, codec_prediction=codec,
        mean_prediction=mean, shuffled_prediction=control.predict(x[select]),
        match_ids=ids[select], clip_ids=data['clip_ids'][select],
        view_index=np.tile(np.arange(4), select.sum()//4), target_names=np.asarray(ROLE_TARGET_NAMES))
    report = {'status': 'passed_development_evaluator_analysis', 'scope': 'development selection, not fresh confirmation',
              'capture_audit_sha256': sha256(ROOT/'results/videomae_capture_audit.json'),
              'analysis_code_sha256': sha256(Path(__file__)), 'target_names': list(ROLE_TARGET_NAMES),
              'analysis_dependency_sha256': {name: sha256(ROOT/name) for name in ('src/mira_interp/probes.py', 'src/mira_interp/development.py')},
              'encoder': 'frozen independent VideoMAE; no MIRA hidden features or actions supplied',
              'encoder_weight_sha256': VIDEOMAE_WEIGHT_SHA, 'feature_dimension': 9216,
              'match_counts': audit['match_counts'], 'rows_per_match': 32,
              'alphas': list(ALPHAS), 'selection_curve': curve, 'selected_alpha': fitted.alpha,
              'selection_rule': 'minimum equal-match normalized MSE over all12 targets on real selection clips only',
              'selected_alpha_at_boundary': choices['main'] in (0, len(ALPHAS)-1),
              'normalization_and_regression_fit': 'real discovery only', 'label_shuffle_mapping': mapping,
              'real_selection': metrics(y[select], real, ids[select], fitted.y_scale),
              'mean_baseline': metrics(y[select], mean, ids[select], fitted.y_scale),
              'shuffled_selection': metrics(y[select], control.predict(x[select]), ids[select], fitted.y_scale),
              'codec_selection': metrics(y[select], codec, ids[select], fitted.y_scale),
              'codec_prediction_change_mae_by_target': drift.tolist(), 'models': files,
              'selection_predictions': {'path': str(prediction_path), 'sha256': sha256(prediction_path)},
              'per_match_selection': {str(match): {
                  'real': metrics(y[select][ids[select] == match], real[ids[select] == match], ids[select][ids[select] == match], fitted.y_scale),
                  'codec': metrics(y[select][ids[select] == match], codec[ids[select] == match], ids[select][ids[select] == match], fitted.y_scale)
              } for match in np.unique(ids[select])},
              'codec_calibration': audit['codec_calibration'],
              'codec_used_to_fit_or_choose_regressor': False,
              'independent_generated_video_measurement_ready': False,
              'limits': ['Selection was used for alpha choice; these are development results.',
                         'Codec roundtrips may themselves alter visible state; source annotations are reference labels.',
                         'Codec-domain transfer is not generated-rollout or intervention-domain validation.',
                         'Low error must be judged against the intended physical dose and identity/occlusion failures.',
                         'Full-frame square resize deliberately changes aspect ratio and preserves camera edges.']}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'alpha': fitted.alpha,
                      'real_nmse': report['real_selection']['normalized_mse'],
                      'codec_nmse': report['codec_selection']['normalized_mse']}))


if __name__ == '__main__':
    main()
