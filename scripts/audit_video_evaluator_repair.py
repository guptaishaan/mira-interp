#!/usr/bin/env python3
"""Rejoin source labels and independently recompute completed adaptation metrics."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from analyze_video_evaluator import audit_captures
from mira_interp.model_loading import sha256


def main():
    report_path = ROOT/'results/videomae_finetune_v1.json'; report = json.loads(report_path.read_text())
    config_path = ROOT/'configs/videomae_partial_finetune_v1.json'; config = json.loads(config_path.read_text())
    if report['status'] != 'passed_development_partial_finetune' or report['completed_updates'] != 500:
        raise ValueError('Require all500 completed development updates')
    if sha256(config_path) != report['registration_sha256'] or config['status'] != 'registered_before_research_optimization': raise ValueError('Registration changed')
    if not datetime.fromisoformat(config['registered_utc']) <= datetime.fromisoformat(report['started_utc']) <= datetime.fromisoformat(report['finished_utc']): raise ValueError('Registration/training order invalid')
    for name, digest in config['code_sha256'].items():
        if sha256(ROOT/name) != digest: raise ValueError('Registered source changed')
    for source in config['inputs'].values():
        if sha256(Path(source['path'])) != source['sha256']: raise ValueError('Prerequisite report changed')
    if config['head_learning_rate'] != 1e-4 or report['actual_optimizer_learning_rates'] != [1e-5, 1e-4]: raise ValueError('Conservative learning rates not used')
    if [row['step'] for row in report['curve']] != [100,250,500]: raise ValueError('Checkpoint budget changed')
    selected = min(report['curve'], key=lambda row: row['real_selection']['normalized_mse'])
    if selected['step'] != report['selected_step']: raise ValueError('Incorrect selected checkpoint')
    if not report['frozen_prefix_parameters_unchanged'] or report['frozen_prefix_sha256_before'] != report['frozen_prefix_sha256_after']: raise ValueError('Frozen parameter integrity failed')
    if report['independent_generated_video_measurement_ready']: raise ValueError('Unsupported generated-video certification claim')
    for name in ('checkpoint', 'predictions'):
        if sha256(Path(report[name]['path'])) != report[name]['sha256']: raise ValueError('Final model or predictions changed')
    state = torch.load(report['checkpoint']['path'], map_location='cpu', weights_only=True)
    if sha256(Path(config['head_checkpoint']['path'])) != config['head_checkpoint']['sha256']: raise ValueError('Original selected head changed')
    original_head = torch.load(config['head_checkpoint']['path'], map_location='cpu', weights_only=True)
    for name in ('x_mean','x_scale','y_mean','y_scale'):
        if not torch.equal(state['head.'+name], original_head[name]): raise ValueError('Discovery normalization changed during adaptation')
    if not all(torch.isfinite(value).all() for value in state.values()): raise ValueError('Nonfinite saved model')
    for block in [0,1]:
        if torch.count_nonzero(state[f'blocks.{block}.attention.attention.key.bias']): raise ValueError('Originally fixed zero key bias changed')
    if any(not key.startswith(('blocks.0.', 'blocks.1.', 'norm.', 'head.')) for key in state): raise ValueError('Unexpected trained components')
    data, source_audit = audit_captures(ROOT/'results/videomae_capture.json')
    selection = data['split'] == 'selection'; y = data['y'][selection]; ids = data['match_ids'][selection]
    groups, counts = np.unique(ids, return_counts=True)
    if len(groups) != 11 or set(counts) != {32}: raise ValueError('Expected eleven equally weighted selection matches')
    with np.load(report['predictions']['path'], allow_pickle=False) as saved:
        if not np.array_equal(saved['y'], y) or not np.array_equal(saved['match_ids'], ids): raise ValueError('Saved predictions are not aligned to source labels')
        scale = state['head.y_scale'].numpy().astype(np.float64); differences = []
        for domain in ('real', 'codec'):
            predicted = saved[domain+'_prediction']; error = predicted-y
            if predicted.shape != (352,12) or not np.isfinite(predicted).all(): raise ValueError('Invalid endpoint predictions')
            mse = np.mean(error**2, axis=0); normalized = mse/scale**2
            expected = {'mae': np.mean(np.abs(error), axis=0), 'rmse': np.sqrt(mse),
                'r2': 1-mse/np.var(y, axis=0), 'normalized_mse_by_target': normalized}
            actual = report[domain+'_selection']
            for metric, value in expected.items():
                if not np.allclose(value, actual[metric], rtol=1e-10, atol=1e-8): raise ValueError('Recomputed state metrics disagree')
                differences.append(float(np.max(np.abs(value-np.asarray(actual[metric])))))
            if not np.isclose(normalized.mean(), actual['normalized_mse'], rtol=1e-10, atol=1e-12): raise ValueError('Aggregate error mismatch')
        drift = np.abs(saved['codec_prediction']-saved['real_prediction']).mean(axis=0)
        if not np.allclose(drift, report['codec_prediction_change_mae_by_target'], rtol=1e-10, atol=1e-8): raise ValueError('Reconstruction drift mismatch')
    audit = {'status': 'passed_development_video_evaluator_audit', 'analysis_sha256': sha256(report_path),
        'registration_sha256': sha256(config_path), 'audit_script_sha256': sha256(Path(__file__)),
        'capture_report_sha256': source_audit['capture_report_sha256'], 'selection_matches': 11, 'selection_view_rows': 352,
        'source_endpoint_labels_independently_rejoined': True, 'metric_max_absolute_difference': max(differences),
        'checkpoint_and_prediction_hashes_verified': True, 'discovery_normalizers_unchanged': True,
        'frozen_prefix_hash_unchanged': True, 'original_zero_key_biases_unchanged': True,
        'registered_before_optimization': True, 'old_or_fresh_confirmation_used': False,
        'scope': 'development annotation decoding and codec-domain transfer; generated intervention calibration remains unestablished'}
    out = ROOT/'results/videomae_finetune_v1_audit.json'
    if out.exists(): raise ValueError('Do not overwrite completed audits')
    out.write_text(json.dumps(audit, indent=2)+'\n'); print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()
