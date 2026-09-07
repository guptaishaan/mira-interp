#!/usr/bin/env python3
"""Score unchanged frozen predictors on the partial-adaptation input pipeline."""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from finetune_video_evaluator import load_models
from mira_interp.development import metrics
from mira_interp.model_loading import sha256
from mira_interp.probes import RidgeModel


def main():
    final_path = ROOT/'results/videomae_finetune_v1.json'; final = json.loads(final_path.read_text())
    output = ROOT/'results/videomae_matched_input_calibration.json'
    if output.exists() or final['status'] != 'passed_development_partial_finetune': raise ValueError('Complete adaptation once before calibration')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    prefix_path = ROOT/'results/videomae_prefix.json'; cache = json.loads(prefix_path.read_text())
    if cache['status'] != 'passed': raise ValueError('Require passed prefix cache')
    for artifact in cache['artifacts'].values():
        if sha256(Path(artifact['path'])) != artifact['sha256']: raise ValueError('Cache changed')
    with np.load(cache['artifacts']['metadata']['path'], allow_pickle=False) as metadata:
        role = metadata['split']; keep = role == 'selection'; y = metadata['y'][keep]; ids = metadata['match_ids'][keep]
        if set(role) != {'discovery','selection'} or len(y) != 352: raise ValueError('Unexpected or forbidden cohort')
    ridge_report = json.loads((ROOT/'results/videomae_analysis.json').read_text())
    artifact = ridge_report['models']['main']
    if sha256(Path(artifact['path'])) != artifact['sha256']: raise ValueError('Frozen ridge changed')
    with np.load(artifact['path'], allow_pickle=False) as saved:
        ridge = RidgeModel(float(saved['alpha']), *[saved[name].copy() for name in ('x_mean','x_scale','y_mean','y_scale','coefficient')])
    encoder, tail, preprocess, loading = load_models(); tail.eval()
    scale = tail.head.y_scale.detach().cpu().numpy().astype(np.float64)
    reports, arrays = {}, {'y': y, 'match_ids': ids}
    for domain, key in [('real','real'),('codec','codec')]:
        values = np.load(cache['artifacts'][key]['path'], mmap_mode='r', allow_pickle=False)
        if domain == 'real': values = np.asarray(values[keep])
        descriptors, head_predictions = [], []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, len(values), 4):
                batch = torch.tensor(np.asarray(values[start:start+4]), dtype=torch.float32, device='cuda')
                descriptor = tail.descriptor(batch)
                prediction = tail.head.predict(descriptor)
                descriptors.append(descriptor.float().cpu().numpy()); head_predictions.append(prediction.float().cpu().numpy())
        descriptors = np.concatenate(descriptors); mlp = np.concatenate(head_predictions).astype(np.float64); linear = ridge.predict(descriptors)
        reports[domain+'_selection'] = {'MLP': metrics(y, mlp, ids, scale), 'ridge': metrics(y, linear, ids, ridge.y_scale)}
        arrays[domain+'_MLP_prediction'], arrays[domain+'_ridge_prediction'] = mlp, linear
    if not np.isclose(reports['real_selection']['MLP']['normalized_mse'], final['initial_real_selection']['normalized_mse'], rtol=1e-10, atol=1e-12):
        raise ValueError('Unadapted pipeline no longer reproduces the recorded step0 score')
    before = {row['clip_id']:row['codec_calibration'] for row in json.loads((ROOT/'results/videomae_capture.json').read_text())['records'] if row['split']=='selection'}
    differences = [row['codec_calibration']['rgb_mse_0_255']-before[row['clip_id']]['rgb_mse_0_255'] for row in cache['records'] if row['split']=='selection']
    prediction_path = Path('/data2/ishaangp/mira-interp/video_evaluator_v1/partial_finetune/matched_frozen_predictions.npz')
    np.savez_compressed(prediction_path, **arrays)
    report = {'status':'passed_matched_evaluator_input_calibration','scope':'unchanged predictors; same prefix/reconstruction inputs as adapted model; no fitting or selection',
        'script_sha256':sha256(Path(__file__)),'prefix_report_sha256':sha256(prefix_path),'adapted_report_sha256':sha256(final_path),
        'loading':loading,'ridge_model':artifact,'target_names':final['target_names'],**reports,
        'reconstruction_source_mse_shift_mean':float(np.mean(differences)),'reconstruction_source_mse_shift_max_abs':float(np.max(np.abs(differences))),
        'codec_pixel_arrays_bitwise_equal_to_original_pipeline':False,
        'predictions':{'path':str(prediction_path),'sha256':sha256(prediction_path)},
        'no_additional_model_selection':True,'independent_generated_video_measurement_ready':False}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(json.dumps({'status':report['status'],'real_mlp_nmse':reports['real_selection']['MLP']['normalized_mse'],'codec_mlp_nmse':reports['codec_selection']['MLP']['normalized_mse']}))


if __name__ == '__main__': main()
