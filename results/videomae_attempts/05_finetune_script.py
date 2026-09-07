#!/usr/bin/env python3
"""Ordered reserved pilot, frozen-prefix capture, and bounded last-two-block adaptation."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'external/mira/src'))
import numpy as np
import torch
from analyze_video_evaluator import audit_captures
from mira_interp.development import metrics
from mira_interp.model_loading import sha256
from mira_interp.video_evaluator import load_encoder, preprocess_frames, pool_last_tubelet, endpoint_labels, load_roundtrip_codec, codec_roundtrip
from mira_interp.video_evaluator_training import SpatialStateHead, enable_last_blocks
from mira_interp.video_evaluator_finetune import VideoMAETail, frozen_prefix, optimizer_for, parameter_digest

BASE = Path('/data2/ishaangp/mira-interp/video_evaluator_v1/partial_finetune')
ASSETS = Path('/data2/ishaangp/mira-interp/models/videomae-base')
MIRA_ASSETS = Path('/data2/ishaangp/mira-interp/assets')
CODE_NAMES = ('scripts/finetune_video_evaluator.py', 'src/mira_interp/video_evaluator_finetune.py',
              'src/mira_interp/video_evaluator_training.py', 'src/mira_interp/video_evaluator.py')


def save(path, report):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def load_models():
    report = json.loads((ROOT/'results/videomae_mlp.json').read_text())
    if report['status'] != 'passed_development_mlp_analysis':
        raise ValueError('Complete cached-feature MLP comparison first')
    for name, expected in report['code_sha256'].items():
        if sha256(ROOT/name) != expected: raise ValueError('MLP implementation changed')
    selected = next(row for row in report['runs'] if row['seed'] == report['selected_seed'])
    path = Path(selected['checkpoint']['path'])
    if sha256(path) != selected['checkpoint']['sha256']: raise ValueError('MLP head changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    head = SpatialStateHead(*[state[name].numpy() for name in ('x_mean', 'x_scale', 'y_mean', 'y_scale')])
    head.load_state_dict(state, strict=True)
    encoder, preprocess, loading = load_encoder(ASSETS)
    encoder.to('cuda'); head.to('cuda'); encoder.eval()
    counts = enable_last_blocks(encoder)
    tail = VideoMAETail(encoder, head).cuda()
    counts.update(trainable=sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                  frozen=sum(p.numel() for p in encoder.parameters() if not p.requires_grad),
                  original_zero_key_biases_fixed=True)
    if any(p.dtype != torch.float32 for p in encoder.parameters()) or any(p.dtype != torch.float32 for p in head.parameters()):
        raise ValueError('All encoder/head parameters must remain FP32')
    return encoder, tail, preprocess, {'encoder': loading, 'encoder_trainability': counts,
        'head_checkpoint': selected['checkpoint'], 'head_seed': selected['seed'], 'head_step': selected['selected_step'],
        'head_parameters': sum(p.numel() for p in head.parameters()), 'mlp_report_sha256': sha256(ROOT/'results/videomae_mlp.json')}


def prefix_batch(encoder, frames, preprocess):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return frozen_prefix(encoder, preprocess_frames(frames, preprocess, 'cuda')).float().cpu().numpy()


def pilot(report, path):
    started = time.monotonic(); torch.manual_seed(0)
    encoder, tail, preprocess, report['loading'] = load_models()
    source = json.loads((ROOT/'results/development_prepare_pilot.json').read_text())['records'][0]
    if source['split'] != 'pilot' or sha256(Path(source['artifact_path'])) != source['sha256']:
        raise ValueError('Require unchanged reserved source pilot')
    with np.load(source['artifact_path'], allow_pickle=False) as clip:
        frames, labels = clip['frames'], endpoint_labels(clip)
    cache = json.loads((ROOT/'results/videomae_pilot.json').read_text())['records'][0]
    if sha256(Path(cache['artifact_path'])) != cache['sha256']: raise ValueError('Frozen pilot features changed')
    with np.load(cache['artifact_path'], allow_pickle=False) as cached:
        original_feature = cached['X'][0].copy()
    one_video = preprocess_frames(frames[:1], preprocess, 'cuda')
    tail.eval()
    with torch.no_grad():
        feature = pool_last_tubelet(encoder(pixel_values=one_video).last_hidden_state)
    fp32_error = float(np.abs(feature.cpu().numpy()[0]-original_feature).max())
    if not np.allclose(feature.cpu().numpy()[0], original_feature, rtol=1e-5, atol=1e-5):
        raise ValueError('Strict FP32 pilot encoder no longer matches original capture')
    video = preprocess_frames(frames, preprocess, 'cuda')
    before_frozen = parameter_digest((name, p) for name, p in encoder.named_parameters() if not p.requires_grad)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prefix = frozen_prefix(encoder, video)
        full_descriptor = pool_last_tubelet(encoder(pixel_values=video).last_hidden_state)
        cached_descriptor = tail.descriptor(prefix.detach().cpu().float().cuda())
    feature_error = float((cached_descriptor-full_descriptor).abs().max())
    if not torch.allclose(cached_descriptor, full_descriptor, rtol=1e-5, atol=1e-5):
        raise ValueError('Cached frozen prefix does not reproduce full mixed-precision encoder')
    target = (torch.tensor(labels['y'], device='cuda', dtype=torch.float32)-tail.head.y_mean)/tail.head.y_scale
    with torch.autocast('cuda', dtype=torch.bfloat16):
        full_prediction = tail.head(pool_last_tubelet(encoder(pixel_values=video).last_hidden_state))
        loss_full = torch.mean((full_prediction-target)**2)
    loss_full.backward()
    gradients = {name: p.grad.detach().clone() for name, p in tail.named_parameters() if p.grad is not None}
    if len(gradients) != sum(p.requires_grad for p in tail.parameters()): raise ValueError('Not all selected parameters receive gradients')
    tail.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss_cached = torch.mean((tail(prefix.detach().cpu().float().cuda())-target)**2)
    loss_cached.backward()
    gradient_error = max(float((p.grad-gradients[name]).abs().max()) for name, p in tail.named_parameters() if p.requires_grad)
    cached_gradients = {name: p.grad.detach().clone() for name, p in tail.named_parameters() if p.requires_grad}
    tail.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        repeated_loss = torch.mean((tail.head(pool_last_tubelet(encoder(pixel_values=video).last_hidden_state))-target)**2)
    repeated_loss.backward()
    diagnostics = {}
    for name, p in tail.named_parameters():
        if not p.requires_grad: continue
        reference, cached_grad = gradients[name].float(), cached_gradients[name].float()
        denominator = max(float(reference.norm()), 1e-30)
        diagnostics[name] = {'full_gradient_norm': float(reference.norm()),
            'cache_max_abs': float((cached_grad-reference).abs().max()),
            'cache_relative_l2': float((cached_grad-reference).norm())/denominator,
            'repeat_max_abs': float((p.grad-reference).abs().max()),
            'repeat_relative_l2': float((p.grad-reference).norm())/denominator}
    report.update(gradient_diagnostics=diagnostics, prefix_requires_grad=prefix.requires_grad,
        prefix_native_dtype=str(prefix.dtype), prefix_native_stride=list(prefix.stride()),
        restored_prefix_stride=list(prefix.detach().cpu().float().cuda().to(prefix.dtype).stride()),
        attention_implementation=encoder.config._attn_implementation,
        cached_full_descriptor_max_abs=feature_error, fp32_reference_max_abs=fp32_error,
        full_loss=float(loss_full.detach()), cached_loss=float(loss_cached.detach()), repeated_loss=float(repeated_loss.detach()))
    save(path, report)
    if any(not torch.allclose(cached_gradients[name], gradients[name], rtol=1e-5, atol=1e-5) for name, p in tail.named_parameters() if p.requires_grad):
        raise ValueError('Cached-prefix gradients differ from full frozen-prefix computation')
    before_trainable = parameter_digest(tail.named_parameters())
    optimizer = optimizer_for(tail)
    torch.nn.utils.clip_grad_norm_(tail.parameters(), 1., error_if_nonfinite=True); optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss_after = torch.mean((tail(prefix.detach())-target)**2)
    loss_after.backward(); torch.nn.utils.clip_grad_norm_(tail.parameters(), 1., error_if_nonfinite=True); optimizer.step()
    if parameter_digest((name, p) for name, p in encoder.named_parameters() if not p.requires_grad) != before_frozen:
        raise ValueError('Frozen encoder parameters changed')
    if parameter_digest(tail.named_parameters()) == before_trainable: raise ValueError('Trainable parameters did not update')
    report.update(status='passed', source_clip_sha256=source['sha256'], fp32_reference_max_abs=fp32_error,
        cached_full_descriptor_max_abs=feature_error, cached_full_gradient_max_abs=gradient_error,
        prefix_native_dtype=str(prefix.dtype), prefix_cached_dtype='float32', parameters_fp32=True,
        frozen_parameters_unchanged=True, trainable_parameters_changed=True, discarded_pilot_updates=2,
        initial_standardized_loss=float(loss_full.detach()), after_one_update_loss=float(loss_after.detach()),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started)
    save(path, report); print(json.dumps(report, indent=2))


def capture(report, path):
    started = time.monotonic()
    data, audit = audit_captures(ROOT/'results/videomae_capture.json')
    output = BASE/'prefix'; output.mkdir(parents=True, exist_ok=False)
    bytes_needed = (1344+352)*1568*768*4
    if shutil.disk_usage(output).free-bytes_needed < 25*1024**3: raise ValueError('Insufficient prefix-cache disk reserve')
    real_path, codec_path = output/'real.npy', output/'codec_selection.npy'
    real = np.lib.format.open_memmap(real_path, mode='w+', dtype=np.float32, shape=(1344, 1568, 768))
    reconstruction = np.lib.format.open_memmap(codec_path, mode='w+', dtype=np.float32, shape=(352, 1568, 768))
    encoder, tail, preprocess, report['loading'] = load_models(); encoder.eval()
    codec, report['codec_loading'] = load_roundtrip_codec(MIRA_ASSETS); codec.cuda()
    del tail
    original = json.loads((ROOT/'results/videomae_capture.json').read_text())
    records = sorted(original['records'], key=lambda row: (row['match_id'], row['clip_id']))
    offset = 0; report['records'] = []; save(path, report)
    for index, row in enumerate(records):
        source = Path(row['source_artifact_path'])
        if sha256(source) != row['source_sha256']: raise ValueError('Source video changed before prefix capture')
        with np.load(source, allow_pickle=False) as clip:
            frames = clip['frames']; endpoint = endpoint_labels(clip)
            if frames.dtype != np.float32 or frames.shape != (4, 16, 3, 288, 512): raise ValueError('Expected original float video')
        if not np.array_equal(endpoint['y'], data['y'][4*index:4*index+4]): raise ValueError('Prefix labels or view order changed')
        real[4*index:4*index+4] = prefix_batch(encoder, frames, preprocess)
        calibration = None
        if row['split'] == 'selection':
            reconstructed, calibration = codec_roundtrip(codec, frames, device='cuda')
            reconstruction[offset:offset+4] = prefix_batch(encoder, reconstructed, preprocess); offset += 4
        report['records'].append({'clip_id': row['clip_id'], 'match_id': row['match_id'], 'split': row['split'],
            'source_sha256': row['source_sha256'], 'row_start': 4*index, 'codec_calibration': calibration})
        report.update(completed_clips=index+1, expected_clips=336, elapsed_seconds=time.monotonic()-started)
        save(path, report)
        print(json.dumps({key: report[key] for key in ('completed_clips', 'expected_clips', 'elapsed_seconds')}), flush=True)
    if offset != 352: raise ValueError('Missing selection reconstruction rows')
    real.flush(); reconstruction.flush(); del real, reconstruction
    metadata_path = output/'metadata.npz'
    np.savez_compressed(metadata_path, **{name: data[name] for name in ('y', 'match_ids', 'split', 'clip_ids')}, view_index=np.tile(np.arange(4), 336))
    report.update(status='passed', capture_report_sha256=audit['capture_report_sha256'],
        artifacts={name: {'path': str(value), 'sha256': sha256(value)} for name, value in [('real', real_path), ('codec', codec_path), ('metadata', metadata_path)]},
        rows=1344, codec_rows=352, shape_per_row=[1568, 768], dtype='float32',
        arithmetic='FP32 weights, BF16 autocast, frozen first10 encoder blocks; cache stores resulting values asFP32 without further rounding',
        elapsed_seconds=time.monotonic()-started)
    save(path, report); print(json.dumps({'status': 'passed', 'elapsed_seconds': report['elapsed_seconds']}))


def predict(tail, prefixes):
    tail.eval(); values = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(prefixes), 4):
            batch = torch.tensor(np.asarray(prefixes[start:start+4]), dtype=torch.float32, device='cuda')
            values.append(tail.predict(batch).float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def train(report, path):
    started = time.monotonic(); torch.manual_seed(0)
    cache_path = ROOT/'results/videomae_prefix.json'; cache = json.loads(cache_path.read_text())
    if cache['status'] != 'passed' or cache['code_sha256'] != report['code_sha256'] or cache['rows'] != 1344 or cache['codec_rows'] != 352:
        raise ValueError('Complete same-code development prefix capture first')
    for artifact in cache['artifacts'].values():
        if sha256(Path(artifact['path'])) != artifact['sha256']: raise ValueError('Cached prefix artifact changed')
    with np.load(cache['artifacts']['metadata']['path'], allow_pickle=False) as metadata:
        y, ids, roles = (metadata[name].copy() for name in ('y', 'match_ids', 'split'))
    if set(roles) != {'discovery', 'selection'}: raise ValueError('Confirmation is forbidden')
    real = np.load(cache['artifacts']['real']['path'], mmap_mode='r', allow_pickle=False)
    codec = np.load(cache['artifacts']['codec']['path'], mmap_mode='r', allow_pickle=False)
    discovery, selection = np.flatnonzero(roles == 'discovery'), np.flatnonzero(roles == 'selection')
    if len(discovery) != 992 or len(selection) != 352: raise ValueError('Development observation counts changed')
    # A finite contiguous evaluation buffer avoids repeatedly indexing the full mmap.
    selected_prefix = np.asarray(real[selection]); encoder, tail, preprocess, report['loading'] = load_models()
    optimizer = optimizer_for(tail); rng = np.random.default_rng(0)
    yscale = tail.head.y_scale.detach().cpu().numpy(); ymean = tail.head.y_mean.detach().cpu().numpy()
    target = (y-ymean)/yscale
    before_frozen = parameter_digest((name, p) for name, p in encoder.named_parameters() if not p.requires_grad)
    initial = predict(tail, selected_prefix)
    report.update(prefix_report_sha256=sha256(cache_path), checkpoint_steps=[100, 250, 500], updates=500,
        batch_size=4, accumulation=4, effective_batch=16, backbone_learning_rate=1e-5, head_learning_rate=.001,
        weight_decay=.01, gradient_clip_norm=1., seed=0, initial_real_selection=metrics(y[selection], initial, ids[selection], yscale),
        target_names=json.loads((ROOT/'results/videomae_analysis.json').read_text())['target_names'],
        selection_rule='minimum equal-match real-selection12-target normalized MSE over steps100/250/500; no codec selection',
        independent_generated_video_measurement_ready=False, curve=[])
    best = None; save(path, report)
    for step in range(1, 501):
        tail.train(); optimizer.zero_grad(set_to_none=True); total_loss = 0.
        for micro in range(4):
            indices = rng.choice(discovery, size=4, replace=True)
            prefix = torch.tensor(np.asarray(real[indices]), dtype=torch.float32, device='cuda')
            expected = torch.tensor(target[indices], dtype=torch.float32, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = torch.mean((tail(prefix)-expected)**2)/4
            if not torch.isfinite(loss): raise ValueError('Nonfinite partial fine-tuning loss')
            loss.backward(); total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(tail.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        if step % 10 == 0:
            report.update(completed_updates=step, last_batch_loss=total_loss, elapsed_seconds=time.monotonic()-started)
            save(path, report); print(json.dumps({key: report[key] for key in ('completed_updates', 'last_batch_loss', 'elapsed_seconds')}), flush=True)
        if step in report['checkpoint_steps']:
            score = metrics(y[selection], predict(tail, selected_prefix), ids[selection], yscale)
            report['curve'].append({'step': step, 'real_selection': score}); save(path, report)
            if best is None or score['normalized_mse'] < best[0]: best = (score['normalized_mse'], step, copy.deepcopy(tail.state_dict()))
    if parameter_digest((name, p) for name, p in encoder.named_parameters() if not p.requires_grad) != before_frozen: raise ValueError('Frozen prefix changed during training')
    tail.load_state_dict(best[2], strict=True); real_prediction = predict(tail, selected_prefix); codec_prediction = predict(tail, codec)
    output = BASE/'fit'; output.mkdir(parents=True, exist_ok=False)
    checkpoint = output/f'step{best[1]}.pt'; torch.save(tail.state_dict(), checkpoint)
    predictions = output/'selection_predictions.npz'; np.savez_compressed(predictions, y=y[selection], real_prediction=real_prediction, codec_prediction=codec_prediction, match_ids=ids[selection])
    report.update(status='passed_development_partial_finetune', selected_step=best[1],
        real_selection=metrics(y[selection], real_prediction, ids[selection], yscale), codec_selection=metrics(y[selection], codec_prediction, ids[selection], yscale),
        codec_prediction_change_mae_by_target=np.abs(codec_prediction-real_prediction).mean(axis=0).tolist(), frozen_prefix_parameters_unchanged=True,
        checkpoint={'path': str(checkpoint), 'sha256': sha256(checkpoint)}, predictions={'path': str(predictions), 'sha256': sha256(predictions)},
        elapsed_seconds=time.monotonic()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save(path, report); print(json.dumps({'status': report['status'], 'selected_step': best[1], 'real_nmse': report['real_selection']['normalized_mse']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['pilot', 'capture', 'train']); args = parser.parse_args()
    if not torch.cuda.is_available(): raise ValueError('Explicit authorized visible GPU required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    codes = {name: sha256(ROOT/name) for name in CODE_NAMES}
    path = ROOT/'results'/f"videomae_{ {'pilot': 'finetune_pilot', 'capture': 'prefix', 'train': 'finetune'}[args.stage] }.json"
    if path.exists(): raise ValueError('Do not overwrite partial fine-tuning artifacts')
    report = {'status': 'running', 'stage': args.stage, 'scope': 'original31/11 development only; no confirmation',
              'code_sha256': codes, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'deterministic_algorithms': True, 'attention_backend': 'math', 'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']}
    if args.stage != 'pilot':
        evidence = ROOT/'results/videomae_finetune_pilot.json'; checked = json.loads(evidence.read_text())
        if checked['status'] != 'passed' or checked['code_sha256'] != codes: raise ValueError('Same-code reserved partial fine-tuning pilot must pass')
        report['pilot_sha256'] = sha256(evidence)
    save(path, report)
    try: {'pilot': pilot, 'capture': capture, 'train': train}[args.stage](report, path)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}'); save(path, report); raise


if __name__ == '__main__': main()
