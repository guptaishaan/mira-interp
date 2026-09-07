#!/usr/bin/env python3
"""Register and run the fixed conservative warm-start VideoMAE adaptation."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from finetune_video_evaluator import BASE, CODE_NAMES, load_models, predict, save
from mira_interp.development import metrics
from mira_interp.model_loading import sha256
from mira_interp.video_evaluator_finetune import optimizer_for, parameter_digest

CONFIG = ROOT/'configs/videomae_partial_finetune_v1.json'
REPORT = ROOT/'results/videomae_finetune_v1.json'
CODES = ('scripts/train_video_evaluator_partial.py', *CODE_NAMES, 'src/mira_interp/development.py')


def sources():
    paths = {'pilot': ROOT/'results/videomae_finetune_pilot.json',
             'prefix': ROOT/'results/videomae_prefix.json', 'head': ROOT/'results/videomae_mlp.json'}
    expected = {'pilot': 'passed', 'prefix': 'passed', 'head': 'passed_development_mlp_analysis'}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if any(reports[name]['status'] != expected[name] for name in paths): raise ValueError('All engineering and data prerequisites must pass')
    for key in ('pilot', 'prefix'):
        if reports[key]['code_sha256'] != {name: sha256(ROOT/name) for name in CODE_NAMES}: raise ValueError('Verified computation changed')
    if reports['prefix']['pilot_sha256'] != sha256(paths['pilot']): raise ValueError('Prefix capture used a different pilot')
    return reports, {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()}


def register():
    if CONFIG.exists() or REPORT.exists(): raise ValueError('Do not overwrite the prospective training registration')
    reports, inputs = sources()
    selected_head = next(row for row in reports['head']['runs'] if row['seed'] == reports['head']['selected_seed'])
    config = {'status': 'registered_before_research_optimization', 'scope': 'original31/11 development only; confirmation forbidden',
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': inputs, 'code_sha256': {name: sha256(ROOT/name) for name in CODES}, 'seed': 0,
        'updates': 500, 'microbatch': 4, 'accumulation': 4, 'effective_batch': 16,
        'backbone_learning_rate': 1e-5, 'head_learning_rate': 1e-4, 'weight_decay': .01, 'gradient_clip_norm': 1.,
        'checkpoint_steps': [100, 250, 500], 'match_counts': {'discovery': 31, 'selection': 11},
        'trainable_encoder_blocks': [10, 11], 'fixed_original_key_biases': True,
        'normalization': 'original discovery-only feature and target normalizers, fixed throughout',
        'head_initialization': f"already selected frozen-feature MLP seed{selected_head['seed']} step{selected_head['selected_step']}, unchanged checkpoint hash bound below",
        'head_checkpoint': selected_head['checkpoint'],
        'selection_rule': 'minimum real-selection equal-match12-target normalized MSE across exactly100/250/500 updates',
        'codec_used_for_training_or_selection': False, 'precision': 'FP32 parameters; BF16 autocast; native BF16 prefix restored; deterministic FP32 spatial averaging',
        'attention_backend': 'math', 'deterministic_algorithms': True,
        'prospective_amendment': 'HeadLR1e-4 replaces proposed1e-3 before any research optimization. The discarded reserved pilot showed first-update loss0.497->0.933; use conservative adaptation of the already trained head, with no further hyperparameter search.'}
    save(CONFIG, config); print(json.dumps({'status': config['status'], 'path': str(CONFIG), 'sha256': sha256(CONFIG)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--register', action='store_true'); args = parser.parse_args()
    if args.register: register(); return
    if REPORT.exists() or (BASE/'fit_v1').exists(): raise ValueError('Do not overwrite existing partial-training results')
    config = json.loads(CONFIG.read_text()); bound, current = sources()
    if config['inputs'] != current or config['code_sha256'] != {name: sha256(ROOT/name) for name in CODES}: raise ValueError('Registered inputs or implementation changed')
    if not torch.cuda.is_available(): raise ValueError('Explicit authorized visible GPU required')
    torch.set_num_threads(4); torch.manual_seed(config['seed']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    cache = bound['prefix']
    for artifact in cache['artifacts'].values():
        if sha256(Path(artifact['path'])) != artifact['sha256']: raise ValueError('Prefix artifact changed after capture')
    with np.load(cache['artifacts']['metadata']['path'], allow_pickle=False) as metadata:
        y, ids, roles = (metadata[name].copy() for name in ('y', 'match_ids', 'split'))
    if set(roles) != {'discovery', 'selection'} or any(len(set(roles[ids == match])) != 1 for match in np.unique(ids)):
        raise ValueError('Development identities/splits overlap or include confirmation')
    discovery, selection = np.flatnonzero(roles == 'discovery'), np.flatnonzero(roles == 'selection')
    if len(discovery) != 992 or len(selection) != 352: raise ValueError('Expected992/352 rows')
    real = np.load(cache['artifacts']['real']['path'], mmap_mode='r', allow_pickle=False)
    codec = np.load(cache['artifacts']['codec']['path'], mmap_mode='r', allow_pickle=False)
    if real.shape != (1344, 1568, 768) or codec.shape != (352, 1568, 768): raise ValueError('Invalid prefix shapes')
    selected_prefix = np.asarray(real[selection])
    encoder, tail, preprocess, loading = load_models()
    optimizer = optimizer_for(tail)
    optimizer.param_groups[0]['lr'] = config['backbone_learning_rate']
    optimizer.param_groups[1]['lr'] = config['head_learning_rate']
    yscale, ymean = (getattr(tail.head, name).detach().cpu().numpy().astype(np.float64) for name in ('y_scale', 'y_mean'))
    target = (y-ymean)/yscale; rng = np.random.default_rng(config['seed'])
    before_frozen = parameter_digest((name, p) for name, p in encoder.named_parameters() if not p.requires_grad)
    initial = predict(tail, selected_prefix)
    baseline = next(row for row in bound['head']['runs'] if row['seed'] == bound['head']['selected_seed'])
    report = {'status': 'running', 'registration_sha256': sha256(CONFIG), 'loading': loading,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': config['scope'], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'actual_optimizer_learning_rates': [group['lr'] for group in optimizer.param_groups],
        'initial_real_selection': metrics(y[selection], initial, ids[selection], yscale),
        'frozen_head_real_selection': baseline['real_selection'], 'frozen_head_codec_selection': baseline['codec_selection'],
        'target_names': bound['head']['target_names'], 'completed_updates': 0, 'curve': [],
        'independent_generated_video_measurement_ready': False}
    save(REPORT, report); started = time.monotonic(); best = None
    try:
        for step in range(1, config['updates']+1):
            tail.train(); optimizer.zero_grad(set_to_none=True); total_loss = 0.
            for micro in range(config['accumulation']):
                indices = rng.choice(discovery, size=config['microbatch'], replace=True)
                prefix = torch.tensor(np.asarray(real[indices]), dtype=torch.float32, device='cuda')
                expected = torch.tensor(target[indices], dtype=torch.float32, device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16): loss = torch.mean((tail(prefix)-expected)**2)/config['accumulation']
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward(); total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(tail.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True); optimizer.step()
            if step % 10 == 0:
                report.update(completed_updates=step, last_batch_loss=total_loss, elapsed_seconds=time.monotonic()-started)
                save(REPORT, report); print(json.dumps({key: report[key] for key in ('completed_updates', 'last_batch_loss', 'elapsed_seconds')}), flush=True)
            if step in config['checkpoint_steps']:
                score = metrics(y[selection], predict(tail, selected_prefix), ids[selection], yscale)
                report['curve'].append({'step': step, 'real_selection': score}); save(REPORT, report)
                if best is None or score['normalized_mse'] < best[0]: best = (score['normalized_mse'], step, copy.deepcopy(tail.state_dict()))
        after_frozen = parameter_digest((name, p) for name, p in encoder.named_parameters() if not p.requires_grad)
        if after_frozen != before_frozen: raise ValueError('Frozen prefix changed')
        tail.load_state_dict(best[2], strict=True)
        real_prediction, codec_prediction = predict(tail, selected_prefix), predict(tail, codec)
        output = BASE/'fit_v1'; output.mkdir(parents=True, exist_ok=False)
        checkpoint = output/f'step{best[1]}.pt'; torch.save(tail.state_dict(), checkpoint)
        predictions = output/'selection_predictions.npz'
        np.savez_compressed(predictions, y=y[selection], real_prediction=real_prediction, codec_prediction=codec_prediction, match_ids=ids[selection])
        report.update(status='passed_development_partial_finetune', selected_step=best[1],
            real_selection=metrics(y[selection], real_prediction, ids[selection], yscale),
            codec_selection=metrics(y[selection], codec_prediction, ids[selection], yscale),
            codec_prediction_change_mae_by_target=np.abs(codec_prediction-real_prediction).mean(axis=0).tolist(),
            frozen_prefix_sha256_before=before_frozen, frozen_prefix_sha256_after=after_frozen,
            frozen_prefix_parameters_unchanged=True, checkpoint={'path': str(checkpoint), 'sha256': sha256(checkpoint)},
            predictions={'path': str(predictions), 'sha256': sha256(predictions)},
            finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        if sha256(CONFIG) != report['registration_sha256']: raise ValueError('Registration changed during optimization')
        save(REPORT, report); print(json.dumps({'status': report['status'], 'selected_step': best[1], 'real_nmse': report['real_selection']['normalized_mse']}))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}'); save(REPORT, report); raise


if __name__ == '__main__': main()
