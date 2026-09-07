#!/usr/bin/env python3
"""Fixed three-seed development MLP comparison on unchanged VideoMAE captures."""
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from analyze_video_evaluator import audit_captures
from mira_interp.development import metrics
from mira_interp.model_loading import sha256
from mira_interp.video_evaluator_training import SpatialStateHead


def save(path, report):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def prediction(model, x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model.predict(part) for part in x.split(64)]).numpy().astype(np.float64)


def main():
    torch.set_num_threads(4)
    out = Path('/data2/ishaangp/mira-interp/video_evaluator_v1/head_mlp')
    report_path = ROOT/'results/videomae_mlp.json'
    if out.exists() or report_path.exists():
        raise ValueError('Use a new version rather than overwrite head experiments')
    source = ROOT/'results/videomae_analysis.json'
    baseline = json.loads(source.read_text())
    if baseline['status'] != 'passed_development_evaluator_analysis':
        raise ValueError('Frozen baseline and independent capture audit must pass')
    data, audited = audit_captures(ROOT/'results/videomae_capture.json')
    model_path = Path(baseline['models']['main']['path'])
    if sha256(model_path) != baseline['models']['main']['sha256']:
        raise ValueError('Discovery normalizer artifact changed')
    with np.load(model_path, allow_pickle=False) as model:
        normalizers = [model[name] for name in ('x_mean', 'x_scale', 'y_mean', 'y_scale')]
    train, selection = data['split'] == 'discovery', data['split'] == 'selection'
    xtrain = torch.from_numpy(data['X'][train]); xselect = torch.from_numpy(data['X'][selection]); xcodec = torch.from_numpy(data['X_codec_reconstruction'])
    ytrain = torch.tensor((data['y'][train]-normalizers[2])/normalizers[3], dtype=torch.float32)
    yselect, ids = data['y'][selection], data['match_ids'][selection]
    if len(xtrain) != 992 or len(xselect) != 352:
        raise ValueError('Expected unchanged992/352 endpoint observations')
    report = {'status': 'running', 'scope': 'original31/11 development matches only; no confirmation',
        'baseline_report_sha256': sha256(source), 'capture_report_sha256': audited['capture_report_sha256'],
        'code_sha256': {name: sha256(ROOT/name) for name in ('scripts/train_video_evaluator_head.py', 'src/mira_interp/video_evaluator_training.py')},
        'target_names': baseline['target_names'], 'seeds': [0, 1, 2], 'hidden_width': 256, 'dropout': .1,
        'updates_per_seed': 1000, 'batch_size': 64, 'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .01,
        'gradient_clip_norm': 1., 'checkpoint_steps': [100, 250, 500, 1000], 'normalization': 'unchanged discovery-only frozen ridge normalizers',
        'selection_rule': 'minimum equal-match real-selection12-target normalized MSE; choose step per seed then best seed',
        'codec_used_for_selection_or_training': False, 'independent_generated_video_measurement_ready': False,
        'training_weights': 'uniform rows, equivalent to equal match weights because every match has32 rows', 'runs': []}
    out.mkdir(parents=True); save(report_path, report)
    for seed in report['seeds']:
        started = time.monotonic(); torch.manual_seed(seed)
        model = SpatialStateHead(*normalizers); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        rng = np.random.default_rng(seed); curve = []; best = None
        for step in range(1, 1001):
            model.train(); batch = rng.integers(len(xtrain), size=64)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xtrain[batch])-ytrain[batch])**2)
            if not torch.isfinite(loss): raise ValueError('Nonfinite MLP loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            if step in report['checkpoint_steps']:
                score = metrics(yselect, prediction(model, xselect), ids, normalizers[3])
                curve.append({'step': step, 'last_batch_loss': float(loss.detach()), 'real_selection': score})
                if best is None or score['normalized_mse'] < best[0]:
                    best = (score['normalized_mse'], step, copy.deepcopy(model.state_dict()))
                print(json.dumps({'seed': seed, 'step': step, 'selection_nmse': score['normalized_mse'], 'elapsed_seconds': time.monotonic()-started}), flush=True)
        model.load_state_dict(best[2], strict=True)
        real, codec = prediction(model, xselect), prediction(model, xcodec)
        model_file = out/f'seed{seed}_step{best[1]}.pt'; torch.save(model.state_dict(), model_file)
        predictions = out/f'seed{seed}_predictions.npz'
        np.savez_compressed(predictions, y=yselect, real_prediction=real, codec_prediction=codec, match_ids=ids)
        run = {'seed': seed, 'selected_step': best[1], 'parameters': sum(p.numel() for p in model.parameters()),
            'curve': curve, 'real_selection': metrics(yselect, real, ids, normalizers[3]),
            'codec_selection': metrics(yselect, codec, ids, normalizers[3]),
            'codec_prediction_change_mae_by_target': np.abs(codec-real).mean(axis=0).tolist(),
            'checkpoint': {'path': str(model_file), 'sha256': sha256(model_file)},
            'predictions': {'path': str(predictions), 'sha256': sha256(predictions)}, 'elapsed_seconds': time.monotonic()-started}
        report['runs'].append(run); save(report_path, report)
    best = min(report['runs'], key=lambda row: row['real_selection']['normalized_mse'])
    report.update(status='passed_development_mlp_analysis', selected_seed=best['seed'], selected_step=best['selected_step'],
                  baseline_real_nmse=baseline['real_selection']['normalized_mse'], baseline_codec_nmse=baseline['codec_selection']['normalized_mse'])
    if sha256(source) != report['baseline_report_sha256']:
        raise ValueError('Frozen baseline report changed during head training')
    save(report_path, report); print(json.dumps({'status': report['status'], 'selected_seed': best['seed'], 'real_nmse': best['real_selection']['normalized_mse']}))


if __name__ == '__main__':
    main()
