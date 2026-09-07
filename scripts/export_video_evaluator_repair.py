#!/usr/bin/env python3
"""Export completed evaluator development repairs, without fitting or reading videos."""
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {name: ROOT/'results'/file for name, file in [('ridge', 'videomae_analysis.json'), ('MLP', 'videomae_mlp.json'), ('partial', 'videomae_finetune_v1.json'), ('matched', 'videomae_matched_input_calibration.json')]}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if [reports[name]['status'] for name in paths] != ['passed_development_evaluator_analysis', 'passed_development_mlp_analysis', 'passed_development_partial_finetune', 'passed_matched_evaluator_input_calibration']:
        raise ValueError('All evaluator analyses must complete before export')
    targets = reports['ridge']['target_names']
    if any(report['target_names'] != targets for report in reports.values()): raise ValueError('Target ordering changed')
    head = {domain: reports['matched'][domain]['MLP'] for domain in ('real_selection','codec_selection')}
    ridge = {domain: reports['matched'][domain]['ridge'] for domain in ('real_selection','codec_selection')}
    models = [('Frozen ridge', ridge), ('Frozen MLP', head), ('Last 2 blocks + MLP', reports['partial'])]
    for label, model in models:
        for domain in ['real_selection', 'codec_selection']:
            if model[domain]['n_rows'] != 352 or model[domain]['n_matches'] != 11: raise ValueError('Compared cohorts differ')
    out = ROOT/'results/videomae_repair_exports'; out.mkdir(exist_ok=True); figures = ROOT/'figures'
    files = []
    table = out/'target_metrics.csv'
    with table.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['model', 'domain', 'target', 'mae', 'rmse', 'r2', 'normalized_mse'])
        for label, model in models:
            for domain in ['real_selection', 'codec_selection']:
                for i, target in enumerate(targets):
                    writer.writerow([label, domain, target, *[model[domain][metric][i] for metric in ['mae', 'rmse', 'r2', 'normalized_mse_by_target']]])
    files.append(table)
    table = out/'summary.csv'
    with table.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['model', 'real_nmse', 'codec_nmse'])
        for label, model in models: writer.writerow([label, model['real_selection']['normalized_mse'], model['codec_selection']['normalized_mse']])
    files.append(table)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    x = np.arange(3)
    for offset, domain, label, color in [(-.18, 'real_selection', 'Real video', '#1565c0'), (.18, 'codec_selection', 'Codec reconstruction', '#d95f02')]:
        axes[0].bar(x+offset, [model[domain]['normalized_mse'] for _, model in models], width=.34, label=label, color=color)
    axes[0].set_xticks(x, [label.replace(' ', '\n', 1) for label, _ in models]); axes[0].set_ylabel('Selection normalized MSE (lower is better)')
    axes[0].legend(fontsize=9); axes[0].set_title('Same independent VideoMAE measurement task')
    curve = [(0, reports['partial']['initial_real_selection']['normalized_mse'])]+[(row['step'], row['real_selection']['normalized_mse']) for row in reports['partial']['curve']]
    axes[1].plot(*zip(*curve), marker='o', color='#287d3c', label='Partial fine-tuning')
    axes[1].axhline(head['real_selection']['normalized_mse'], color='#777777', linestyle='--', label='Frozen MLP')
    chosen = reports['partial']['selected_step']; score = dict(curve)[chosen]
    axes[1].scatter([chosen], [score], marker='*', s=160, color='#d95f02', zorder=4, label='Selected checkpoint')
    axes[1].set_xlabel('Optimizer updates (step 0 is unadapted BF16 computation)'); axes[1].set_ylabel('Real-selection normalized MSE'); axes[1].legend(fontsize=8)
    axes[1].set_title('Fixed 500-update training schedule')
    for axis in axes: axis.grid(axis='y', alpha=.2)
    fig.text(.5, .015, '11 selection matches; checkpoint/hyperparameter choices use this cohort. No fresh confirmation or physical-steering certification.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, 1])
    for extension in ['png', 'pdf']:
        path = figures/f'videomae_repair_summary.{extension}'; fig.savefig(path, dpi=180); files.append(path)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = ['#1565c0', '#d95f02', '#287d3c']
    for (label, model), color in zip(models, colors):
        values = [np.nan if v is None else v for v in model['real_selection']['r2']]
        ax.plot(np.arange(12), values, marker='o', label=label, color=color)
    names = [t.replace('ball_minus_ego', 'Ball−ego').replace('ego', 'Ego').replace('.location.', ' pos ').replace('.velocity.', ' vel ') for t in targets]
    ax.set_xticks(np.arange(12), names, rotation=40, ha='right'); ax.set_ylabel('Real-selection R² (equal match weights)')
    for border in [2.5, 5.5, 8.5]: ax.axvline(border, color='#cccccc', lw=.8)
    ax.axhline(0, color='black', lw=.7); ax.grid(axis='y', alpha=.2); ax.legend(ncol=3, fontsize=9)
    ax.set_title('Evaluator adaptation by state coordinate')
    fig.text(.5, .015, '12 contemporaneous state coordinates from 352 views. Source-frame pixels are present; world-axis ball-relative velocity is not camera velocity.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, 1])
    for extension in ['png', 'pdf']:
        path = figures/f'videomae_repair_targets.{extension}'; fig.savefig(path, dpi=180); files.append(path)
    plt.close(fig)
    manifest = {'status': 'passed_export', 'input_sha256': {name: sha(path) for name, path in paths.items()}, 'script_sha256': sha(Path(__file__)),
                'target_metric_rows': 72, 'scope': 'development-only reported metrics; no refitting', 'files': {str(path.relative_to(ROOT)): sha(path) for path in files}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n'); print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
