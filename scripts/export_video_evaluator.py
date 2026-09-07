#!/usr/bin/env python3
"""Export completed independent VideoMAE development metrics; never fit or read clips."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'results/videomae_analysis.json'
    report = json.loads(source.read_text())
    if report['status'] != 'passed_development_evaluator_analysis' or report['match_counts'] != {'discovery': 31, 'selection': 11}:
        raise ValueError('Require the completed development-only evaluator report')
    if report['independent_generated_video_measurement_ready']:
        raise ValueError('This export is only calibrated on recorded/codec-reconstructed video')
    targets = report['target_names']
    variants = ['real_selection', 'codec_selection', 'mean_baseline', 'shuffled_selection']
    if len(targets) != 12 or len(set(targets)) != 12:
        raise ValueError('Expected twelve unique ego/ball-relative targets')
    for variant in variants:
        for metric in ['mae', 'rmse', 'r2', 'normalized_mse_by_target']:
            values = report[variant][metric]
            if len(values) != 12 or not all(value is None and metric == 'r2' or isinstance(value, (float, int)) and np.isfinite(value) for value in values):
                raise ValueError('Invalid reported target metrics')
    out = ROOT/'results/videomae_exports'; out.mkdir(exist_ok=True)
    figures = ROOT/'figures'; figures.mkdir(exist_ok=True)
    table = out/'target_metrics.csv'
    with table.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'target', 'mae', 'rmse', 'r2', 'normalized_mse', 'codec_prediction_change_mae'])
        for variant in variants:
            for i, target in enumerate(targets):
                writer.writerow([variant, target, *[report[variant][metric][i] for metric in ['mae', 'rmse', 'r2', 'normalized_mse_by_target']],
                                 report['codec_prediction_change_mae_by_target'][i] if variant == 'codec_selection' else ''])
    summary = out/'summary.csv'
    with summary.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['variant', 'normalized_mse', 'selection_matches', 'view_rows'])
        for variant in variants:
            writer.writerow([variant, report[variant]['normalized_mse'], 11, 352])
    names = [target.replace('ball_minus_ego', 'Ball−ego').replace('ego', 'Ego').replace('.location.', ' pos ').replace('.velocity.', ' vel ') for target in targets]
    styles = [('real_selection', 'Real video', '#1565c0', 'o'),
              ('codec_selection', 'Codec reconstruction', '#d95f02', 's'),
              ('shuffled_selection', 'Whole-match shuffled labels', '#777777', 'x')]
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    for variant, label, color, marker in styles:
        ax.plot(np.arange(12), [np.nan if value is None else value for value in report[variant]['r2']],
                marker=marker, color=color, label=label, linewidth=1.7, markersize=5)
    ax.axhline(0, color='black', linewidth=.7)
    for edge in [2.5, 5.5, 8.5]: ax.axvline(edge, color='#cccccc', linewidth=.8)
    ax.set_xticks(np.arange(12), names, rotation=42, ha='right')
    ax.set_ylabel('Selection R² (equal match weights)')
    ax.set_title('Frozen VideoMAE endpoint state decoding')
    ax.grid(axis='y', alpha=.2); ax.legend(loc='upper left', ncol=3, fontsize=9)
    ax.set_ylim(min(-.10, *[min(report[v]['r2'])-.04 for v in variants]), max(.60, *[max(report[v]['r2'])+.08 for v in variants]))
    fig.text(.5, .015, '11 selection matches; alpha selected here. Target-frame pixels are present. Codec transfer is not generated-rollout validation.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, 1])
    files = [table, summary]
    for extension in ['png', 'pdf']:
        path = figures/f'videomae_state_r2.{extension}'; fig.savefig(path, dpi=180); files.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for axis, (indices, title, unit) in zip(axes, [(np.r_[0:3, 6:9], 'Position', 'Unreal units'), (np.r_[3:6, 9:12], 'Velocity', 'Unreal units / second')]):
        for offset, variant, label, color in [(-.26, 'real_selection', 'Real video', '#1565c0'), (0, 'codec_selection', 'Codec reconstruction', '#d95f02'), (.26, 'mean_baseline', 'Discovery mean', '#999999')]:
            axis.bar(np.arange(6)+offset, np.asarray(report[variant]['mae'])[indices], width=.25, label=label, color=color)
        axis.set_xticks(np.arange(6), [names[i] for i in indices], rotation=35, ha='right')
        axis.set_ylabel(f'MAE ({unit})'); axis.set_title(title); axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Independent evaluator errors remain substantial in physical units')
    fig.text(.5, .015, 'Same 352 views from 11 selection matches; reconstruction labels inherit the source annotations. No confidence or physical-control claim.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .055, 1, .94])
    for extension in ['png', 'pdf']:
        path = figures/f'videomae_state_mae.{extension}'; fig.savefig(path, dpi=180); files.append(path)
    plt.close(fig)
    manifest = {'status': 'passed_export', 'input_report_sha256': digest(source), 'script_sha256': digest(Path(__file__)),
                'scope': 'development selection only; no refit or new analysis', 'csv_target_rows': 48,
                'files': {str(path.relative_to(ROOT)): digest(path) for path in files}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
