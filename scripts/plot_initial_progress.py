#!/usr/bin/env python3
"""Export two compact figures from completed audited decoding reports; no fitting."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'development': 'results/development_probes_v3/development_report.json',
    'probe_audit': 'results/development_probes_v3/probe_audit.json',
    'selection': 'results/development_probes_v3/development_selection.json',
    'development_capture': 'results/development_capture_audit.json',
    'development_manifest': 'results/development_capture_manifest.json',
    'fresh': 'results/fresh_confirmation_v3/confirmation.json',
    'fresh_audit': 'results/fresh_confirmation_v3/confirmation_audit.json',
    'fresh_capture': 'results/fresh_confirmation_v3/capture_audit.json',
    'registration': 'configs/fresh_evaluation_v3.json',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_sources():
    hashes = {key: sha(ROOT / path) for key, path in SOURCES.items()}
    j = {key: json.loads((ROOT / path).read_text()) for key, path in SOURCES.items()}
    d, a, f, fa, reg = (j[x] for x in ('development', 'probe_audit', 'fresh', 'fresh_audit', 'registration'))
    require(d['status'] == 'passed_development_analysis' and a['status'] == 'passed'
            and a['development_report_sha256'] == hashes['development']
            and a['development_selection_sha256'] == hashes['selection']
            and a['capture_audit_sha256'] == hashes['development_capture']
            and a['capture_manifest_sha256'] == hashes['development_manifest'], 'Development gates differ')
    require(j['development_capture']['status'] == 'passed'
            and j['development_capture']['all_layers_complete']
            and d['provenance']['match_counts'] == {'discovery': 31, 'selection': 11}
            and not d['confirmation_used'] and not d['models_refitted_with_selection'], 'Development scope differs')
    require(f['status'] == 'passed_fresh_observational_confirmation' and fa['status'] == 'passed'
            and fa['confirmation_sha256'] == hashes['fresh'] and f['n_matches'] == fa['n_matches'] == 23
            and f['registered_choices_sha256'] == fa['registered_choices_sha256'] == hashes['registration']
            and fa['prior_probe_audit_sha256'] == hashes['probe_audit']
            and f['capture_audit_sha256'] == fa['capture_audit_sha256'] == hashes['fresh_capture']
            and j['fresh_capture']['status'] == 'passed', 'Fresh confirmation gates differ')
    require(fa['all_published_metrics_and_intervals_recomputed'] and fa['paired_whole_match_bootstrap_checked']
            and not f['models_refitted'] and not f['new_choices_from_confirmation']
            and reg['status'] == 'frozen_before_fresh_capture'
            and reg['development_report_sha256'] == hashes['development']
            and reg['development_selection_sha256'] == f['development_selection_sha256'] == hashes['selection'],
            'Frozen choices or metrics audit differ')
    return j, hashes


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with path.open(newline='') as f:
        actual = list(csv.DictReader(f))
    require(actual == [{k: str(v) for k, v in row.items()} for row in rows], 'CSV roundtrip differs')


def main():
    code_sha = sha(__file__)
    source, hashes = load_sources()
    destination, data = ROOT / 'figures/progress', ROOT / 'results/progress'
    destination.mkdir(parents=True, exist_ok=True);data.mkdir(parents=True, exist_ok=True)
    require(not (data / 'figure_manifest.json').exists(), 'Preserve prior completed figure export')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#30343b', 'text.color': '#30343b',
                         'xtick.color': '#424750', 'ytick.color': '#424750', 'pdf.fonttype': 42})
    layers = []
    for row in source['development']['stage1']:
        group = row['targets']['role12']['groups']['position']
        model, metrics = group['main'], group['main']['selection']
        require(metrics['n_matches'] == 11 and len(group['target_names']) == 6, 'Layer metric scope differs')
        layers.append(dict(readout=row['name'], kind=row['kind'], summary=row['feature_spec']['summary'],
                           site_index='' if row['feature_spec']['site_index'] is None else row['feature_spec']['site_index'],
                           temporal=row['feature_spec']['temporal'], feature_width=row['feature_width'],
                           alpha=model['alpha'], selection_matches=11, target_family='role12/position',
                           normalized_mse=metrics['normalized_mse']))
    require(len(layers) == 37 and all(np.isfinite(r['normalized_mse']) for r in layers), 'Invalid layer coverage')
    colors = {'mean': '#2369a0', 'spatial': '#d47727'}
    fig, ax = plt.subplots(figsize=(8, 4.45))
    for summary, label in [('mean', 'Residual mean'), ('spatial', 'Residual spatial bins')]:
        rows = sorted([r for r in layers if r['kind'] == 'residual' and r['summary'] == summary], key=lambda r: r['site_index'])
        require([r['site_index'] for r in rows] == list(range(17)) and all(r['temporal'] == 'current' for r in rows), 'Missing current layer')
        ax.plot(range(17), [r['normalized_mse'] for r in rows], '.-', lw=2, ms=7, color=colors[summary], label=label)
    for name, label, color, style in [('codec_mean_X', 'Codec mean', '#888888', ':'),
                                     ('codec_spatial_X', 'Codec spatial bins', '#648d77', '--'),
                                     ('RGB_X', 'RGB baseline', '#a78ca6', '-.')]:
        rows = [r for r in layers if r['readout'] == name]
        require(len(rows) == 1, 'Missing baseline')
        ax.axhline(rows[0]['normalized_mse'], color=color, linestyle=style, lw=1.6, label=label)
    ax.set(xlim=(-.35, 16.35), ylim=(0, 1.15), xticks=range(17),
           xticklabels=['Input'] + [str(x) for x in range(16)], xlabel='Residual site: input, then block output',
           ylabel='Position error (normalized MSE; lower is better)')
    ax.grid(axis='y', alpha=.15)
    ax.set_title('Position decoding across layers', loc='left', fontweight='bold', pad=40)
    fig.text(.105, .883, '31 training matches · 11 selection matches · ego and ball-relative world XYZ', fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.10), ncol=3, frameon=False, fontsize=8.5,
              columnspacing=1.1, handlelength=2)
    fig.text(.105, .032, 'Observed target frames are present. These selection results guided the frozen probe choices.', fontsize=8.5)
    fig.subplots_adjust(left=.105, right=.975, bottom=.18, top=.78)
    for suffix in ('png', 'pdf'):
        fig.savefig(destination / ('layer_position_decoding.' + suffix), dpi=180)
    plt.close(fig)
    fresh_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.15), sharey=True)
    for ax, family in zip(axes, ('position', 'velocity')):
        key = source['registration']['choices']['role12'][family]['primary']
        require(source['fresh']['comparisons']['role12'][family]['choices']['primary'] == key, 'Fresh model choice differs')
        metrics = source['fresh']['metrics'][key]
        expected = [f'{entity}.{field}.{axis}' for entity in ('ego', 'ball_minus_ego')
                    for field in ('location' if family == 'position' else 'velocity',) for axis in 'xyz']
        require(metrics['target_names'] == expected, 'Fresh target ordering differs')
        m = metrics['main'];values = np.asarray(m['r2']);ci = np.asarray(m['r2_ci95'])
        require(m['n_matches'] == 23 and m['bootstrap_unit'] == 'whole_match' and m['bootstrap_replicates'] == 500
                and values.shape == (6,) and ci.shape == (6, 2) and np.isfinite(ci).all()
                and (ci[:, 0] <= values).all() and (values <= ci[:, 1]).all(), 'Invalid confidence intervals')
        for i, target in enumerate(expected):
            fresh_rows.append(dict(family=family, model_key=key, target=target, r2=values[i], ci95_low=ci[i, 0],
                                   ci95_high=ci[i, 1], n_matches=23, n_rows=m['n_examples'], bootstrap_unit=m['bootstrap_unit'],
                                   bootstrap_replicates=m['bootstrap_replicates'], bootstrap_seed=m['bootstrap_seed']))
        color = colors['mean' if family == 'position' else 'spatial']
        ax.errorbar(range(6), values, yerr=np.stack([values-ci[:, 0], ci[:, 1]-values]), fmt='o', color=color, capsize=3, ms=6, lw=1.6)
        ax.axhline(0, color='#888888', lw=1)
        ax.axvline(2.5, color='#dddddd', lw=1)
        ax.set(xlim=(-.5, 5.5), ylim=(-.12, 1.04), xticks=range(6), xticklabels=['X', 'Y', 'Z'] * 2)
        ax.set_title(family.capitalize(), loc='left', fontweight='bold', pad=24)
        subtitle = 'Block 5 · residual mean' if family == 'position' else 'Block 6 · mean + temporal difference'
        ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=8.5)
        ax.text(.25, -.17, 'Ego', ha='center', transform=ax.transAxes)
        ax.text(.75, -.17, 'Ball relative to ego', ha='center', transform=ax.transAxes)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Fresh-match decoding R²')
    fig.suptitle('Physical-state decoding on 23 fresh matches', x=.075, y=.985, ha='left', fontsize=12, fontweight='bold')
    fig.text(.075, .033, 'World axes · 95% intervals from 500 whole-match resamples · observed target frames are present', fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.985, top=.75, bottom=.23, wspace=.15)
    for suffix in ('png', 'pdf'):
        fig.savefig(destination / ('fresh_state_decoding.' + suffix), dpi=180)
    plt.close(fig)
    write_csv(data / 'layer_position_decoding.csv', layers)
    write_csv(data / 'fresh_state_decoding.csv', fresh_rows)
    require(all(sha(ROOT / SOURCES[k]) == v for k, v in hashes.items()) and sha(__file__) == code_sha, 'Sources or code changed')
    paths = [data / (name + '.csv') for name in ('layer_position_decoding', 'fresh_state_decoding')]
    paths += [destination / (name + '.' + ext) for name in ('layer_position_decoding', 'fresh_state_decoding') for ext in ('png', 'pdf')]
    report = dict(status='passed_audited_initial_progress_export', script_sha256=code_sha,
                  sources={k: {'path': SOURCES[k], 'sha256': v} for k, v in hashes.items()},
                  outputs=[{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths],
                  layer_rows=len(layers), fresh_target_rows=len(fresh_rows), models_refitted=False,
                  new_bootstrap_performed=False, source_files_unchanged=True, gpu_used=False,
                  figure1='All17 current sites, both descriptors and all3 baselines; exact role12 position selection NMSE.',
                  figure2='Frozen primary models by family, all12 ego/ball-relative coordinates; saved fresh-match R2 intervals.',
                  limits=['Contemporaneous observational annotation decoding; target frames are present.',
                          'Development selection results informed model choices. Fresh choices were frozen before evaluation.',
                          'Horizontal velocities remain weak. Decoding does not establish causal physical control.',
                          'Confidence intervals resample matches; no training-seed or population guarantee.'])
    (data / 'figure_manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'outputs': [str(p.relative_to(ROOT)) for p in paths]}))


if __name__ == '__main__':
    main()
