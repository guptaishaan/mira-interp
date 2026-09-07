#!/usr/bin/env python3
"""Reproduce completed forward maps and visualize discovery conditional means."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '8'
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mira_interp.probes import match_weights, ridge_path
from analyze_feature_geometry import registered_context, validate_registered_args, verify_bindings, validate_descriptor


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT / 'configs/feature_development_v1.json')
    args = p.parse_args()
    report_hash = sha(args.report)
    report = json.loads(args.report.read_text())
    if report['status'] != 'passed_development_feature_analysis' or report['input_npz_sha256'] != sha(args.input):
        raise ValueError('Completed geometry and unchanged input required')
    protocol = json.loads(args.protocol.read_text())
    protocol, gate, bindings = registered_context(args.protocol, args.input,
        Path(protocol['selected_probe_path']), Path(protocol['gate_path']))
    bindings[args.report.resolve()] = report_hash
    if (report['protocol_sha256'] != sha(args.protocol) or report['code_sha256'] != protocol['code_sha256']
            or report['gate_sha256'] != protocol['gate_sha256']
            or report['selected_probe_sha256'] != protocol['selected_probe_sha256']):
        raise ValueError('Completed analysis provenance differs from the registered protocol')
    execution = SimpleNamespace(**report['execution_args'])
    validate_registered_args(execution, protocol)
    if execution.stage != 'geometry' or report['geometry'] is None:
        raise ValueError('A completed separately gated geometry stage is required')
    selected = json.loads(Path(protocol['selected_probe_path']).read_text())
    dimension = validate_descriptor(selected, gate)
    out = args.report.parent
    if any((out / name).exists() for name in ('export.json', 'geometry_maps.npz', 'geometry.png', 'geometry.pdf')):
        raise ValueError('Do not overwrite a completed export')
    g = report['geometry']; variable = g['variable']
    if (variable != execution.variable or g['heldout_interval'] != execution.heldout_interval
            or g['minimum_heading_speed'] != execution.minimum_heading_speed
            or any(model['alpha'] != execution.ridge_alpha for model in g['models'].values())):
        raise ValueError('Geometry result parameters differ from registered execution')
    with np.load(args.input, allow_pickle=False) as z:
        roles = z['split'].astype(str)
        if set(roles) != {'discovery', 'selection'}:
            raise ValueError('Confirmation arrays forbidden')
        x, ids, velocity = z['X'].astype(float), z['match_ids'], z['velocity']
        height = z['position'][:, 2]
    if x.shape != (8064, dimension) or velocity.shape != (8064, 3):
        raise ValueError('Unexpected registered descriptor or label dimensions')
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    keep = speed >= g['minimum_heading_speed'] if variable == 'heading' else np.ones(len(x), bool)
    u = (np.arctan2(velocity[:, 1], velocity[:, 0]) + np.pi) % (2 * np.pi) - np.pi if variable == 'heading' else (
        height if variable == 'height' else speed if variable == 'speed' else velocity[:, ('vx', 'vy', 'vz').index(variable)])
    x, ids, roles, u = x[keep], ids[keep], roles[keep], u[keep]
    low, high = g['heldout_interval']
    train = (roles == 'discovery') & ~((u >= low) & (u <= high))
    evaluate = (roles == 'selection') & (u >= low) & (u <= high)
    w = match_weights(ids[train])
    z = (u - g['physical_mean_fit_only']) / g['physical_scale_fit_only']
    basis = {'affine': z[:, None], 'quadratic': np.column_stack((z, z*z))}
    if variable == 'heading':
        first = np.column_stack((np.sin(u), np.cos(u)))
        basis = {'periodic_first': first, 'periodic_second': np.column_stack((first, np.sin(2*u), np.cos(2*u)))}
    saved = {'physical_mean': np.array(g['physical_mean_fit_only']), 'physical_scale': np.array(g['physical_scale_fit_only']),
             'variable': np.array(variable), 'heldout_interval': np.array([low, high])}
    checked = {}
    for name, a in basis.items():
        model = ridge_path(a[train], x[train], w, alphas=(g['models'][name]['alpha'],))[0]
        if model.fingerprint() != g['models'][name]['model_sha256']:
            raise ValueError('Forward geometry model did not reproduce')
        residual = np.mean((model.predict(a[evaluate]) - x[evaluate]) ** 2, axis=1)
        per_match = {m: float(residual[ids[evaluate] == m].mean()) for m in np.unique(ids[evaluate])}
        if max(abs(per_match[m] - g['models'][name]['per_match_raw_mse'][m]) for m in per_match) > 1e-10:
            raise ValueError('Held-value errors did not reproduce')
        for key in ('x_mean', 'x_scale', 'y_mean', 'y_scale', 'coefficient'):
            saved[f'{name}__{key}'] = getattr(model, key)
        checked[name] = model.fingerprint()

    # Display-only PCA of discovery conditional means; no raw-space metric uses it.
    lo, hi = (-np.pi, np.pi) if variable == 'heading' else np.quantile(u[roles == 'discovery'], [.01, .99])
    edges = np.linspace(lo, hi, 17)
    means, centers, counts, match_counts = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        rows = (roles == 'discovery') & (u >= a) & (u < b)
        if rows.sum() < 8 or len(np.unique(ids[rows])) < 2:
            continue
        weight = match_weights(ids[rows])
        means.append(weight @ x[rows]); centers.append(float(weight @ u[rows]))
        counts.append(int(rows.sum())); match_counts.append(len(np.unique(ids[rows])))
    means = np.asarray(means)
    if len(means) < 3:
        raise ValueError('Insufficient conditional means for display')
    center = means.mean(0)
    _, singular, vectors = np.linalg.svd(means - center, full_matrices=False)
    coords = (means - center) @ vectors[:3].T
    variance = singular**2 / np.sum(singular**2)
    saved.update(pca_mean=center, pca_components=vectors[:3], conditional_mean_values=np.array(centers),
                 conditional_means=means, conditional_mean_counts=np.array(counts))
    np.savez_compressed(out / 'geometry_maps.npz', **saved)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for i in range(min(3, coords.shape[1])):
        axes[0].plot(centers, coords[:, i], '.-', label=f'PC{i+1}: {variance[i]:.1%}')
    axes[0].axvspan(low, high, alpha=.12, color='gray', label='Held interval for forward fits')
    axes[0].set(xlabel=f'Ball {variable} (radians for heading; otherwise simulator units)', ylabel='Conditional-mean PCA coordinate', title='Discovery display only; PCA is not a manifold test')
    axes[0].legend(fontsize=8)
    names = list(g['models'])
    losses = [g['mean_baseline']['raw_activation_mse']] + [g['models'][k]['raw_activation_mse'] for k in names]
    axes[1].bar(range(3), losses, color=['gray', '#3578a3', '#cf762b'])
    axes[1].set_xticks(range(3), ['Mean baseline'] + [n.replace('_', ' ') for n in names], rotation=15)
    axes[1].set(ylabel='Raw descriptor MSE', title=f'Held values on {len(g["evaluation_match_ids"])} selection matches')
    fig.suptitle(f'Block 15 spatial descriptor: ball {variable}\nObservational variation; camera, actions and other state are not fixed')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'geometry.{ext}', dpi=180)
    plt.close(fig)
    verify_bindings(bindings)
    result = dict(status='passed', input_sha256=sha(args.input), report_sha256=report_hash,
                  protocol_sha256=sha(args.protocol), code_sha256=protocol['code_sha256'],
                  forward_models_reproduced=checked, maps_sha256=sha(out / 'geometry_maps.npz'),
                  figures={name: sha(out / name) for name in ('geometry.png', 'geometry.pdf')},
                  pca_scope='display-only discovery conditional means; includes held-interval discovery rows excluded from forward fits',
                  pca_variance_ratio=variance.tolist(), conditional_bin_rows=counts,
                  conditional_bin_match_counts=match_counts, nonlinear_geometry_established=False,
                  confounds='No nuisance adjustment; whole-view features vary with camera, other state and actions',
                  script_sha256=sha(Path(__file__)))
    (out / 'export.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'variable': variable, 'models_reproduced': len(checked)}))


if __name__ == '__main__':
    main()
