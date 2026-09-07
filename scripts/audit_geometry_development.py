#!/usr/bin/env python3
"""Audit four registered saved forward maps without fitting new coefficients."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '8'
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import numpy as np
from package_development import ROOT, read, require, sha


def weights(ids):
    counts = Counter(ids)
    return np.asarray([1 / (len(counts) * counts[value]) for value in ids])


def normalize(values, w):
    mean = np.sum(w[:, None] * values, axis=0)
    scale = np.sqrt(np.sum(w[:, None] * np.square(values - mean), axis=0))
    threshold = np.finfo(float).eps * 16 * np.maximum(1, np.max(np.abs(values), axis=0))
    return mean, np.where(scale > threshold, scale, 1.)


def close(actual, expected, name, *, rtol=1e-8, atol=1e-10):
    require(np.shape(actual) == np.shape(expected) and np.allclose(actual, expected, rtol=rtol, atol=atol), name)


def model_hash(alpha, model):
    h = hashlib.sha256(str(float(alpha)).encode())
    for key in ['x_mean', 'x_scale', 'y_mean', 'y_scale', 'coefficient']:
        a = np.ascontiguousarray(model[key], dtype='<f8')
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'configs/feature_development_v1.json')
    p.add_argument('--directory', type=Path, default=ROOT / 'results/geometry_development_v1')
    args = p.parse_args()
    out = args.directory / 'geometry_audit.json'
    require(not out.exists(), 'Prior geometry audit is immutable')
    started = time.monotonic()
    protocol = read(args.protocol)
    require(protocol['status'] == 'registered_before_geometry_and_sparse_research'
            and protocol['geometry_order'] == ['height', 'vz', 'speed', 'heading'], 'Wrong registered comparison set')
    frozen = {args.protocol: sha(args.protocol), Path(__file__): sha(Path(__file__)),
              ROOT / 'scripts/package_development.py': sha(ROOT / 'scripts/package_development.py')}
    for key in ['input', 'selected_probe', 'gate']:
        path = Path(protocol[key + '_path'])
        require(sha(path) == protocol[key + '_sha256'], 'Registered feature input changed')
        frozen[path] = protocol[key + '_sha256']
    for relative, expected in protocol['code_sha256'].items():
        require(sha(ROOT / relative) == expected, 'Registered analysis/export code changed')
        frozen[ROOT / relative] = expected
    with np.load(protocol['input_path'], allow_pickle=False) as saved:
        roles = saved['split'].astype(str)
        require(set(roles) == {'discovery', 'selection'}, 'Confirmation rows refused before labels')
        x, ids, velocity, position = saved['X'].astype(float), saved['match_ids'], saved['velocity'], saved['position']
    require(x.shape == (8064, 1536) and Counter(roles) == {'discovery': 5952, 'selection': 2112}, 'Wrong descriptor rows')
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    physical = {'height': position[:, 2], 'vz': velocity[:, 2], 'speed': speed,
                'heading': (np.arctan2(velocity[:, 1], velocity[:, 0]) + np.pi) % (2 * np.pi) - np.pi}
    audit = {'status': 'running', 'protocol_sha256': sha(args.protocol), 'input_npz_sha256': protocol['input_sha256'],
             'selected_probe_sha256': protocol['selected_probe_sha256'], 'scope': 'development_only',
             'coefficients_refitted': False, 'confirmation_arrays_read': False, 'comparisons': []}
    for variable in protocol['geometry_order']:
        folder = args.directory / variable
        report_path, export_path, maps_path = [folder / name for name in ['report.json', 'export.json', 'geometry_maps.npz']]
        report, exported = read(report_path), read(export_path)
        for path in [report_path, export_path, maps_path]:
            frozen[path] = sha(path)
        require(report['status'] == 'passed_development_feature_analysis' and exported['status'] == 'passed'
                and report['protocol_sha256'] == exported['protocol_sha256'] == sha(args.protocol)
                and report['code_sha256'] == exported['code_sha256'] == protocol['code_sha256']
                and report['input_npz_sha256'] == exported['input_sha256'] == protocol['input_sha256']
                and exported['report_sha256'] == sha(report_path) and exported['maps_sha256'] == sha(maps_path), 'Completed geometry artifacts differ')
        for name, expected in exported['figures'].items():
            require(sha(folder / name) == expected, 'Exported figure bytes changed')
            frozen[folder / name] = expected
        g = report['geometry']
        require(g['variable'] == variable and g['heldout_interval'] == protocol['geometry'][variable]
                and g['minimum_heading_speed'] == (protocol['minimum_heading_speed'] if variable == 'heading' else None), 'Physical comparison changed')
        u = physical[variable]
        eligible = (speed >= protocol['minimum_heading_speed']) if variable == 'heading' else np.ones(len(x), bool)
        low, high = protocol['geometry'][variable]
        inside = (u >= low) & (u <= high)
        train = (roles == 'discovery') & eligible & ~inside
        test = (roles == 'selection') & eligible & inside
        require(int(train.sum()) == g['fit_rows'] and int(test.sum()) == g['evaluation_rows']
                and sorted(set(ids[train])) == g['fit_match_ids'] and sorted(set(ids[test])) == g['evaluation_match_ids'], 'Fit/held-match rows differ')
        w = weights(ids[train])
        physical_mean = float(np.sum(w * u[train]))
        physical_scale = float(np.sqrt(np.sum(w * (u[train] - physical_mean) ** 2)))
        close(physical_mean, g['physical_mean_fit_only'], 'Physical mean used held-out labels')
        close(physical_scale, g['physical_scale_fit_only'], 'Physical scale used held-out labels')
        z = (u - physical_mean) / physical_scale
        bases = {'affine': z[:, None], 'quadratic': np.column_stack([z, z ** 2])}
        if variable == 'heading':
            bases = {'periodic_first': np.column_stack([np.sin(u), np.cos(u)]),
                     'periodic_second': np.column_stack([np.sin(u), np.cos(u), np.sin(2 * u), np.cos(2 * u)])}
        require(list(g['models']) == list(bases), 'Unexpected geometry model family')
        match_order = sorted(set(ids[test]))
        ymean, yscale = normalize(x[train], w)
        variance = float(np.sum(w * np.square(x[train] - ymean).mean(1)))
        baseline_rows = np.square(x[test] - ymean).mean(1)
        baseline = {str(match): float(baseline_rows[ids[test] == match].mean()) for match in match_order}
        close(np.mean(list(baseline.values())), g['mean_baseline']['raw_activation_mse'], 'Held mean baseline differs')
        models = {}
        with np.load(maps_path, allow_pickle=False) as saved:
            close(saved['heldout_interval'], [low, high], 'Saved held interval differs')
            for name, basis in bases.items():
                a = protocol['ridge_alpha']
                require(g['models'][name]['alpha'] == a, 'Unregistered ridge alpha')
                model = {key: saved[f'{name}__{key}'] for key in ['x_mean', 'x_scale', 'y_mean', 'y_scale', 'coefficient']}
                require(model_hash(a, model) == g['models'][name]['model_sha256'] == exported['forward_models_reproduced'][name], 'Saved model fingerprint differs')
                xmean, xscale = normalize(basis[train], w)
                for key, expected in [('x_mean', xmean), ('x_scale', xscale), ('y_mean', ymean), ('y_scale', yscale)]:
                    close(model[key], expected, 'Non-discovery normalization: ' + key)
                normalized_x = (basis[train] - xmean) / xscale
                normalized_y = (x[train] - ymean) / yscale
                b = model['coefficient']
                rhs = normalized_x.T @ (w[:, None] * normalized_y)
                residual = normalized_x.T @ (w[:, None] * (normalized_x @ b - normalized_y)) + a * b
                relative = np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-12)
                require(np.max(relative) < 1e-6, 'Saved forward map fails independent ridge normal equations')
                predicted = (((basis[test] - xmean) / xscale) @ b) * yscale + ymean
                errors = np.square(predicted - x[test]).mean(1)
                per_match = {str(match): float(errors[ids[test] == match].mean()) for match in match_order}
                for match, error in per_match.items():
                    close(error, g['models'][name]['per_match_raw_mse'][match], 'Held-match error differs')
                loss = float(np.mean(list(per_match.values())))
                close(loss, g['models'][name]['raw_activation_mse'], 'Equal-match loss differs')
                close(loss / variance, g['models'][name]['mse_over_discovery_variance'], 'Discovery variance ratio differs')
                models[name] = {'raw_activation_mse': loss, 'gain_over_mean_baseline': float(np.mean(list(baseline.values())) - loss),
                                'normal_equation_max_relative_residual': float(relative.max()), 'per_match_raw_mse': per_match,
                                'model_sha256': model_hash(a, model)}
            # Verify all displayed discovery conditional means without recomputing PCA.
            display = (roles == 'discovery') & eligible
            edge_low, edge_high = (-np.pi, np.pi) if variable == 'heading' else np.quantile(u[display], [.01, .99])
            edges = np.linspace(edge_low, edge_high, 17)
            means, centers, counts, matches = [], [], [], []
            for left, right in zip(edges[:-1], edges[1:]):
                rows = display & (u >= left) & (u < right)
                if rows.sum() < 8 or len(set(ids[rows])) < 2:
                    continue
                weight = weights(ids[rows])
                means.append(np.sum(weight[:, None] * x[rows], axis=0))
                centers.append(float(np.sum(weight * u[rows])))
                counts.append(int(rows.sum()))
                matches.append(len(set(ids[rows])))
            close(saved['conditional_means'], means, 'PCA display conditional means differ')
            close(saved['conditional_mean_values'], centers, 'PCA display physical coordinates differ')
            require(saved['conditional_mean_counts'].tolist() == counts == exported['conditional_bin_rows']
                    and exported['conditional_bin_match_counts'] == matches, 'PCA display coverage differs')
            close(saved['pca_components'] @ saved['pca_components'].T, np.eye(3), 'PCA components are not orthonormal')
        names = list(bases)
        gains = np.asarray([models[names[0]]['per_match_raw_mse'][str(match)] - models[names[1]]['per_match_raw_mse'][str(match)] for match in match_order])
        draws = np.random.default_rng(20260907).integers(len(match_order), size=(1000, len(match_order)))
        interval = np.quantile(gains[draws].mean(1), [.025, .975]).tolist()
        close(float(gains.mean()), g['paired_mse_gain_complex_over_simple'], 'Complex-versus-simple gain differs')
        close(interval, g['paired_gain_ci95'], 'Paired match-bootstrap interval differs')
        audit['comparisons'].append({'variable': variable, 'report': {'path': str(report_path), 'sha256': sha(report_path)},
            'export': {'path': str(export_path), 'sha256': sha(export_path)}, 'maps': {'path': str(maps_path), 'sha256': sha(maps_path)},
            'figures': exported['figures'], 'heldout_interval': [low, high], 'fit_rows': int(train.sum()),
            'held_selection_rows': int(test.sum()), 'held_selection_matches': len(match_order), 'models': models,
            'mean_baseline_raw_mse': float(np.mean(list(baseline.values()))),
            'paired_mse_gain_complex_over_simple': float(gains.mean()), 'paired_gain_ci95': interval,
            'conditional_discovery_means_recomputed': True})
        print('Audited ' + variable + ': both forward maps, held errors, bootstrap and display means', flush=True)
    require(all(sha(path) == expected for path, expected in frozen.items()), 'Artifacts changed during geometry audit')
    audit.update(status='passed', elapsed_seconds=round(time.monotonic() - started, 3), models_checked=8,
                 all_four_registered_comparisons_complete=True, all_forward_normal_equations_checked=True,
                 all_held_errors_and_intervals_recomputed=True, all_conditional_means_recomputed=True,
                 audit_script_sha256=sha(Path(__file__)), physical_causality_claim=False,
                 nonlinear_geometry_established=False)
    out.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'audit': str(out), 'elapsed_seconds': audit['elapsed_seconds']}))


if __name__ == '__main__':
    main()
