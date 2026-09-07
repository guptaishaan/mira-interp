#!/usr/bin/env python3
"""Independent CPU audit of frozen dictionaries and descriptor retention; no fits."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '8'
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import numpy as np
import torch
from package_development import ROOT, read, require, sha

PROTOCOL = 'a13f51de39a64eaeb08ed48449913bee72b5599a02e1059e94cde2b2ba80aae5'
# Fixed before reconstruction: allow FP32 CPU/GPU GEMM and sparse-ranking roundoff.
METRIC_RTOL, METRIC_ATOL = 1e-4, 1e-6


def weights(ids):
    counts = Counter(ids)
    return np.asarray([1. / (len(counts) * counts[item]) for item in ids])


def close(a, b, name, *, rtol=1e-10, atol=1e-12):
    require(np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=rtol, atol=atol), name)


def fingerprint(alpha, arrays):
    h = hashlib.sha256(str(float(alpha)).encode())
    for key in ('x_mean', 'x_scale', 'y_mean', 'y_scale', 'coefficient'):
        value = np.ascontiguousarray(arrays[key], dtype='<f8')
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def predict(x, model):
    return ((x - model['x_mean']) / model['x_scale']) @ model['coefficient'] * model['y_scale'] + model['y_mean']


def edges(data, role):
    groups = {}
    for i in np.flatnonzero(data['split'] == role):
        key = tuple(str(data[key][i]) for key in ('match_ids', 'clip_ids', 'entity_ids', 'view_index'))
        groups.setdefault(key, []).append(i)
    result = []
    for group in groups.values():
        group.sort(key=lambda i: data['frame_index'][i])
        for a, b in zip(group[:-1], group[1:]):
            if data['frame_index'][b] - data['frame_index'][a] == 1 and abs(data['timestamps'][b] - data['timestamps'][a] - .1) <= .01:
                result.append((a, b))
    return np.asarray(result, dtype=np.int64).reshape(-1, 2)


def numpy_reconstruct(x, state, kind, block, active):
    """Independent explicit affine, top-group selection and linear synthesis."""
    pre = x @ state['encoder.weight'].T + state['encoder.bias']
    if kind == 'relu':
        pre = np.maximum(pre, 0)
    grouped = pre.reshape(len(x), -1, block)
    scores = np.sqrt(np.sum(grouped * grouped, axis=2))
    count = active // block
    selected = np.argpartition(scores, -count, axis=1)[:, -count:]
    mask = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(mask, selected, True, axis=1)
    code = (grouped * mask[:, :, None]).reshape(pre.shape)
    return code @ state['decoder'], code


def equal_match_rows(values, ids):
    per_match = {str(group): float(np.mean(values[ids == group])) for group in sorted(set(ids))}
    return {'mean': float(np.mean(list(per_match.values()))), 'per_match': per_match}


def matrix_metrics(values, ids):
    by_match = np.asarray([values[ids == group].mean(axis=0) for group in sorted(set(ids))])
    return by_match.mean(axis=0).tolist()


def geometry_context(data, protocol, geometry_audit, bind):
    x = data['X'].astype(np.float64)
    speed = np.hypot(data['velocity'][:, 0], data['velocity'][:, 1])
    physical = {'height': data['position'][:, 2], 'vz': data['velocity'][:, 2], 'speed': speed,
                'heading': (np.arctan2(data['velocity'][:, 1], data['velocity'][:, 0]) + np.pi) % (2*np.pi) - np.pi}
    contexts = {}
    for comparison in geometry_audit['comparisons']:
        variable = comparison['variable']
        for key in ('report', 'export', 'maps'):
            bind(Path(comparison[key]['path']), comparison[key]['sha256'])
        report = read(comparison['report']['path'])['geometry']
        with np.load(comparison['maps']['path'], allow_pickle=False) as saved:
            maps = {key: saved[key] for key in saved.files}
        value = physical[variable]
        eligible = speed >= protocol['minimum_heading_speed'] if variable == 'heading' else np.ones(len(x), bool)
        discovery = (data['split'] == 'discovery') & eligible
        lo, hi = protocol['geometry'][variable]
        held = (data['split'] == 'selection') & eligible & (value >= lo) & (value <= hi)
        edge_lo, edge_hi = (-np.pi, np.pi) if variable == 'heading' else np.quantile(value[discovery], [.01, .99])
        boundaries = np.linspace(edge_lo, edge_hi, 17)
        bins = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            rows = np.flatnonzero(discovery & (value >= left) & (value < right))
            if len(rows) >= 8 and len(set(data['match_ids'][rows])) >= 2:
                bins.append(rows)
        native = np.asarray([np.sum(weights(data['match_ids'][rows])[:, None] * x[rows], axis=0) for rows in bins])
        close(native, maps['conditional_means'], 'Original fixed-bin conditional means differ')
        train = (data['split'] == 'discovery') & eligible & ~((value >= lo) & (value <= hi))
        w = weights(data['match_ids'][train])
        physical_mean = float(w @ value[train])
        physical_scale = float(np.sqrt(w @ np.square(value[train]-physical_mean)))
        require(str(maps['variable']) == variable, 'Saved physical variable differs')
        close(maps['physical_mean'], physical_mean, 'Saved physical mean differs')
        close(maps['physical_scale'], physical_scale, 'Saved physical scale differs')
        close(maps['pca_mean'], native.mean(0), 'Display PCA center differs')
        z = (value - physical_mean) / physical_scale
        basis = {'affine': z[:, None], 'quadratic': np.column_stack((z, z*z))}
        if variable == 'heading':
            basis = {'periodic_first': np.column_stack((np.sin(value), np.cos(value))),
                     'periodic_second': np.column_stack((np.sin(value), np.cos(value), np.sin(2*value), np.cos(2*value)))}
        predictions = {}
        for name, features in basis.items():
            model = {key: maps[f'{name}__{key}'] for key in ('x_mean', 'x_scale', 'y_mean', 'y_scale', 'coefficient')}
            require(fingerprint(protocol['ridge_alpha'], model) == report['models'][name]['model_sha256'], 'Geometry model changed')
            predictions[name] = predict(features[held], model)
            error = equal_match_rows(np.square(predictions[name] - x[held]).mean(1), data['match_ids'][held])
            close(error['mean'], report['models'][name]['raw_activation_mse'], 'Frozen geometry native error differs')
        contexts[variable] = dict(bins=bins, native=native, held=held, predictions=predictions,
                                  native_held_errors={name: report['models'][name]['raw_activation_mse'] for name in predictions},
                                  pca=maps['pca_components'], boundaries=boundaries,
                                  physical_centers=maps['conditional_mean_values'])
    require(list(contexts) == protocol['geometry_order'], 'Missing geometry comparison')
    return contexts


def retention(x, reconstruction, data, contexts, probe, native_probe):
    pred = predict(reconstruction, probe)
    probe_report = {}
    for role in ('discovery', 'selection'):
        mask = data['split'] == role
        ids = data['match_ids'][mask]
        drift = np.square((pred[mask] - native_probe[mask]) / probe['y_scale'])
        true_native = np.square((native_probe[mask, :3] - data['position'][mask]) / probe['y_scale'][:3])
        true_recon = np.square((pred[mask, :3] - data['position'][mask]) / probe['y_scale'][:3])
        probe_report[role] = {'standardized_prediction_drift_per_target': matrix_metrics(drift, ids),
                            'standardized_prediction_drift_all15': equal_match_rows(drift.mean(1), ids),
                            'native_ball_xyz_normalized_mse': matrix_metrics(true_native, ids),
                            'reconstructed_ball_xyz_normalized_mse': matrix_metrics(true_recon, ids)}
    geometry, arrays = {}, {}
    for variable, context in contexts.items():
        means = np.asarray([np.sum(weights(data['match_ids'][rows])[:, None] * reconstruction[rows], axis=0) for rows in context['bins']])
        original = context['native']
        a, b = original-original.mean(0), means-means.mean(0)
        gram_a, gram_b = a @ a.T, b @ b.T
        upper = np.triu_indices(len(means), 1)
        da = np.maximum(0, np.diag(gram_a)[:, None] + np.diag(gram_a)[None, :] - 2*gram_a)[upper]
        db = np.maximum(0, np.diag(gram_b)[:, None] + np.diag(gram_b)[None, :] - 2*gram_b)[upper]
        held = context['held']
        forward_errors = {}
        for name, prediction in context['predictions'].items():
            forward_errors[name] = {'native_raw_mse': context['native_held_errors'][name],
                                   'reconstructed_raw_mse': equal_match_rows(np.square(prediction-reconstruction[held]).mean(1), data['match_ids'][held])}
        geometry[variable] = {'bin_count': len(means), 'conditional_mean_raw_mse': float(np.square(means-original).mean()),
            'centered_conditional_mean_relative_l2': float(np.linalg.norm(b-a)/np.linalg.norm(a)),
            'centered_gram_relative_frobenius': float(np.linalg.norm(gram_b-gram_a)/np.linalg.norm(gram_a)),
            'pairwise_squared_distance_relative_l2': float(np.linalg.norm(db-da)/np.linalg.norm(da)),
            'pairwise_squared_distance_correlation': float(np.corrcoef(da, db)[0, 1]) if np.std(db) > 0 and np.std(da) > 0 else None,
            'reconstructed_conditional_means_collapsed': bool(np.linalg.norm(b) == 0),
            'reconstructed_over_original_centered_variance': float(np.square(b).sum()/np.square(a).sum()),
            'frozen_physical_map_held_errors': forward_errors,
            'held_reconstruction_raw_mse': equal_match_rows(np.square(reconstruction[held]-x[held]).mean(1), data['match_ids'][held])}
        arrays[variable + '__conditional_means'] = means.astype(np.float32)
        arrays[variable + '__fixed_pca'] = ((means - original.mean(0)) @ context['pca'].T).astype(np.float32)
    return probe_report, geometry, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/feature_development_v1.json')
    p.add_argument('--suite', type=Path, default=ROOT/'results/feature_development_v1/suite_status.json')
    p.add_argument('--geometry-audit', type=Path, default=ROOT/'results/geometry_development_v1/geometry_audit.json')
    p.add_argument('--output', type=Path, default=ROOT/'results/feature_development_v1/sparse_audit.json')
    args = p.parse_args()
    require(not args.output.exists(), 'Prior sparse audit is immutable')
    array_path = args.output.with_name('retention_arrays.npz')
    require(not array_path.exists(), 'Prior retention arrays are immutable')
    started = time.monotonic(); torch.set_num_threads(8)
    frozen = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); actual = sha(path)
        require(expected is None or actual == expected, 'Input hash changed: ' + str(path))
        frozen[path] = actual
        return {'path': str(path), 'sha256': actual}
    bind(Path(__file__)); bind(ROOT/'scripts/package_development.py')
    bind(args.protocol, PROTOCOL); protocol = read(args.protocol)
    suite_info = bind(args.suite); suite = read(args.suite)
    geometry_info = bind(args.geometry_audit, suite['geometry_audit_sha256']); geometry = read(args.geometry_audit)
    require(suite['status'] == geometry['status'] == 'passed' and suite['completed_models'] == 30
            and suite['protocol_sha256'] == geometry['protocol_sha256'] == PROTOCOL, 'Earlier stages have not passed')
    for key in ('input', 'selected_probe', 'gate'):
        bind(protocol[key+'_path'], protocol[key+'_sha256'])
    for relative, expected in protocol['code_sha256'].items():
        bind(ROOT/relative, expected)
    input_audit_path = Path(protocol['gate_path']).with_name('input_audit.json')
    input_audit = read(input_audit_path); gate = read(protocol['gate_path'])
    bind(input_audit_path, gate['input_audit_sha256'])
    require(input_audit['status'] == 'passed_feature_input_export' and input_audit['all_source_label_joins_exact']
            and input_audit['input_npz_sha256'] == protocol['input_sha256']
            and input_audit['selected_probe_sha256'] == protocol['selected_probe_sha256'], 'Feature input gate differs')
    selected = read(protocol['selected_probe_path'])
    for key in ('probe_audit', 'causal_audit'):
        bind(selected[key]['path'], selected[key]['sha256'])
        require(read(selected[key]['path'])['status'] == 'passed', 'Selected probe/causal gate failed')
    selection_path = ROOT/'results/development_probes_v3/development_selection.json'
    archive_path = selection_path.with_name('development_models.npz')
    bind(selection_path, selected['development_selection_sha256']); bind(archive_path, selected['probe_archive_sha256'])
    selection = read(selection_path)
    matches = [(i, m) for i, m in enumerate(selection['models']) if m['name'] == selected['probe_name']]
    require(len(matches) == 1 and matches[0][1] == selected['selected_probe_record'], 'Selected probe differs')
    with np.load(archive_path, allow_pickle=False) as saved:
        prefix = f'site{matches[0][0]}_main_'
        probe = {key: saved[prefix+key] for key in ('x_mean', 'x_scale', 'y_mean', 'y_scale', 'coefficient')}
    require(fingerprint(matches[0][1]['selected_alpha'], probe) == selected['model_sha256'], 'Frozen probe fingerprint differs')
    require(probe['coefficient'].shape == (1536, 15), 'Wrong selected probe width/targets')
    with np.load(protocol['input_path'], allow_pickle=False) as saved:
        roles = saved['split'].astype(str)
        require(set(roles) == {'discovery', 'selection'}, 'Confirmation refused before physical labels')
        data = {key: saved[key] for key in saved.files}
    data['split'] = roles
    x = data['X'].astype(np.float32); ids = data['match_ids']
    require(x.shape == (8064, 1536) and Counter(roles) == {'discovery':5952, 'selection':2112}
            and all(value == 192 for value in Counter(ids).values()), 'Wrong registered descriptor rows')
    require(all(np.isfinite(value).all() for value in data.values() if np.issubdtype(value.dtype, np.number)), 'Nonfinite input')
    train = roles == 'discovery'; w = weights(ids[train])
    mean = np.sum(x[train].astype(float)*w[:, None], axis=0)
    scale = np.full(1536, np.sqrt(np.sum(np.mean(np.square(x[train]-mean), axis=1)*w)))
    normal_hash = hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()
    normalized = ((x-mean)/scale).astype(np.float32)
    true_edges = {role: edges(data, role) for role in ('discovery', 'selection')}
    require({role:len(value) for role,value in true_edges.items()} == {'discovery':4960, 'selection':1760}, 'Temporal identity rows differ')
    contexts = geometry_context(data, protocol, geometry, bind)
    native_probe = predict(x.astype(float), probe)
    arrays = {'native_probe_predictions': native_probe.astype(np.float32)}
    for variable, context in contexts.items():
        arrays[variable+'__native_conditional_means'] = context['native'].astype(np.float32)
        arrays[variable+'__physical_bin_centers'] = context['physical_centers']
        arrays[variable+'__bin_boundaries'] = context['boundaries']
    audit = {'status':'running', 'scope':'development_only', 'protocol_sha256':PROTOCOL,
             'saved_geometry_physical_metadata_independently_recomputed':True,
             'input_npz_sha256':protocol['input_sha256'], 'selected_probe_sha256':protocol['selected_probe_sha256'],
             'input_audit_sha256':sha(input_audit_path), 'geometry_audit':geometry_info, 'training_suite':suite_info,
             'feature_reports':[], 'checkpoints':[], 'models':[], 'confirmation_arrays_read':False,
             'training_or_probe_refits':False, 'normalizer_sha256':normal_hash,
             'comparison_tolerance':{'rtol':METRIC_RTOL,'atol':METRIC_ATOL,'reason':'FP32 CPU/GPU matrix arithmetic and top-group ranking'},
             'retention_scope':'Frozen probe and physical maps; fixed original discovery bins. Lower map error after reconstruction can reflect collapse, not preserved geometry.'}
    require([entry['seed'] for entry in suite['reports']] == [0,1,2], 'Expected three registered seeds')
    for entry in suite['reports']:
        report_path = Path(entry['path']); audit['feature_reports'].append(bind(report_path, entry['sha256']))
        report = read(report_path); seed = entry['seed']
        require(report['status'] == 'passed_development_feature_analysis' and report['protocol_sha256'] == PROTOCOL
                and report['input_npz_sha256'] == protocol['input_sha256'] and report['selected_probe_sha256'] == protocol['selected_probe_sha256']
                and report['code_sha256'] == protocol['code_sha256'] and report['gate_sha256'] == protocol['gate_sha256']
                and report['descriptor_identity'] == 'spatial/block_15_output' and report['confirmation_rows_loaded'] is False,
                'Training report provenance mismatch')
        expected = []
        for active in (32,64):
            expected.extend([(f'{kind}_active{active}.pt', kind, active, 0., 'adjacent') for kind in ('relu','signed','block')])
            expected.extend([(f'block_active{active}_temporal{suffix}.pt', 'block', active, .1, control)
                             for suffix,control in [('', 'adjacent'),('_shuffled','shuffled_within_match')]])
        require(set(report['outputs']) == {v[0] for v in expected} and len(report['sparse']) == 10, 'Missing/extra dictionary')
        for model_index, (filename, kind, active, temporal, control) in enumerate(expected):
            path = report_path.parent/filename; output = report['outputs'][filename]
            audit['checkpoints'].append(bind(path, output['sha256']))
            require(path.stat().st_size == output['bytes'], 'Checkpoint size differs')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            require(set(checkpoint) == {'state_dict','normalizer','report'}, 'Unexpected checkpoint fields')
            saved = checkpoint['report']; require(saved == report['sparse'][model_index], 'Checkpoint/report training metrics differ')
            block = 8 if kind == 'block' else 1
            checks = {'kind':kind,'group_size':block,'latent_scalar_width':3072,'input_dimension':1536,
                      'active_scalar_budget':active,'trainable_parameters':9440256,'steps':1000,'batch_size':128,
                      'learning_rate':.001,'seed':seed,'temporal_weight':temporal,'temporal_control':control,
                      'temporal_edges':4960 if temporal else 0,'normalizer_sha256':normal_hash,
                      'selection_used_for_updates':False,'tracked_entity_claim':False,'temporal_identity_scope':'view_identity_only'}
            require(all(saved.get(key) == value for key,value in checks.items()), 'Unregistered training budget/normalizer/identity')
            require([v['step'] for v in saved['loss_endpoints']] == [1]+list(range(100,1001,100)), 'Incomplete updates')
            require(all(np.isfinite(v['reconstruction']) and np.isfinite(v['temporal_surrogate']) for v in saved['loss_endpoints']), 'Nonfinite training')
            require(set(checkpoint['normalizer']) == {'mean','scale'}, 'Unexpected normalizer')
            for key, value in [('mean',mean),('scale',scale)]:
                tensor = checkpoint['normalizer'][key]
                require(tensor.dtype == torch.float64 and np.array_equal(tensor.numpy(),value), 'Discovery normalizer differs exactly')
            state = checkpoint['state_dict']
            shapes = {'decoder':(3072,1536),'encoder.weight':(3072,1536),'encoder.bias':(3072,)}
            require(set(state) == set(shapes), 'Strict checkpoint state keys differ')
            for key, shape in shapes.items():
                require(tuple(state[key].shape) == shape and state[key].dtype == torch.float32 and torch.isfinite(state[key]).all(), 'Strict checkpoint tensor schema differs')
            state = {key:value.numpy() for key,value in state.items()}
            decoder_norms = np.linalg.norm(state['decoder'].astype(float),axis=1)
            close(decoder_norms, np.ones(3072), 'Decoder rows are not unit norm', rtol=2e-6, atol=2e-6)
            reconstruction = np.empty_like(normalized); codes = np.empty((len(x),3072),np.float32)
            for role in ('discovery','selection'):
                rows = np.flatnonzero(roles == role)
                for start in range(0,len(rows),128):
                    batch = rows[start:start+128]
                    reconstruction[batch],codes[batch] = numpy_reconstruct(normalized[batch],state,kind,block,active)
            require(np.isfinite(reconstruction).all() and np.isfinite(codes).all(), 'Nonfinite independent reconstruction')
            supports = (codes.reshape(len(x),-1,block) != 0).any(2)
            metrics = {}
            max_metric_error = 0.
            for role in ('discovery','selection'):
                rows = np.flatnonzero(roles == role); residual = reconstruction[rows]-normalized[rows]
                norm_mse = np.square(residual).mean(1); raw_mse = np.square(residual*scale).mean(1)
                rowids = ids[rows]; ww = weights(rowids)
                link = true_edges[role]
                intersections = (supports[link[:,0]] & supports[link[:,1]]).sum(1)
                unions = np.maximum(1,(supports[link[:,0]] | supports[link[:,1]]).sum(1))
                overlap = (intersections.astype(np.float32)/unions.astype(np.float32))
                actual = {'normalized_reconstruction_mse':float(ww@norm_mse),'raw_reconstruction_mse':float(ww@raw_mse),
                          'active_scalar_mean':float(ww@(codes[rows] != 0).sum(1)), 'active_block_mean':float(ww@supports[rows].sum(1)),
                          'match_count':len(set(rowids)), 'dead_scalar_fraction':float((~(codes[rows] != 0).any(0)).astype(np.float32).mean()),
                          'dead_block_fraction':float((~supports[rows].any(0)).astype(np.float32).mean()),
                          'adjacent_support_jaccard':float(weights(ids[link[:,0]])@overlap),
                          'per_match_raw_reconstruction_mse':equal_match_rows(raw_mse,rowids)['per_match']}
                target = saved['metrics'][role]
                for key in actual:
                    if isinstance(actual[key],dict):
                        require(set(actual[key]) == set(target[key]), 'Per-match reconstruction coverage differs')
                        pairs = [(actual[key][m],target[key][m]) for m in actual[key]]
                    else:
                        pairs = [(actual[key],target[key])]
                    for a,b in pairs:
                        close(a,b,'Saved metric differs: '+filename+'/'+role+'/'+key,rtol=METRIC_RTOL,atol=METRIC_ATOL)
                        max_metric_error = max(max_metric_error,abs(a-b))
                metrics[role] = actual
            raw = reconstruction.astype(float)*scale+mean
            probe_metrics, geometry_metrics, retained = retention(x,raw,data,contexts,probe,native_probe)
            identity = f'seed{seed}/{filename[:-3]}'
            prefix = identity.replace('/','__')
            for key,value in retained.items():
                arrays[prefix+'__'+key] = value
            audit['models'].append({'identity':identity,'checkpoint_sha256':output['sha256'],'seed':seed,'kind':kind,
                'active_scalar_budget':active,'group_size':block,'temporal_weight':temporal,'temporal_control':control,
                'trainable_parameters':9440256,'metrics':metrics,'max_saved_metric_absolute_difference':max_metric_error,
                'decoder_norm_max_error':float(np.max(np.abs(decoder_norms-1))),
                'frozen_probe':probe_metrics,'geometry_retention':geometry_metrics})
            print(json.dumps({'completed':len(audit['models']),'identity':identity,'selection_mse':metrics['selection']['raw_reconstruction_mse'],
                              'max_metric_difference':max_metric_error,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
            del checkpoint, state, reconstruction, codes, raw
    require(len(audit['checkpoints']) == 30 and len({e['path'] for e in audit['checkpoints']}) == 30, 'Duplicate/missing checkpoints')
    require(all(sha(path) == digest for path,digest in frozen.items()), 'Artifacts changed during audit')
    np.savez_compressed(array_path, **arrays)
    with np.load(array_path,allow_pickle=False) as saved:
        require(set(saved.files) == set(arrays) and all(np.array_equal(saved[key], value) for key,value in arrays.items()), 'Retention archive readback differs')
    audit.update(status='passed', elapsed_seconds=round(time.monotonic()-started,3), audit_script_sha256=sha(Path(__file__)),
                 all_reconstruction_metrics_recomputed=True, all_checkpoints_strict_and_finite=True,
                 all_discovery_normalizers_recomputed=True, all_frozen_probe_predictions_recomputed=True,
                 all_fixed_bin_geometry_retention_recomputed=True, coefficients_refitted=False,
                 descriptor_entity_localized=False, physical_causality_claim=False,
                 artifact_bindings=[{'path':str(path),'sha256':digest} for path,digest in frozen.items()],
                 retention_arrays={'path':str(array_path),'sha256':sha(array_path),'bytes':array_path.stat().st_size,
                                   'contains_source_labels':False,'contents':'fixed-bin conditional means, fixed-PCA projections, native probe predictions and bin coordinates'})
    args.output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','output':str(args.output),'sha256':sha(args.output),'elapsed_seconds':audit['elapsed_seconds']}))

if __name__ == '__main__':
    main()
