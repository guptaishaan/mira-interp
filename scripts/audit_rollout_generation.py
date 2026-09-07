#!/usr/bin/env python3
"""Independently audit persisted rollout coverage, timing, actions, and array metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from mira_interp.probes import RidgeModel

PATHS = ('probe_linear', 'affine_forward', 'quadratic_forward', 'random_norm', 'wrong_variable_ball_x')
DOSES = (-600., -300., 300., 600.)
SUPPORT = (86.37890625, 1810.9219970703125)
PROBE = 'spatial/block_15_output/absolute30/position'
SHAPES = {'frames': (4,24,3,288,512), 'latents': (4,12,9,16,32),
          'native_edit_tile': (9,16,2048), 'edited_tile': (9,16,2048),
          'native_edit_descriptor': (1,1536), 'edited_descriptor': (1,1536)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def checked(entry, parse=True):
    path = Path(entry['path'])
    require(sha(path) == entry['sha256'], f'Bound file changed: {path}')
    return read(path) if parse else path


def close(a, b, message, *, atol=1e-5, rtol=1e-5):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
            and np.allclose(a, b, atol=atol, rtol=rtol), message)


def validate_trace(trace, registration):
    require(len(trace) == 45, 'Expected one context plus4*(10 denoising+1 cache) calls')
    dtype = trace[0]['latent_dtype']
    require(dtype in {'torch.float32', 'torch.bfloat16'}, 'Unexpected sampler precision')
    schedule = registration['schedule_values_bf16' if dtype == 'torch.bfloat16' else 'schedule_values_fp32']
    for index, row in enumerate(trace):
        require(row['call_index'] == index and row['latent_dtype'] == row['tau_dtype'] == dtype, 'Trace dtype/order changed')
        for key in ('latent_input_sha256', 'action_features_sha256'):
            require(len(row[key]) == 64 and all(c in '0123456789abcdef' for c in row[key]), 'Malformed trace hash')
        if index == 0:
            expected = ('context_cache_initialization', -1, -1, 1., 8, False, True, False)
            nominal = 1.
        else:
            latent, step = divmod(index-1, 11)
            cache = step == 10
            expected = ('generated_cache_update' if cache else 'denoising', latent, 9 if cache else step,
                        1. if cache else schedule[step], 1, True, cache, latent == 0 and step == 8)
            nominal = 1. if cache else registration['schedule_values_fp32'][step]
        actual = tuple(row[k] for k in ('phase','generated_latent_index','step_index','tau','time_tokens','cache_present','return_kv','target_call'))
        require(actual == expected and row['nominal_schedule_tau'] == nominal, f'Actual sampler call{index} differs')
    return True


def load_arrays(row):
    path = Path(row['generated_artifact_path'])
    require(path.stat().st_size == row['generated_bytes'] and sha(path) == row['generated_sha256'], 'Generated artifact hash/size mismatch')
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == set(SHAPES), 'Unexpected generated archive keys')
        arrays = {k: z[k] for k in z.files}
    for key, value in arrays.items():
        require(value.shape == SHAPES[key] and value.dtype == np.float32 and np.isfinite(value).all(), f'Invalid generated array:{key}')
    require(arrays['frames'].min() >= 0 and arrays['frames'].max() <= 1, 'Saved video outside0..1')
    require(array_hash(arrays['latents']) == row['latent_sha256'], 'Latent tensor hash mismatch')
    require(np.isfinite(row['raw_decoded_min']) and np.isfinite(row['raw_decoded_max'])
            and row['raw_decoded_min'] <= row['raw_decoded_max'], 'Invalid raw range')
    require(0 <= row['decoded_clipped_values'] <= arrays['frames'].size, 'Invalid clipping count')
    if row['decoded_clipped_values'] == 0:
        require(array_hash(arrays['frames']) == row['raw_decoded_sha256'], 'Unclipped video hash mismatch')
        require(array_hash(arrays['frames'][:,:16]) == row['raw_decoded_context_sha256'], 'Unclipped context hash mismatch')
    return arrays


def descriptor(tile, q):
    # Independent definition: explicit twelve equal 3x4 token bin averages.
    bins = np.stack([tile[y:y+3,x:x+4].mean(axis=(0,1), dtype=np.float64)
                     for y in (0,3,6) for x in (0,4,8,12)])
    return (bins @ q.astype(np.float64)).reshape(1,1536)


def lift(delta, q):
    bins = delta.reshape(3,4,128).astype(np.float64) @ q.astype(np.float64).T
    return np.repeat(np.repeat(bins, 3, axis=0), 4, axis=1)


def pilot_dose_for_base(raw):
    require(np.isfinite(raw), 'Finite baseline probe height required')
    base = float(np.clip(raw, *SUPPORT))
    return 300. if float(np.clip(base+300., *SUPPORT)) > base else -300.


def expected_delta(row, probe, maps, q):
    base = float(row['base_height_unclipped_from_baseline'])
    clipped = float(np.clip(base, *SUPPORT))
    target = float(np.clip(clipped + row['dose'], *SUPPORT))
    effective = target-clipped
    heights = {'base_height_raw': base, 'base_height_clipped': clipped, 'requested_dose': float(row['dose']),
        'target_height_clipped': target, 'effective_dose': effective, 'base_clipped': clipped != base,
        'target_clipped': target != clipped+row['dose']}
    require(row['height'] == heights, 'Clipping/effective dose differs from registered definition')
    name = row['intervention_type']
    if name == 'baseline' or effective == 0:
        return np.zeros((1,1536))
    def forward(kind, value):
        z = (value-float(maps['physical_mean']))/float(maps['physical_scale'])
        x = np.array([[z]] if kind == 'affine' else [[z,z*z]])
        get = lambda k: maps[kind+'__'+k]
        return ((x-get('x_mean'))/get('x_scale') @ get('coefficient'))*get('y_scale')+get('y_mean')
    gradient = probe.coefficient * probe.y_scale / probe.x_scale[:,None]
    if name == 'probe_linear':
        return (effective*gradient[:,2]/np.dot(gradient[:,2],gradient[:,2]))[None]
    if name in {'affine_forward','quadratic_forward'}:
        kind = name.split('_')[0]
        return forward(kind,target)-forward(kind,clipped)
    quadratic = forward('quadratic',target)-forward('quadratic',clipped)
    direction = (np.random.default_rng(row['seed']+1777).normal(size=1536)
                 if name == 'random_norm' else gradient[:,0])
    candidate = (np.sign(effective)*direction)[None]
    return candidate * (np.linalg.norm(lift(quadratic,q))/np.linalg.norm(lift(candidate,q)))


def audit_array_metrics(row, data, baseline_row, baseline, probe, maps, q):
    require(np.array_equal(data['native_edit_tile'], baseline['native_edit_tile']), 'Pre-edit tile differs across paired conditions')
    require(np.array_equal(data['latents'][:,:8], baseline['latents'][:,:8])
            and row['latent_context_bitwise_equal_to_baseline'] is True, 'Fixed latent context changed')
    context_difference = np.max(np.abs(data['frames'][:,:16]-baseline['frames'][:,:16]))
    close(context_difference, row['decoded_context_max_abs_difference'], 'Context difference annotation differs', atol=0, rtol=0)
    require(row['decoded_context_bitwise_equal_to_baseline'] ==
            (row['raw_decoded_context_sha256'] == baseline_row['raw_decoded_context_sha256']), 'Raw context hash equality annotation differs')
    if row['decoded_context_bitwise_equal_to_baseline']:
        require(context_difference == 0, 'Equal raw context hashes contradict stored pixels')
    for key in ('native_edit_descriptor','edited_descriptor'):
        tile_key = 'native_edit_tile' if key == 'native_edit_descriptor' else 'edited_tile'
        close(data[key], descriptor(data[tile_key],q), 'Saved descriptor does not match tile', atol=2e-4, rtol=2e-5)
    before, after = data['native_edit_descriptor'], data['edited_descriptor']
    # The frozen probe is a deterministic linear map; no source target is loaded.
    base_height = float(probe.predict(baseline['native_edit_descriptor'])[0,2])
    close(base_height, row['base_height_unclipped_from_baseline'], 'Baseline height is not frozen model readout', atol=1e-8, rtol=1e-10)
    close((probe.predict(after)-probe.predict(before))[0], row['actual_site_position_proxy_delta'], 'Saved proxy shift differs', atol=1e-7, rtol=1e-8)
    requested = expected_delta(row,probe,maps,q)
    raw_lift = lift(requested,q)
    close(np.linalg.norm(requested), row['requested_descriptor_delta_l2'], 'Requested descriptor norm differs', atol=2e-5, rtol=2e-5)
    close(np.linalg.norm(raw_lift), row['requested_raw_lift_l2'], 'Requested raw lift norm differs', atol=2e-4, rtol=2e-5)
    close(np.linalg.norm(after-before), row['descriptor_delta_l2'], 'Actual descriptor norm differs', atol=1e-6, rtol=2e-5)
    close(np.linalg.norm(data['edited_tile']-data['native_edit_tile']), row['actual_edit_l2'], 'Actual raw edit norm differs', atol=1e-5, rtol=2e-5)
    close(np.linalg.norm(after-before-requested), row['descriptor_rounding_error_l2'], 'Rounding-error annotation differs', atol=2e-4, rtol=2e-5)
    ideal = data['native_edit_tile'].astype(np.float64)+raw_lift
    # FP32 lifting followed by BF16 residual storage can round either way near a tie.
    # Bound the rounding error by one BF16 spacing, rather than demand CPU/GPU bit identity.
    spacing = np.maximum(np.abs(ideal)/128, 1e-5) if row['world_model_calls'][9]['latent_dtype'] == 'torch.bfloat16' else np.maximum(np.abs(ideal)*2e-6,2e-5)
    require(np.all(np.abs(data['edited_tile']-ideal) <= spacing+2e-5), 'Edited tile does not implement the registered lift within storage precision')
    difference = data['latents'][:,8:].astype(float)-baseline['latents'][:,8:]
    future_norm = np.sqrt(np.square(difference).sum(axis=(0,2,3,4)))
    close(future_norm, row['future_latent_l2_by_time'], 'Future latent change differs', atol=1e-8, rtol=1e-10)
    if row['intervention_type'] == 'baseline' or row['height']['effective_dose'] == 0:
        require(np.array_equal(data['native_edit_tile'],data['edited_tile']) and np.array_equal(data['frames'],baseline['frames'])
                and np.array_equal(data['latents'],baseline['latents'])
                and row['raw_decoded_sha256'] == baseline_row['raw_decoded_sha256'], 'Zero dose/effective dose changed generated output')
    return {'record_id': row['record_id'], 'generated_sha256': row['generated_sha256'],
            'decoded_context_exact': bool(context_difference == 0),
            'max_lift_storage_error': float(np.max(np.abs(data['edited_tile']-ideal)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot','selection','confirmation'), required=True)
    parser.add_argument('--directory', type=Path, default=ROOT/'results/rollout_steering_v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.directory/(args.phase+'_audit.json')
    require(not output.exists(), 'Completed audits are immutable; choose a new destination')
    started = time.monotonic()
    registration_path = args.directory/'registration.json'
    registration, registration_sha = read(registration_path), sha(registration_path)
    require(registration['status'] == 'registered_before_rollout_steering', 'Missing frozen registration')
    require(registration['paths'] == list(PATHS) and registration['doses_uu'] == [-600.,-300.,0.,300.,600.]
            and registration['height_support_uu'] == list(SUPPORT) and registration['seeds'] == [2026090701,2026090702], 'Registered conditions changed')
    require((registration['context_frames'],registration['generated_frames'],registration['n_diffusion_steps'],registration['intervention_step_index'])
            == (16,8,10,8) and registration['site'] == 'block_15_output' and registration['view'] == registration['generated_latent_index'] == 0, 'Intervention alignment changed')
    for name, expected in registration['code_sha256'].items():
        require(sha(ROOT/name) == expected, 'Registered generation code changed:'+name)
    for item in registration['bindings'].values():
        checked(item,parse=False)
    for item in registration['probe_bindings'].values():
        checked(item,parse=False)
    input_report = checked(registration['bindings']['inputs'])
    input_audit = checked(registration['bindings']['input_audit'])
    require(input_audit['status'] == 'passed_rollout_input_audit' and input_audit['input_report_sha256'] == registration['bindings']['inputs']['sha256']
            and input_report['records'] == registration['records'], 'Source audit/cohort binding differs')
    probe_selection = checked(registration['probe_bindings']['selection'])
    index = next(i for i,m in enumerate(probe_selection['models']) if m['name'] == PROBE)
    with np.load(checked(registration['probe_bindings']['archive'],False),allow_pickle=False) as z:
        values = {key:z[f'site{index}_main_{key}'] for key in ('alpha','x_mean','x_scale','y_mean','y_scale','coefficient')}
    values['alpha'] = float(values['alpha'])
    probe = RidgeModel(**values)
    require(probe.fingerprint() == probe_selection['models'][index]['model_sha256'], 'Selected probe archive differs')
    require(probe.fingerprint() == registration['probe_sha256'], 'Frozen probe fingerprint differs')
    with np.load(checked(registration['bindings']['maps'],False),allow_pickle=False) as z:
        maps = {k:z[k] for k in z.files}
    q = np.linalg.qr(np.random.default_rng(20260907).standard_normal((2048,128)))[0].astype(np.float32)
    close(q.T@q, np.eye(128), 'Descriptor projection is not orthonormal', atol=1e-6,rtol=1e-6)
    path = args.directory/(args.phase+'.json')
    report = read(path)
    require(report['status'] == ('passed_rollout_steering_pilot' if args.phase == 'pilot' else 'passed_rollout_generation')
            and report['registration_sha256'] == registration_sha, 'Phase is incomplete or uses different registration')
    require(report['physical_control_established'] is False and report['source_targets_loaded'] is False
            and report['independent_video_evaluation_completed'] is False, 'Generation manifest makes unsupported claims')
    sources = [r for r in registration['records'] if r['role'] == args.phase]
    require(len(sources) == {'pilot':1,'selection':10,'confirmation':22}[args.phase], 'Wrong phase match count')
    seeds = registration['seeds'][:1] if args.phase == 'pilot' else registration['seeds']
    rows = report['records']
    if args.phase == 'pilot':
        require(registration['pilot_dose_candidates_uu'] == [300.,-300.] and registration['pilot_dose_rule'], 'Pilot support rule is not registered')
        base_row = next(r for r in rows if r['intervention_type'] == 'baseline')
        pilot_dose = pilot_dose_for_base(base_row['base_height_unclipped_from_baseline'])
        grid = [('baseline',0.),('quadratic_forward',pilot_dose)]
    else:
        grid = [('baseline',0.)]+[(n,d) for n in PATHS for d in DOSES]
    expected = {(s['clip_id'],s['match_id'],seed,name,dose) for s in sources for seed in seeds for name,dose in grid}
    require(len(rows) == report['n_records'] == len(expected)
            and {(r['clip_id'],r['match_id'],r['seed'],r['intervention_type'],r['dose']) for r in rows} == expected, 'Incomplete/duplicate registered condition grid')
    if args.phase == 'pilot':
        active = next(r for r in rows if r['intervention_type'] == 'quadratic_forward')
        require(active['height']['effective_dose'] != 0 and active['actual_edit_l2'] > 0, 'Pilot active-edit control was clipped to a no-op')
    require(len({r['record_id'] for r in rows}) == len(rows) and report['match_count'] == len(sources)
            and report['seed_count'] == len(seeds), 'IDs or reported coverage differ')
    if args.phase != 'pilot':
        pilot = read(args.directory/'pilot_audit.json')
        require(pilot['status'] == 'passed_rollout_generation_audit' and pilot['manifest_sha256'] == sha(args.directory/'pilot.json'), 'Independent pilot audit required')
        for index, worker in enumerate(report['workers']):
            sub = checked(worker)
            require(sub['worker_index'] == index and sub['worker_count'] == 2 and sub['registration_sha256'] == registration_sha
                    and sub['status'] == 'passed_rollout_generation', 'Worker metadata differs')
            subset = {s['clip_id'] for s in sources[index::2]}
            selected_rows = [r for r in rows if r['clip_id'] in subset]
            require(sorted(sub['records'],key=lambda x:x['record_id']) == sorted(selected_rows,key=lambda x:x['record_id']), 'Worker partition differs')
        require(len(report['workers']) == 2, 'Expected two workers')
    if args.phase == 'confirmation':
        audit = read(args.directory/'selection_audit.json')
        require(audit['status'] == 'passed_rollout_generation_audit' and audit['manifest_sha256'] == sha(args.directory/'selection.json'), 'Completed selection audit required')
    if args.phase == 'pilot':
        controls = report['controls']
        require(report['full_inferences'] == 5 and all(controls.get(k) is True for k in ('unhooked_vs_capture_bitwise_equal','future_placeholder_bitwise_invariance','zero_edit_bitwise_equal','future_actions_unchanged')), 'Pilot controls failed')
        evidence = controls['persisted_control_hashes']
        require(set(evidence) == {'unhooked','hooked','alternate255','zero_edit'}, 'Persisted pilot hash evidence missing')
        require(all(v == evidence['hooked'] for v in evidence.values()), 'Pilot control output/action hashes differ')
    audited = []
    for source in sources:
        require(sha(Path(source['path'])) == source['sha256'], 'Source artifact changed')
        # Read only four action streams; source pixels and physical labels are not inspected.
        with np.load(source['path'],allow_pickle=False) as z:
            action_hash = array_hash(z['actions'])
        for seed in seeds:
            pair = [r for r in rows if r['clip_id'] == source['clip_id'] and r['seed'] == seed]
            baseline_row = next(r for r in pair if r['intervention_type'] == 'baseline')
            baseline = load_arrays(baseline_row)
            for row in pair:
                saved = checked(row['record_report'])
                require(saved == {k:v for k,v in row.items() if k != 'record_report'}, 'Per-condition report differs from manifest')
                identifier = f"{source['clip_id']}_seed{seed}_{row['intervention_type']}_dose{int(row['dose']):+d}"
                require(row['record_id'] == identifier and row['baseline_record_id'] == baseline_row['record_id']
                        and row['split'] == source['role'] and row['source_sha256'] == source['sha256']
                        and row['registration_sha256'] == registration_sha and row['status'] == 'passed_generation', 'Condition identity/lineage differs')
                require(row['physical_control_established'] is False and row['source_targets_loaded'] is False
                        and row['action_batch_sha256'] == action_hash, 'Source/action/scope annotation invalid')
                validate_trace(row['world_model_calls'], registration)
                base_trace = baseline_row['world_model_calls']
                require([c['action_features_sha256'] for c in row['world_model_calls']] == [c['action_features_sha256'] for c in base_trace], 'Action conditioning changed')
                require(row['world_model_calls'][:10] == base_trace[:10], 'Paired sampler inputs differ before intervention')
                data = baseline if row is baseline_row else load_arrays(row)
                audited.append(audit_array_metrics(row,data,baseline_row,baseline,probe,maps,q))
                if args.phase == 'pilot':
                    require(evidence['hooked'] == {k:baseline_row[k] for k in ('raw_decoded_sha256','raw_decoded_context_sha256','latent_sha256','action_batch_sha256')}, 'Pilot hash evidence differs from saved baseline')
                del data
            del baseline
        print(json.dumps({'phase':args.phase,'audited_records':len(audited),'completed_match':source['match_id']}),flush=True)
    require(report['generated_artifact_bytes'] == sum(r['generated_bytes'] for r in rows), 'Total byte annotation differs')
    require(report['all_context_pixels_unchanged'] == all(r['decoded_context_bitwise_equal_to_baseline'] for r in rows), 'Context equality summary differs')
    result = {'status':'passed_rollout_generation_audit','phase':args.phase,'manifest_sha256':sha(path),
        'registration_sha256':registration_sha,'script_sha256':sha(Path(__file__)), 'records':audited,
        'n_records':len(rows),'n_matches':len(sources),'n_seeds':len(seeds), 'complete_registered_grid':True,
        'source_report_and_generated_hashes_verified':True,'exact_action_conditioning_preserved':True,
        'preintervention_sampler_inputs_bitwise_equal':True,'fixed_context_latents_unchanged':True,
        'all_stored_context_pixels_exact':all(r['decoded_context_exact'] for r in audited),
        'all45_sampler_calls_validated':True,'saved_tile_descriptor_lift_metrics_recomputed':True,
        'source_labels_loaded':False,'source_future_pixels_loaded':False,'gpu_used':False,
        'physical_control_established':False,'independent_generated_accuracy_established':False,
        'limits':['Discarded pilot controls verified by persisted hash equality and executed-code binding, not independently rerun.',
                  'Clipped pixel archives cannot reconstruct unclipped decoder values; raw hashes remain producer evidence when clipping occurred.',
                  'CPU lift recomputation allows one BF16 storage spacing and small FP32 reduction tolerances; this is an engineering integrity audit.'],
        'elapsed_seconds':time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'},indent=2))


if __name__ == '__main__':
    main()
