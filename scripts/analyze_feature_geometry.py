#!/usr/bin/env python3
"""Run approved development geometry/custom sparse pilots, never confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# Match the reproduction exporter before importing NumPy/BLAS.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '8'
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mira_interp.feature_geometry import geometry_report, train_dictionary, validate_development


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_gate(gate, input_sha, probe_sha, *, base_dir):
    """A separate reviewer binds completed probe AND causal audits to this input."""
    if gate.get('status') != 'passed_feature_development_prerequisites':
        raise ValueError('Development prerequisite review has not passed')
    if gate.get('input_npz_sha256') != input_sha or gate.get('selected_probe_sha256') != probe_sha:
        raise ValueError('Gate does not bind this input and selected probe')
    if gate['probe_audit']['sha256'] == gate['causal_audit']['sha256']:
        raise ValueError('Distinct probe and causal evidence reports are required')
    for name in ('probe_audit', 'causal_audit'):
        evidence = gate[name]
        path = Path(evidence['path'])
        path = path if path.is_absolute() else base_dir / path
        if sha256(path) != evidence['sha256']:
            raise ValueError(f'{name} has changed since review')
        report = json.loads(path.read_text())
        if report.get('status') not in ('passed', 'passed_development_probe_audit', 'passed_model_intervention_audit'):
            raise ValueError(f'{name} has not passed')
    if gate.get('scope') != 'development_only' or gate.get('physical_causality_claim') is not False:
        raise ValueError('The approved scope must be development-only without a physical-causality claim')
    if gate.get('causal_scope') not in ('internal_model_output', 'independently_measured_generated_output'):
        raise ValueError('State whether causal evidence concerns internal outputs or independently measured generated outputs')


def validate_descriptor(selection, gate):
    if not selection.get('descriptor_identity') or selection['descriptor_identity'] != gate.get('descriptor_identity'):
        raise ValueError('Selected probe and gate must agree on the descriptor identity')
    dimension = selection.get('feature_dim')
    if dimension not in (1536, 2048) or gate.get('feature_dim') != dimension:
        raise ValueError('Register the matching1536D spatial or2048D mean descriptor dimension')
    if selection.get('temporal_readout') != 'current' or gate.get('temporal_readout') != 'current':
        raise ValueError('This pilot requires a current descriptor, not concatenated current+delta features')
    return dimension


def registered_context(protocol_path, input_path, probe_path, gate_path):
    """Bind the exact registered sources, input and reviewed prerequisite gate."""
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('status') != 'registered_before_geometry_and_sparse_research':
        raise ValueError('A finalized feature research protocol is required')
    bindings = {protocol_path.resolve(): sha256(protocol_path)}
    for prefix, path in (('input', input_path), ('selected_probe', probe_path), ('gate', gate_path)):
        if Path(protocol[prefix + '_path']).resolve() != path.resolve() or sha256(path) != protocol[prefix + '_sha256']:
            raise ValueError(f'Registered {prefix} path or bytes changed')
        bindings[path.resolve()] = protocol[prefix + '_sha256']
    required_code = {'scripts/analyze_feature_geometry.py', 'scripts/export_feature_geometry.py',
                     'src/mira_interp/feature_geometry.py', 'src/mira_interp/probes.py'}
    if set(protocol['code_sha256']) != required_code:
        raise ValueError('Register the complete analysis/export/model/probe code set')
    for relative, expected in protocol['code_sha256'].items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise ValueError(f'Code changed since registration: {relative}')
        bindings[path.resolve()] = expected
    equivalence = ROOT / 'results/bsf_equivalence.json'
    if sha256(equivalence) != protocol['bsf_equivalence_sha256'] or json.loads(equivalence.read_text()).get('status') != 'passed':
        raise ValueError('Registered BSF operator audit changed or failed')
    bindings[equivalence.resolve()] = protocol['bsf_equivalence_sha256']
    gate = json.loads(gate_path.read_text())
    validate_gate(gate, protocol['input_sha256'], protocol['selected_probe_sha256'], base_dir=gate_path.resolve().parent)
    audit_path = gate_path.resolve().parent / 'input_audit.json'
    if sha256(audit_path) != gate['input_audit_sha256']:
        raise ValueError('Reviewed input export audit changed')
    audit = json.loads(audit_path.read_text())
    if (audit.get('status') != 'passed_feature_input_export' or audit['input_npz_sha256'] != protocol['input_sha256']
            or audit['selected_probe_sha256'] != protocol['selected_probe_sha256'] or audit['rows'] != 8064
            or audit['match_counts'] != {'discovery': 31, 'selection': 11}
            or not audit['all_source_label_joins_exact'] or not audit['all_arrays_readback_exact']):
        raise ValueError('Incomplete or mismatched geometry input audit')
    for key in ('entity_mapping_verified', 'view_identity_verified', 'descriptor_entity_localized', 'temporal_correspondence_scope'):
        if gate[key] != audit[key]:
            raise ValueError('Reviewed identity scope differs from source evidence')
    bindings[audit_path] = gate['input_audit_sha256']
    for name in ('probe_audit', 'causal_audit'):
        evidence = gate[name]
        path = Path(evidence['path'])
        path = path if path.is_absolute() else gate_path.resolve().parent / path
        bindings[path.resolve()] = evidence['sha256']
    return protocol, gate, bindings


def validate_registered_args(args, protocol):
    if args.stage == 'geometry':
        if (args.variable not in protocol['geometry'] or args.heldout_interval != protocol['geometry'][args.variable]
                or args.ridge_alpha != protocol['ridge_alpha'] or args.device != 'cpu'):
            raise ValueError('Geometry variable/interval/alpha/device differs from registered CPU comparison')
        expected_speed = protocol['minimum_heading_speed'] if args.variable == 'heading' else None
        if args.minimum_heading_speed != expected_speed:
            raise ValueError('Heading eligibility differs from registration')
    elif args.stage == 'sparse':
        sparse = protocol['sparse']
        if (args.width != sparse['width'] or args.active_budgets != sparse['active_budgets']
                or args.steps != sparse['steps'] or args.seed not in sparse['seeds']
                or args.temporal_weight != sparse['temporal_weight']
                or args.temporal_shuffle_control != sparse['temporal_shuffled_control'] or args.expected_dt != .1
                or args.device != sparse['device'] or args.device != 'cuda'):
            raise ValueError('Sparse capacity/activity/training/temporal arguments differ from registration')
    else:
        raise ValueError('Geometry and sparse stages must run separately under the ordered review gates')


def verify_bindings(bindings):
    if any(sha256(path) != expected for path, expected in bindings.items()):
        raise ValueError('Registered source, input or evidence changed during execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--selected-probe', required=True, type=Path)
    parser.add_argument('--gate', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/feature_development_v1.json')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stage', choices=('geometry', 'sparse'), required=True)
    parser.add_argument('--variable', choices=('height', 'vx', 'vy', 'vz', 'speed', 'heading'), default='speed')
    parser.add_argument('--heldout-interval', type=float, nargs=2)
    parser.add_argument('--minimum-heading-speed', type=float)
    parser.add_argument('--ridge-alpha', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--active-budgets', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--temporal-weight', type=float, default=0.)
    parser.add_argument('--temporal-shuffle-control', action='store_true')
    parser.add_argument('--expected-dt', type=float, default=0.1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new output directory; completed artifacts are never overwritten')
    protocol, gate, bindings = registered_context(args.protocol, args.input, args.selected_probe, args.gate)
    validate_registered_args(args, protocol)
    input_hash, probe_hash, gate_hash = sha256(args.input), sha256(args.selected_probe), sha256(args.gate)
    validate_gate(gate, input_hash, probe_hash, base_dir=args.gate.resolve().parent)
    selection = json.loads(args.selected_probe.read_text())
    dimension = validate_descriptor(selection, gate)
    with np.load(args.input, allow_pickle=False) as source:
        roles = source['split'].astype(str)
        if set(roles) != {'discovery', 'selection'}:
            raise ValueError('Refusing to load arrays containing confirmation rows')
        if len(roles) > 50000:
            raise ValueError('Pilot row budget exceeds50000; review memory use before expanding')
        data = {key: source[key] for key in ('X', 'velocity', 'match_ids', 'split', 'clip_ids',
                                             'entity_ids', 'view_index', 'frame_index', 'timestamps')}
        if args.variable == 'height':
            data['position'] = source['position']
    counts = validate_development(data, dimension=dimension)
    if not 8 <= args.width <= 8192 or args.width % 8 or not args.active_budgets or any(
        value <= 0 or value > 128 or value >= args.width or value % 8 for value in args.active_budgets
    ) or len(set(args.active_budgets)) != len(args.active_budgets):
        raise ValueError('Width must be a multiple of8 up to8192; unique active budgets must be multiples of8 up to128 and below width')
    torch.set_num_threads(4)
    result = {'status': 'running', 'scope': 'development_only', 'match_counts': counts,
              'protocol_sha256': sha256(args.protocol), 'protocol_path': str(args.protocol.resolve()),
              'execution_args': {key: getattr(args, key) for key in ('stage', 'variable', 'heldout_interval',
                  'minimum_heading_speed', 'ridge_alpha', 'steps', 'width', 'active_budgets', 'seed', 'device',
                  'temporal_weight', 'temporal_shuffle_control', 'expected_dt')},
              'input_npz_sha256': input_hash, 'selected_probe_sha256': probe_hash, 'gate_sha256': gate_hash,
              'descriptor_identity': selection['descriptor_identity'], 'dimension': dimension,
              'temporal_readout': 'current', 'causal_evidence_scope': gate['causal_scope'],
              'descriptor_entity_localized': gate.get('descriptor_entity_localized', False),
              'temporal_correspondence_scope': gate.get('temporal_correspondence_scope', 'not_verified'),
              'confirmation_rows_loaded': False, 'physical_causality_claim': False,
              'selected_probe_used_for': 'descriptor provenance only; no probe refit or probe-success claim',
              'geometry': None, 'sparse': [], 'outputs': {}}
    if args.stage in ('geometry', 'both'):
        if args.heldout_interval is None:
            raise ValueError('A prospectively chosen held-value interval is required')
        velocity = data['velocity'].astype(float)
        keep = np.ones(len(velocity), bool)
        speed = np.hypot(velocity[:, 0], velocity[:, 1])
        if args.variable == 'heading':
            threshold = args.minimum_heading_speed
            if threshold is None or not np.isfinite(threshold) or threshold <= 0:
                raise ValueError('Heading requires an explicit positive minimum speed')
            keep = speed >= threshold
            values = np.arctan2(velocity[:, 1], velocity[:, 0])
            values = (values + np.pi) % (2 * np.pi) - np.pi
        elif args.variable == 'height':
            if data['position'].shape != (len(velocity), 3) or not np.isfinite(data['position']).all():
                raise ValueError('Height requires finite selected-entity position[N,3]')
            values = data['position'][:, 2]
        elif args.variable == 'speed':
            values = speed
        else:
            values = velocity[:, ('vx', 'vy', 'vz').index(args.variable)]
        result['geometry'] = geometry_report(data['X'][keep], values[keep], data['match_ids'][keep],
            data['split'][keep], heldout_interval=args.heldout_interval, circular=args.variable == 'heading',
            alpha=args.ridge_alpha)
        result['geometry'].update(variable=args.variable, speed_definition='horizontal sqrt(vx²+vy²)',
            minimum_heading_speed=args.minimum_heading_speed, below_speed_threshold_rows=int((~keep).sum()))
    args.output_dir.mkdir(parents=True)
    if args.stage in ('sparse', 'both'):
        for active in args.active_budgets:
            variants = [('relu', 0., 'adjacent'), ('signed', 0., 'adjacent'), ('block', 0., 'adjacent')]
            if args.temporal_weight:
                variants.append(('block', args.temporal_weight, 'adjacent'))
                if args.temporal_shuffle_control:
                    variants.append(('block', args.temporal_weight, 'shuffled_within_match'))
            for kind, temporal, temporal_control in variants:
                model, normalizer, report = train_dictionary(data, kind=kind, width=args.width, active=active,
                    steps=args.steps, seed=args.seed, device=args.device, temporal_weight=temporal,
                    batch_size=protocol['sparse']['batch_size'], learning_rate=protocol['sparse']['learning_rate'],
                    group_size=protocol['sparse']['group_size'],
                    entity_mapping_verified=gate.get('entity_mapping_verified') is True,
                    view_identity_verified=gate.get('view_identity_verified') is True,
                    expected_dt=args.expected_dt, temporal_control=temporal_control)
                name = f'{kind}_active{active}' + ('_temporal' if temporal else '')
                if temporal_control != 'adjacent':
                    name += '_shuffled'
                path = args.output_dir / f'{name}.pt'
                torch.save({'state_dict': model.state_dict(), 'normalizer':
                            {key: torch.from_numpy(value) for key, value in normalizer.items()}, 'report': report}, path)
                result['outputs'][path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
                result['sparse'].append(report)
    verify_bindings(bindings)
    result['status'] = 'passed_development_feature_analysis'
    result['code_sha256'] = protocol['code_sha256']
    (args.output_dir / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'report': str(args.output_dir / 'report.json')}))


if __name__ == '__main__':
    main()
