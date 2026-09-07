#!/usr/bin/env python3
"""Summarize already audited edit delivery; no new model execution or fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(values):
        return dict(n=0, minimum=None, median=None, maximum=None)
    assert np.isfinite(values).all()
    return dict(n=len(values), minimum=float(values.min()), median=float(np.median(values)), maximum=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('selection', 'confirmation'), required=True)
    args = parser.parse_args()
    paths = {key: ROOT / path for key, path in {
        'generation': f'results/rollout_steering_v2/{args.phase}.json',
        'generation_audit': f'results/rollout_steering_v2/{args.phase}_audit.json',
        'measurement': f'results/generated_evaluation_v2/{args.phase}.json',
        'measurement_audit': f'results/generated_evaluation_v2/{args.phase}_audit.json',
        'development': 'results/development_probes_v3/development_report.json',
        'development_audit': 'results/development_probes_v3/probe_audit.json',
    }.items()}
    bindings = {key: sha(path) for key, path in paths.items()}
    code_hash = sha(__file__)
    data = {key: json.loads(path.read_text()) for key, path in paths.items()}
    gen, audit, measurement, ma = [data[key] for key in ('generation', 'generation_audit', 'measurement', 'measurement_audit')]
    assert gen['status'] == 'passed_rollout_generation' and all(r['split'] == args.phase for r in gen['records'])
    assert audit['status'] == 'passed_rollout_generation_audit' and audit['manifest_sha256'] == bindings['generation']
    assert audit['saved_tile_descriptor_lift_metrics_recomputed'] and audit['complete_registered_grid']
    assert ma['status'] == 'passed_generated_measurement_audit' and ma['evaluation_sha256'] == bindings['measurement']
    assert ma['generation_manifest_sha256'] == bindings['generation'] and ma['generation_audit_sha256'] == bindings['generation_audit']
    assert measurement['records'] == gen['n_records'] == {'selection': 420, 'confirmation': 924}[args.phase]
    assert data['development_audit']['status'] == 'passed'
    assert data['development_audit']['development_report_sha256'] == bindings['development']
    descriptor = next(r for r in data['development']['stage1'] if r['name'] == 'spatial/block_15_output')
    names = descriptor['targets']['absolute30']['groups']['position']['target_names']
    target_index = names.index('ball.location.z')
    rows = []
    for record in gen['records']:
        if record['intervention_type'] == 'baseline':
            continue
        effective = record['height']['effective_dose']
        requested = record['requested_descriptor_delta_l2']
        actual = record['descriptor_delta_l2']
        mismatch = record['descriptor_rounding_error_l2']
        rows.append(dict(record_id=record['record_id'], path=record['intervention_type'],
                         requested_dose=record['dose'], effective_dose=effective,
                         actual_internal_height_probe_delta=record['actual_site_position_proxy_delta'][target_index],
                         linear_probe_delivery_fraction=record['actual_site_position_proxy_delta'][target_index] / effective
                         if record['intervention_type'] == 'probe_linear' and abs(effective) > 1e-9 else None,
                         relative_descriptor_mismatch=mismatch / requested if requested > 1e-9 else None,
                         actual_to_requested_raw_norm=record['actual_edit_l2'] / record['requested_raw_lift_l2']
                         if record['requested_raw_lift_l2'] > 1e-9 else None,
                         descriptor_delta_cosine=(actual**2 + requested**2 - mismatch**2) / (2 * actual * requested)
                         if actual > 1e-9 and requested > 1e-9 else None))
    folder = ROOT / 'results/generated_evaluation_v2' / (args.phase + '_delivery')
    folder.mkdir(exist_ok=False)
    csv_path = folder / 'per_condition.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    grouped = {}
    for path in sorted({r['path'] for r in rows}):
        selected = [r for r in rows if r['path'] == path]
        grouped[path] = dict(n_conditions=len(selected), zero_effective_dose_conditions=sum(abs(r['effective_dose']) <= 1e-9 for r in selected),
                             **{key: summary([r[key] for r in selected]) for key in
                                ('linear_probe_delivery_fraction', 'relative_descriptor_mismatch',
                                 'actual_to_requested_raw_norm', 'descriptor_delta_cosine')})
    assert all(sha(paths[key]) == value for key, value in bindings.items()) and sha(__file__) == code_hash
    report = dict(status='passed_audited_edit_delivery_summary', phase=args.phase,
                  analysis_type='Posthoc descriptive engineering summary of previously independently recomputed metadata.',
                  sources={key: dict(path=str(paths[key].relative_to(ROOT)), sha256=value) for key, value in bindings.items()},
                  script_sha256=code_hash, target_names=names, target_index=target_index, summary=grouped,
                  per_condition=dict(path=str(csv_path.relative_to(ROOT)), sha256=sha(csv_path)),
                  model_execution=False, new_fitting_or_selection=False,
                  limits=['Delivery fraction measures the internal probe scalar used to construct linear edits; it is not independent physical validation.',
                          'Raw and descriptor norms differ in scale; controls match the quadratic path only.',
                          'These summaries do not measure downstream propagation, explain all rounding effects, or establish physical control.'])
    (folder / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], phase=args.phase, summary=grouped)))


if __name__ == '__main__':
    main()
