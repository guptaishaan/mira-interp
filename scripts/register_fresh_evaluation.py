#!/usr/bin/env python3
"""Freeze development-selected primary readouts and comparisons before fresh capture."""
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from mira_interp.model_loading import sha256
from prepare_development import save
d=ROOT/'results/development_probes_v3'
report=json.loads((d/'development_report.json').read_text())
selection=json.loads((d/'development_selection.json').read_text())
audit=json.loads((d/'probe_audit.json').read_text())
if report['status']!='passed_development_analysis' or audit['status']!='passed':
    raise ValueError('Development analysis and independent probe audit must pass')
if audit['development_selection_sha256']!=sha256(d/'development_selection.json') or audit['model_archive_sha256']!=sha256(d/'development_models.npz'):
    raise ValueError('Audited model selection changed')
if audit['development_report_sha256']!=sha256(d/'development_report.json'):
    raise ValueError('Audited development scores changed')
rows=report['stage1']+report['stage2']
choices={}
for definition in ('absolute30','role12'):
    choices[definition]={}
    for group in ('all','position','velocity'):
        def best(candidates):
            chosen=min(candidates,key=lambda r:r['targets'][definition]['groups'][group]['main']['selection']['normalized_mse'])
            return f"{chosen['name']}/{definition}/{group}"
        choices[definition][group]={
            'primary':best([r for r in rows if r['kind']=='residual']),
            'mean_current':best([r for r in report['stage1'] if r['name'].startswith('mean/')]),
            'spatial_current':best([r for r in report['stage1'] if r['name'].startswith('spatial/')]),
            'codec_spatial':f'codec_spatial_X/{definition}/{group}',
            'codec_mean':f'codec_mean_X/{definition}/{group}',
            'RGB':f'RGB_X/{definition}/{group}'}
out=ROOT/'configs/fresh_evaluation_v3.json'
if out.exists():
    raise ValueError('Fresh evaluation already registered; do not overwrite')
save(out,dict(registered_utc=datetime.now(timezone.utc).isoformat(),status='frozen_before_fresh_capture',
              scope='fresh-match observational confirmation; not physical causal control',choices=choices,
              development_selection_sha256=sha256(d/'development_selection.json'),model_archive_sha256=sha256(d/'development_models.npz'),
              development_report_sha256=sha256(d/'development_report.json'),
              probe_audit_sha256=sha256(d/'probe_audit.json'),candidate_manifest_sha256=sha256(ROOT/'data/fresh_confirmation_split_manifest.json'),
              feature_capture_protocol_sha256=sha256(ROOT/'configs/development_v3.json'),
              capture_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'scripts/capture_fresh_confirmation.py',ROOT/'src/mira_interp/development_capture.py',ROOT/'src/mira_interp/model_loading.py']},
              evaluation_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'scripts/confirm_development.py',ROOT/'scripts/aggregate_captures.py',ROOT/'src/mira_interp/probes.py',ROOT/'src/mira_interp/development.py']},
              model_refit=False,rows='latent2..7 perview',bootstrap='500 paired whole-match resamples; descriptive intervals, no family-wide significance claim',
              choices_rule='minimum development selection standardizedMSE within each target family and residual readouts; report fixed current-mean/current-spatial and codec/RGB comparators',
              no_fresh_activation_or_prediction_inspected=True))
print(json.dumps(choices,indent=2))
