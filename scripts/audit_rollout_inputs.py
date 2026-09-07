#!/usr/bin/env python3
"""Independent extended-input audit: strict exclusions and exact original float context."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from mira_interp.data import TARGET_NAMES
from mira_interp.model_loading import sha256

QUALITY_REASONS = {
    'QualityExclusion: fully frozen physical-state transition in proposed live clip',
    'QualityExclusion: demolition-affected entity in proposed live clip',
    'QualityExclusion: cross-view ball trajectories disagree beyond 100 uu after up to four frames of lag',
}


def save(path, report):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def main():
    path = ROOT/'results/rollout_inputs_v1_audit.json'
    if path.exists(): raise ValueError('Preserve prior rollout input audits')
    config_path, input_path = ROOT/'configs/rollout_inputs_v1.json', ROOT/'results/rollout_inputs_v1.json'
    config, inputs = (json.loads(value.read_text()) for value in (config_path, input_path))
    report = {'status':'running','audit_script_sha256':sha256(Path(__file__)),
        'input_report_sha256':sha256(input_path),'registration_sha256':sha256(config_path),
        'started_utc':datetime.now(timezone.utc).isoformat(),'records':[],'exclusions':inputs.get('exclusions',[])}
    save(path, report)
    try:
        if inputs['status'] != 'passed_quality_qualified_rollout_inputs' or inputs['errors'] or len(inputs['records']) != 33:
            raise ValueError('Require the completed33-input preparation with no integrity errors')
        if inputs['registration_sha256'] != sha256(config_path): raise ValueError('Registration changed')
        for group in ('source_manifest_sha256','code_sha256'):
            for name, digest in config[group].items():
                if sha256(ROOT/name) != digest: raise ValueError('Registered source/code changed')
        if any(row['reason'] not in QUALITY_REASONS for row in inputs['exclusions']):
            raise ValueError('An exclusion is not one of the three prospective physical-quality reasons')
        candidates = {(c['source']['match_id'],c['role']):c['source'] for c in config['candidates']}
        passed = {(row['match_id'],row['role']):row for row in inputs['records']}
        excluded = {(row['match_id'],row['role']) for row in inputs['exclusions']}
        if len(candidates) != len(config['candidates']) or len(passed) != len(inputs['records']) or len(excluded) != len(inputs['exclusions']):
            raise ValueError('Duplicate candidate/pass/exclusion match identities')
        if set(passed)&excluded or set(passed)|excluded != set(candidates): raise ValueError('Candidates were replaced or dropped')
        float_sources = {}
        report['float_report_sha256'] = {}
        for name in ('results/development_prepare_pilot.json','results/development_prepare.json','results/fresh_confirmation_float.json'):
            origin = ROOT/name; data = json.loads(origin.read_text())
            if data['status'] != 'passed': raise ValueError('Floating predecessor preparation must pass')
            report['float_report_sha256'][name] = sha256(origin)
            for row in data['records']:
                key = (row['clip_id'],row['split'])
                if key in float_sources: raise ValueError('Duplicate floating predecessor')
                float_sources[key] = row
        for row in inputs['records']:
            source = candidates[(row['match_id'],row['role'])]
            if source['clip_id'] != row['clip_id'] or row['parent_source_sha256'] != source['sha256']:
                raise ValueError('Extended clip is not its registered predecessor')
            if row['source_component_sha256'] != source['source_component_sha256'] or row['source_shard_sha256'] != source['source_shard_sha256']:
                raise ValueError('Component/shard lineage changed')
            predecessor = float_sources[(row['clip_id'],row['role'])]
            if predecessor['match_id'] != row['match_id'] or predecessor['parent_sha256'] != source['sha256']:
                raise ValueError('Floating predecessor has different match/source identity')
            artifacts = [Path(row['path']),Path(source['artifact_path']),Path(predecessor['artifact_path'])]
            expected = [row['sha256'],source['sha256'],predecessor['sha256']]
            if any(sha256(value) != digest for value,digest in zip(artifacts,expected)): raise ValueError('Input artifact hash changed')
            with np.load(artifacts[0],allow_pickle=False) as extended, np.load(artifacts[1],allow_pickle=False) as original, np.load(artifacts[2],allow_pickle=False) as floating:
                shapes = {'frames':(4,24,3,288,512),'actions':(4,24,9),'targets':(4,24,30),
                          'timestamps':(4,24),'video_pts':(4,24),'source_frame_indices':(24,),'player_ids':(4,)}
                arrays = {name:extended[name] for name in shapes}
                for name, shape in shapes.items():
                    if arrays[name].shape != shape or not np.isfinite(arrays[name]).all(): raise ValueError('Invalid extended array schema or nonfinite values')
                if arrays['frames'].dtype != np.float32 or arrays['frames'].min() < -1e-3 or arrays['frames'].max() > 255+1e-3:
                    raise ValueError('Extended frames must preserve float0..255 pixels')
                if arrays['actions'].dtype != np.uint8 or not np.isin(arrays['actions'],[0,1]).all(): raise ValueError('Invalid source action encoding')
                if list(extended['target_names']) != list(TARGET_NAMES) or not np.array_equal(arrays['player_ids'],source['player_ids']): raise ValueError('Target or player identity changed')
                if not np.array_equal(arrays['source_frame_indices'],np.arange(source['plan']['source_start_frame'],source['plan']['source_start_frame']+24)):
                    raise ValueError('Source frame indices changed')
                if not np.allclose(np.diff(arrays['video_pts'],axis=1),.05,rtol=0,atol=1e-6): raise ValueError('Video PTS are not contiguous20FPS')
                origins = np.array([view['calibrated_origin_seconds'] for view in source['timeline_calibration']['views']])
                if not np.array_equal(arrays['timestamps'],origins[:,None]+arrays['source_frame_indices'][None,:]/20): raise ValueError('Clock mapping changed')
                for name in ('frames','actions','targets','timestamps','video_pts'):
                    if not np.array_equal(arrays[name][:,:16],floating[name]): raise ValueError('Original16 FLOAT predecessor changed: '+name)
                for name in ('actions','targets','timestamps','video_pts'):
                    if not np.array_equal(arrays[name][:,:16],original[name]): raise ValueError('Original16 source metadata changed: '+name)
                if not np.array_equal(arrays['source_frame_indices'][:16],floating['source_frame_indices']) or not np.array_equal(arrays['player_ids'],floating['player_ids']):
                    raise ValueError('Floating predecessor source/view indices differ')
            physical = row['physical_audit']
            if physical['frozen_transitions'] or physical['demolished_entity_frames'] or not physical['all_player_identities_checked'] or not physical['all_team_ids_checked']:
                raise ValueError('Recorded physical integrity checks did not pass')
            if any(not np.isfinite(v['mean_ball_residual_uu']) or v['mean_ball_residual_uu']>100 for v in physical['cross_view_ball_alignment']):
                raise ValueError('Recorded cross-view physical-quality gate did not pass')
            report['records'].append({'clip_id':row['clip_id'],'match_id':row['match_id'],'role':row['role'],
                'extended_sha256':row['sha256'],'float_predecessor_sha256':predecessor['sha256'],
                'context16_float_pixels_exact':True,'context16_labels_actions_pts_indices_exact':True})
            save(path,report); print(json.dumps({'audited':len(report['records']),'expected':33}),flush=True)
        if sha256(input_path)!=report['input_report_sha256'] or sha256(config_path)!=report['registration_sha256']: raise ValueError('Frozen inputs changed during audit')
        report.update(status='passed_rollout_input_audit',match_counts=dict(Counter(row['role'] for row in inputs['records'])),
            exclusions_whitelisted=True,exclusion_counts=dict(Counter(row['reason'] for row in inputs['exclusions'])),
            context16_float_pixels_exact=True,no_replacements=True,expected_candidates=len(candidates),passed_inputs=len(passed),
            future_reference_frames_must_be_hidden_from_generator=True,generated_video_ground_truth_established=False,
            completed_utc=datetime.now(timezone.utc).isoformat())
        save(path,report); print(json.dumps({'status':report['status'],'match_counts':report['match_counts']}))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');save(path,report);raise


if __name__=='__main__':main()
