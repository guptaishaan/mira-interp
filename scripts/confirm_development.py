#!/usr/bin/env python3
"""Audit and evaluate fixed revised readouts on the newly reserved matches."""
import os
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['OMP_NUM_THREADS']='8'
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from mira_interp.model_loading import sha256
from mira_interp.probes import load_selected_models,match_bootstrap_metrics,paired_mse_gain
from mira_interp.development import role_targets,causal_row_indices,temporal_features
from prepare_development import save
from aggregate_captures import verify_loading

def main():
    lock_path=ROOT/'configs/fresh_evaluation_v3.json'
    lock=json.loads(lock_path.read_text())
    if any(sha256(ROOT/p)!=h for group in ('capture_code_sha256','evaluation_code_sha256') for p,h in lock[group].items()):
        raise ValueError('Capture or evaluation code changed after registration')
    d=ROOT/'results/development_probes_v3'
    if lock['status']!='frozen_before_fresh_capture' or sha256(d/'development_selection.json')!=lock['development_selection_sha256'] or sha256(d/'development_models.npz')!=lock['model_archive_sha256']:
        raise ValueError('Frozen choice/model integrity failed')
    output=ROOT/'results/fresh_confirmation_v3'
    if (output/'confirmation.json').exists():
        raise ValueError('Fresh confirmation already evaluated; do not overwrite')
    qpath=ROOT/'data/qualified_fresh_confirmation_clip_manifest.json'
    qualified=json.loads(qpath.read_text())
    if qualified['registered_manifest_sha256']!=lock['candidate_manifest_sha256'] or qualified['status']!='passed':
        raise ValueError('Reserved cohort lineage failed')
    original={r['clip_id']:r for r in qualified['records']}
    expected_matches=set(qualified['qualified_matches'])
    old_matches={r['match_id'] for r in json.loads((ROOT/'data/qualified_split_manifest.json').read_text())['matches']}
    if expected_matches&old_matches:
        raise ValueError('Old and fresh matches overlap')
    prep_path=ROOT/'results/fresh_confirmation_float.json'
    prep=json.loads(prep_path.read_text())
    if prep['status']!='passed' or prep['qualified_manifest_sha256']!=sha256(qpath):
        raise ValueError('Source preparation incomplete')
    inputs={r['clip_id']:r for r in prep['records']}
    worker_paths=sorted((ROOT/'results').glob('fresh_capture_worker*.json'))
    worker_records=[]
    for worker_path in worker_paths:
        worker=json.loads(worker_path.read_text())
        if worker['status']!='passed' or worker['registration_sha256']!=sha256(lock_path) or worker['preparation_sha256']!=sha256(prep_path) or worker['code_sha256']!=lock['capture_code_sha256'] or worker['errors']:
            raise ValueError('Fresh workers have not completed matching execution')
        loading=worker['model_loading']
        verify_loading(loading)
        if not loading['world_model_strict_load']['passed'] or not loading['codec_strict_load']['passed'] or not loading['codec_pair_integrity']['passed']:
            raise ValueError('Strict pretrained model loading failed')
        if loading['world_model_strict_load']['state_tensors']!=1183 or loading['codec_strict_load']['state_tensors']!=936:
            raise ValueError('Unexpected checkpoint architecture')
        worker_records.extend(worker['records'])
    if len(worker_records)!=len(original) or {r['clip_id'] for r in worker_records}!=set(original):
        raise ValueError('Complete worker records required')
    paths=sorted((Path('/data2/ishaangp/mira-interp/captures/fresh_confirmation_v3')).glob('*.json'))
    sides=[json.loads(p.read_text()) for p in paths]
    if {r['clip_id'] for r in sides}!=set(original) or len(sides)!=len(original):
        raise ValueError('Fresh capture incomplete or duplicated')
    pieces={};checked=[]
    shapes={'mean_X':(32,17,2048),'spatial_X':(32,17,1536),'codec_spatial_X':(32,4608),'codec_mean_X':(32,32),'RGB_X':(32,1536),'y':(32,30)}
    for side in sorted(sides,key=lambda r:(r['match_id'],r['clip_id'])):
        cid=side['clip_id'];src=original[cid];inp=inputs[cid];path=Path(side['path'])
        if side['match_id']!=src['match_id'] or side['split']!='confirmation' or side['registration_sha256']!=sha256(lock_path) or side['code_sha256']!=lock['capture_code_sha256']:
            raise ValueError('Capture registration or identity mismatch')
        if sha256(path)!=side['sha256'] or side['input_sha256']!=inp['sha256'] or sha256(Path(inp['artifact_path']))!=inp['sha256']:
            raise ValueError('Capture/input bytes changed')
        if sha256(Path(src['artifact_path']))!=src['sha256'] or inp['parent_sha256']!=src['sha256']:
            raise ValueError('Source lineage failed')
        with np.load(path,allow_pickle=False) as z,np.load(src['artifact_path'],allow_pickle=False) as raw:
            if z['split'].tolist()!=['confirmation']*32 or z['match_ids'].tolist()!=[src['match_id']]*32 or z['clip_ids'].tolist()!=[cid]*32:
                raise ValueError('Row identities changed')
            a={k:z[k] for k in z.files}
            if a['sites'].tolist()!=['block_0_input']+[f'block_{i}_output' for i in range(16)] or not np.array_equal(a['target_names'],raw['target_names']):
                raise ValueError('Site/target column identities changed')
            exact={'y':raw['targets'][:,1::2].reshape(32,30),'timestamps':raw['timestamps'][:,1::2].reshape(32),
                   'source_frame_index':np.tile(raw['source_frame_indices'][1::2],4),'canonical_player_ids':np.tile(raw['player_ids'],(32,1)),
                   'view_index':np.repeat(np.arange(4),8),'latent_frame_index':np.tile(np.arange(8),4)}
            if any(not np.array_equal(a[k],v) for k,v in exact.items()):
                raise ValueError('Fresh source labels/timing/identity not exact')
        for k,shape in shapes.items():
            if a[k].shape!=shape or not np.isfinite(a[k]).all() or a[k].dtype!=(np.float64 if k=='y' else np.float16):
                raise ValueError('Invalid feature/label arrays')
        for k in (*shapes,'timestamps','view_index','latent_frame_index','match_ids','clip_ids','split'):
            pieces.setdefault(k,[]).append(a[k])
        checked.append(dict(clip_id=cid,sha256=side['sha256']))
    data={k:np.concatenate(v) for k,v in pieces.items()}
    current,previous=causal_row_indices(data['clip_ids'],data['view_index'],data['latent_frame_index'])
    y=np.concatenate((data['y'],role_targets(data['y'],data['view_index'])),axis=1)[current]
    ids=data['match_ids'][current]
    if len(np.unique(ids))!=qualified['n_matches'] or len(ids)!=qualified['n_matches']*192:
        raise ValueError('Fresh evaluation rows are incomplete')
    audit=dict(status='passed',scope='fresh confirmation capture integrity only',registered_choice_sha256=sha256(lock_path),
               qualified_manifest_sha256=sha256(qpath),n_matches=qualified['n_matches'],n_clips=len(checked),rows=len(ids),
               every_source_label_and_identity_join_exact=True,all_output_hashes_checked=True,old_match_overlap=False,checked=checked)
    save(output/'capture_audit.json',audit)
    selection=json.loads((d/'development_selection.json').read_text())
    models={m.name:m for m in load_selected_models(d/'development_models.npz',selection)}
    specs={m['name']:m for m in selection['models']}
    chosen_names=set(n for groups in lock['choices'].values() for choices in groups.values() for n in choices.values())
    predictions={};metrics={}
    for name in sorted(chosen_names):
        spec=specs[name]['feature_spec'];full=data[spec['array']]
        if spec['site_index'] is not None:
            full=full[:,spec['site_index'],:]
        x=temporal_features(full,current,previous) if spec['temporal']=='current_and_delta' else full[current]
        target=y[:,specs[name]['joined_target_indices']];model=models[name]
        pred=model.main.predict(x);control=model.shuffled.predict(x)
        predictions[name]=pred
        metrics[name]=dict(target_names=specs[name]['target_names'],feature_spec=spec,
                           main=match_bootstrap_metrics(target,pred,ids,model.main.y_scale,seed=20260907,replicates=500),
                           shuffled=match_bootstrap_metrics(target,control,ids,model.main.y_scale,seed=20260907,replicates=500),
                           mean_baseline=match_bootstrap_metrics(target,np.broadcast_to(model.main.y_mean,target.shape),ids,model.main.y_scale,seed=20260907,replicates=500))
        metrics[name]['gain_vs_shuffled']=paired_mse_gain(target,pred,control,ids,model.main.y_scale,seed=20260907)
    comparisons={}
    for definition,groups in lock['choices'].items():
        comparisons[definition]={}
        for group,choices in groups.items():
            primary=choices['primary'];model=models[primary].main
            target=y[:,specs[primary]['joined_target_indices']]
            compare={label:paired_mse_gain(target,predictions[primary],predictions[name],ids,model.y_scale,seed=20260907) for label,name in choices.items() if label!='primary'}
            compare['mean_baseline']=paired_mse_gain(target,predictions[primary],np.broadcast_to(model.y_mean,target.shape),ids,model.y_scale,seed=20260907)
            compare['spatial_current_vs_mean_current']=paired_mse_gain(target,predictions[choices['spatial_current']],predictions[choices['mean_current']],ids,model.y_scale,seed=20260907)
            comparisons[definition][group]=dict(choices=choices,paired_gains=compare)
    result=dict(status='passed_fresh_observational_confirmation',evaluated_utc=datetime.now(timezone.utc).isoformat(),
                registered_choices_sha256=sha256(lock_path),development_selection_sha256=lock['development_selection_sha256'],model_archive_sha256=lock['model_archive_sha256'],
                capture_audit_sha256=sha256(output/'capture_audit.json'),n_matches=qualified['n_matches'],n_rows=len(ids),
                models_refitted=False,new_choices_from_confirmation=False,physical_control_established=False,
                metrics=metrics,comparisons=comparisons,script_sha256=sha256(Path(__file__)),
                limits=['Teacher-forced target pixels remain present; decoding is not predictive rollout accuracy.',
                        'Quality-qualified official-dev matches; not a guarantee that the pretrained model never used official dev for validation.',
                        '500 paired match-bootstrap intervals are descriptive; no family-wide significance or training-seed uncertainty claim.'])
    if sha256(lock_path)!=audit['registered_choice_sha256'] or sha256(d/'development_models.npz')!=lock['model_archive_sha256'] or sha256(d/'development_selection.json')!=lock['development_selection_sha256']:
        raise ValueError('Fixed choices changed during evaluation')
    if any(sha256(Path(side['path']))!=side['sha256'] for side in sides) or any(sha256(ROOT/p)!=h for p,h in lock['evaluation_code_sha256'].items()):
        raise ValueError('Confirmation inputs/code changed during evaluation')
    save(output/'confirmation.json',result)
    print(json.dumps({'status':result['status'],'n_matches':result['n_matches'],'n_rows':len(ids)}))

if __name__=='__main__':
    main()
