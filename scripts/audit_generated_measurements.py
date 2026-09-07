#!/usr/bin/env python3
"""Independent cache, frozen-evaluator and generated-response metric audit."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='8'
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
import numpy as np
import torch
from package_development import ROOT, read, require, sha

PATHS=('probe_linear','affine_forward','quadratic_forward','random_norm','wrong_variable_ball_x')
DOSES=(-600.,-300.,0.,300.,600.)


def equal(actual,expected,path='metric'):
    if isinstance(actual,dict):
        require(isinstance(expected,dict) and set(actual)==set(expected),'Dictionary schema differs: '+path)
        for key in actual:equal(actual[key],expected[key],path+'/'+key)
    elif isinstance(actual,(list,tuple)):
        require(isinstance(expected,(list,tuple)) and len(actual)==len(expected),'Sequence schema differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):equal(a,b,path+'/'+str(i))
    elif actual is None or isinstance(actual,(str,bool)):
        require(actual==expected,'Value differs: '+path)
    else:
        a,b=np.asarray(actual),np.asarray(expected)
        if a.dtype.kind in 'OUS':require(actual==expected,'List differs: '+path)
        else:require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                     and np.allclose(a,b,rtol=1e-10,atol=1e-8),'Numerical value differs: '+path)


def ranks(values):
    return np.array([np.count_nonzero(values<value)+(np.count_nonzero(values==value)-1)/2 for value in values],float)


def ordering_independent(doses,response):
    x,y=np.asarray(doses,float),np.asarray(response,float)
    require(x.shape==y.shape==(5,) and np.isfinite(x).all() and np.isfinite(y).all() and np.all(x[1:]>=x[:-1]),'Bad dose curve')
    unique=np.unique(x)
    consistent=all(float(y[x==value].max()-y[x==value].min())<=1e-6 for value in unique)
    result={'distinct_effective_doses':len(unique),'duplicate_effective_responses_consistent':bool(consistent)}
    if len(unique)==1:
        return {**result,'spearman':None,'slope':None,'positive_adjacent_fraction':None,'nondecreasing_with_nonzero_span':False}
    dx,dy=ranks(x),ranks(y);dx-=dx.mean();dy-=dy.mean()
    correlation=None if np.linalg.norm(dy)==0 else float((dx@dy)/(np.linalg.norm(dx)*np.linalg.norm(dy)))
    xx=x-x.mean();yy=y-y.mean();changes=[y[i+1]-y[i] for i in range(4) if x[i+1]>x[i]]
    return {**result,'spearman':correlation,'slope':float((xx@yy)/(xx@xx)),
            'positive_adjacent_fraction':float(sum(value>0 for value in changes)/len(changes)),
            'nondecreasing_with_nonzero_span':bool(consistent and min(changes)>=-1e-6 and y[4]-y[0]>1e-3)}


def bootstrap(values,draws):
    a=np.asarray(values,float)
    return {'mean':float(a.mean()),'match_bootstrap_ci95':np.quantile(np.mean(a[draws],axis=1),[.025,.975]).tolist()}


def independent_summary(records,predictions,scale,protocol):
    indices={r['record_id']:i for i,r in enumerate(records)}
    baseline_rows=[r for r in records if r['record_id']==r['baseline_record_id']]
    ball=predictions[...,:6]+predictions[...,6:]
    pairs=[]
    for baseline in baseline_rows:
        bi=indices[baseline['record_id']]
        for path in PATHS:
            branches=[r for r in records if r['baseline_record_id']==baseline['record_id'] and r['intervention_type']==path]
            require(len(branches)==4 and {r['dose'] for r in branches}==set(DOSES)-{0.},'Incomplete per-path dose grid')
            branches.append(baseline);branches.sort(key=lambda r:r['dose'])
            row_indices=[indices[r['record_id']] for r in branches]
            effective=[float(r['height']['effective_dose']) for r in branches]
            require(effective[2]==0.,'Baseline is not zero effective dose')
            delta=predictions[row_indices]-predictions[bi]
            primary=ball[row_indices,0,:,2]-ball[bi,0,:,2]
            allviews=(ball[row_indices,:,:,2]-ball[bi,:,:,2]).mean(axis=1)
            collateral=[]
            for t in range(4):
                values=[float(delta[extreme,0,t,coordinate]/scale[coordinate])**2
                        for extreme in (0,4) for coordinate in range(12) if coordinate!=8]
                collateral.append(float(np.sqrt(sum(values)/22)))
            pairs.append({'match_id':baseline['match_id'],'seed':baseline['seed'],'path':path,'effective_doses':effective,
                'view0_height_changes':primary.tolist(),'allviews_mean_height_changes':allviews.tolist(),
                'view0_extreme_contrast':(primary[4]-primary[0]).tolist(),
                'allviews_extreme_contrast':(allviews[4]-allviews[0]).tolist(),'view0_nontarget_role_rms':collateral,
                'ordering_by_time':[ordering_independent(effective,primary[:,t]) for t in range(4)],
                'any_clipped_dose':any(r['height']['target_clipped'] for r in branches),
                'decoded_context_changed':any(not r['decoded_context_bitwise_equal_to_baseline'] for r in branches)})
    matches=sorted({p['match_id'] for p in pairs});draws=np.random.default_rng(protocol['bootstrap_seed']).integers(len(matches),size=(protocol['bootstrap_draws'],len(matches)))
    summaries=[]
    for path in PATHS:
        group={match:[p for p in pairs if p['match_id']==match and p['path']==path] for match in matches}
        require(all(len(value)==2 and len({r['seed'] for r in value})==2 for value in group.values()),'Incomplete paired seeds')
        for t in range(4):
            summary={'path':path,'time_index':t,'seconds_after_context':protocol['seconds_after_context'][t],'matches':len(matches)}
            for metric in ('view0_extreme_contrast','allviews_extreme_contrast','view0_nontarget_role_rms'):
                values=[sum(p[metric][t] for p in group[match])/2 for match in matches]
                summary[metric]=bootstrap(values,draws)
                if metric=='view0_extreme_contrast':summary['positive_contrast_matches']=sum(value>0 for value in values)
            dose_means=[sum(sum(p['view0_height_changes'][d][t] for p in group[match])/2 for match in matches)/len(matches) for d in range(5)]
            summary['view0_mean_height_by_requested_dose']=dose_means
            local=[p for match in matches for p in group[match]]
            summary['nondecreasing_nonzero_pair_fraction']=sum(p['ordering_by_time'][t]['nondecreasing_with_nonzero_span'] for p in local)/len(local)
            by_match=[];undefined=0
            for match in matches:
                values=[p['ordering_by_time'][t]['spearman'] for p in group[match]]
                finite=[value for value in values if value is not None];undefined+=len(values)-len(finite)
                if finite:by_match.append(sum(finite)/len(finite))
            summary.update(mean_match_spearman=sum(by_match)/len(by_match) if by_match else None,
                           eligible_rank_matches=len(by_match),undefined_rank_matches=len(matches)-len(by_match),
                           undefined_rank_pairs=undefined,
                           inconsistent_duplicate_dose_pairs=sum(not p['ordering_by_time'][t]['duplicate_effective_responses_consistent'] for p in local))
            for control in ('random_norm','wrong_variable_ball_x'):
                differences=[]
                for match in matches:
                    controls=[p for p in pairs if p['match_id']==match and p['path']==control]
                    require(len(controls)==2,'Missing paired control')
                    differences.append(sum(p['view0_extreme_contrast'][t] for p in group[match])/2-sum(p['view0_extreme_contrast'][t] for p in controls)/2)
                summary['view0_contrast_minus_'+control]=bootstrap(differences,draws)
            summaries.append(summary)
    return {'per_pair':pairs,'summary':summaries,'matches':len(matches),'paired_seeds_per_match':2}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('pilot','selection','confirmation'),required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/generated_evaluation_v1.json')
    p.add_argument('--generation-dir',type=Path,default=ROOT/'results/rollout_steering_v1')
    p.add_argument('--evaluation-dir',type=Path,default=ROOT/'results/generated_evaluation_v1')
    p.add_argument('--cache-dir',type=Path,default=Path('/data2/ishaangp/mira-interp/generated_evaluation_v1'))
    args=p.parse_args();started=time.monotonic();out=args.evaluation_dir/(args.phase+'_audit.json')
    require(not out.exists(),'Completed measurement audit is immutable');frozen={};torch.set_num_threads(8)
    def bind(path,digest=None):
        path=Path(path).resolve();actual=sha(path);require(digest is None or digest==actual,'Changed input '+str(path));frozen[path]=actual;return actual
    bind(Path(__file__));bind(ROOT/'scripts/package_development.py')
    config_hash=bind(args.protocol);config=read(args.protocol)
    require(config['status']=='registered_before_generated_video_evaluation' and config['paths']==list(PATHS) and config['requested_doses']==list(DOSES)
            and config['endpoint_frames']==[17,19,21,23] and config['seconds_after_context']==[.1,.2,.3,.4]
            and config['bootstrap_draws']==1000 and config['bootstrap_seed']==2026090703,'Unregistered measurement choices')
    for name,digest in config['code_sha256'].items():bind(ROOT/name,digest)
    review_path=ROOT/'results/generated_evaluation_code_review.json';review=read(review_path);bind(review_path)
    require(review['status']=='passed','Independent metric source review failed')
    for name,digest in review['code_sha256'].items():bind(ROOT/name,digest)
    evaluator_path=ROOT/'results/videomae_finetune_v1.json';evaluator_audit_path=ROOT/'results/videomae_finetune_v1_audit.json'
    bind(evaluator_path,config['evaluator_report_sha256']);bind(evaluator_audit_path,config['evaluator_audit_sha256'])
    evaluator,evaluator_audit=read(evaluator_path),read(evaluator_audit_path)
    require(evaluator['status']=='passed_development_partial_finetune' and evaluator['selected_step']==500
            and evaluator_audit['status']=='passed_development_video_evaluator_audit' and evaluator_audit['analysis_sha256']==sha(evaluator_path),'Frozen evaluator calibration gate failed')
    checkpoint=evaluator['checkpoint'];bind(checkpoint['path'],checkpoint['sha256'])
    require(checkpoint['sha256']==config['checkpoint_sha256'],'Wrong evaluator checkpoint')
    state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
    expected_scale=state['head.y_scale'].float().numpy().astype(float)
    require(expected_scale.shape==(12,) and np.isfinite(expected_scale).all() and (expected_scale>0).all(),'Bad calibrated target scales')
    del state
    result_path=args.evaluation_dir/(args.phase+'.json');result=read(result_path);result_hash=bind(result_path)
    manifest_path=args.generation_dir/(args.phase+'.json');manifest=read(manifest_path)
    generation_audit_path=args.generation_dir/(args.phase+'_audit.json');generation_audit=read(generation_audit_path)
    manifest_hash=bind(manifest_path,result['generation_manifest_sha256']);bind(generation_audit_path,result['generation_audit_sha256'])
    require(result['status']=='passed_frozen_generated_video_evaluation' and result['phase']==args.phase and result['protocol_sha256']==config_hash
            and generation_audit['status']=='passed_rollout_generation_audit' and generation_audit['manifest_sha256']==manifest_hash,'Measurement/generation gate differs')
    require(result['evaluator']['checkpoint_sha256']==checkpoint['sha256'] and result['evaluator']['analysis_sha256']==sha(evaluator_path)
            and result['evaluator']['window_end_frames']==config['endpoint_frames'],'Evaluator provenance differs')
    if args.phase!='pilot':
        previous='pilot' if args.phase=='selection' else 'selection'
        previous_path=args.evaluation_dir/(previous+'.json');previous_audit_path=args.evaluation_dir/(previous+'_audit.json')
        previous_audit=read(previous_audit_path);bind(previous_audit_path)
        require(previous_audit['status']=='passed_generated_measurement_audit' and previous_audit['evaluation_sha256']==bind(previous_path),'Previous measurement phase not independently complete')
    records=manifest['records'];require(len(records)==result['records'] and len({r['record_id'] for r in records})==len(records),'Duplicate/missing generated records')
    lookup={r['record_id']:r for r in records};signatures=set()
    for row in records:
        require(row['split']==args.phase and row['baseline_record_id'] in lookup,'Forbidden role or missing baseline')
        baseline=lookup[row['baseline_record_id']]
        require(baseline['dose']==0 and baseline['baseline_record_id']==baseline['record_id']
                and all(row[name]==baseline[name] for name in ('match_id','clip_id','split','seed')),'Unpaired baseline')
        signature=tuple(row[name] for name in ('match_id','clip_id','split','seed','intervention_type','dose'))
        require(signature not in signatures,'Duplicate condition');signatures.add(signature)
        require(row['status']=='passed_generation','Incomplete generated condition')
        # The independent generation audit already verifies large video tensors;
        # recheck their hashes before and after joining model-estimate caches.
        bind(row['generated_artifact_path'],row['generated_sha256'])
    artifact=result['prediction_artifact'];bind(artifact['path'],artifact['sha256'])
    with np.load(artifact['path'],allow_pickle=False) as saved:
        require(set(saved.files)=={'record_ids','role_predictions','absolute_ball_predictions','role_target_scale'},'Prediction artifact schema differs')
        require(saved['record_ids'].astype(str).tolist()==[r['record_id'] for r in records],'Prediction row identity/order differs')
        prediction=saved['role_predictions'];scale=saved['role_target_scale']
        require(prediction.shape==(len(records),4,4,12) and prediction.dtype==np.float64 and np.isfinite(prediction).all(),'Incomplete prediction axes or dtype')
        require(np.array_equal(scale,expected_scale),'Frozen normalization scale changed')
        require(np.array_equal(saved['absolute_ball_predictions'],prediction[...,:6]+prediction[...,6:]),'Absolute ball conversion differs')
    cache_records=[]
    for i,row in enumerate(records):
        cache=args.cache_dir/args.phase/(row['record_id']+'.npz');cache_hash=bind(cache)
        with np.load(cache,allow_pickle=False) as saved:
            require(set(saved.files)=={'role_predictions','generated_sha256','protocol_sha256'} and str(saved['generated_sha256'])==row['generated_sha256']
                    and str(saved['protocol_sha256'])==config_hash and np.array_equal(saved['role_predictions'],prediction[i]),'Per-record estimate cache differs')
        cache_records.append({'record_id':row['record_id'],'path':str(cache),'sha256':cache_hash,'generated_sha256':row['generated_sha256']})
    require(result['any_context_pixels_changed']==(not manifest['all_context_pixels_unchanged']),'Context-change annotation differs')
    computed=None
    if args.phase=='pilot':
        require(len(records)==2 and len({r['match_id'] for r in records})==1 and result['engineering_only']
                and result['repeated_predictions_bitwise_equal'],'Two-condition repeat-verified pilot required')
    else:
        require(len(records)=={'selection':420,'confirmation':924}[args.phase],'Incomplete registered generated dose cohort')
        computed=independent_summary(records,prediction,scale,config)
        for key,value in computed.items():
            if isinstance(value,list):
                require(len(value)==len(result[key]),'Missing reported row '+key)
                for i,(actual,expected) in enumerate(zip(value,result[key])):equal(actual,expected,key+'/'+str(i))
            else:equal(value,result[key],key)
    require(all(sha(path)==digest for path,digest in frozen.items()),'Evidence changed during measurement audit')
    report={'status':'passed_generated_measurement_audit','phase':args.phase,'evaluation_sha256':result_hash,'protocol_sha256':config_hash,
            'generation_manifest_sha256':manifest_hash,'generation_audit_sha256':sha(generation_audit_path),
            'prediction_artifact_sha256':artifact['sha256'],'checkpoint_sha256':checkpoint['sha256'],
            'all_cache_predictions_exact':True,'all_record_source_seed_joins_exact':True,'all_absolute_ball_coordinates_recomputed':True,
            'frozen_calibration_scale_exact':True,'all_summary_metrics_independently_recomputed':computed is not None,
            'new_model_execution_or_adaptation':False,'generated_video_accuracy_established':False,'physical_control_established':False,
            'pilot_repeat_verification':'Attested by frozen producer execution and its recorded equality flag; repeated prediction arrays are not separately retained or independently rerun.',
            'records_checked':len(records),'cache_records':cache_records,'matches':len({r['match_id'] for r in records}),
            'audit_script_sha256':sha(Path(__file__)),'elapsed_seconds':round(time.monotonic()-started,3),
            'artifact_bindings':[{'path':str(path),'sha256':digest} for path,digest in frozen.items()]}
    write=json.dumps(report,indent=2,allow_nan=False)+'\n';out.write_text(write)
    print(json.dumps({'status':report['status'],'output':str(out),'sha256':sha(out),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
