#!/usr/bin/env python3
"""Audit reserved-pilot/full sparse causal fidelity without model execution or fits."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='8'
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
import numpy as np
import torch
from package_development import ROOT, read, require, sha
from audit_sparse_development import numpy_reconstruct, fingerprint, predict

CONDITIONS=('recipient_reconstruction','donor_reconstruction')
MEAN='discovery_mean'
SUMMARY=('standardized_error_gain','gain_difference_from_native_reference','flow_distance_from_native_reference_l2',
         'flow_mse_from_native_reference','proxy_distance_from_native_reference_standardized_rms',
         'descriptor_reconstruction_rmse','endpoint_recipient_source_mse')


def close(a,b,name,*,atol=1e-8,rtol=1e-8):
    require(np.shape(a)==np.shape(b) and np.allclose(a,b,atol=atol,rtol=rtol),name)


def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def descriptor(tile,q):
    return np.concatenate([tile[y*3:(y+1)*3,x*4:(x+1)*4].astype(float).mean((0,1))@q for y in range(3) for x in range(4)])


def lifted(delta,q):
    return np.repeat(np.repeat(delta.reshape(3,4,128)@q.astype(float).T,3,axis=0),4,axis=1)


def flow_metrics(flow,baseline,z,clean,probe,target):
    endpoint=z+np.float32(.5)*flow
    base_endpoint=z+np.float32(.5)*baseline
    views=endpoint[0,7].reshape(4,-1)
    proxies=predict(views.astype(float),probe)
    base=predict(base_endpoint[0,7,:9].reshape(1,-1).astype(float),probe)[0]
    shift=proxies[0]-base
    return {'objective':-float(((proxies[0,2]-target)/probe['y_scale'][2])**2),
            'output_position_proxy':proxies[0].tolist(),'proxy_shift_raw':shift.tolist(),
            'proxy_shift_standardized':(shift/probe['y_scale']).tolist(),
            'flow_delta_l2':norm(flow.astype(float)-baseline),'endpoint_delta_l2':norm(endpoint.astype(float)-base_endpoint),
            'endpoint_recipient_source_mse':float(np.square(endpoint.astype(float)-clean).mean()),
            'view_final_position_proxies':proxies.tolist(),'prediction_sha256':array_hash(flow)}


def compare_metrics(actual,reported,prefix):
    for key,value in actual.items():
        if isinstance(value,str):require(value==reported[key],prefix+'/'+key)
        else:close(value,reported[key],prefix+'/'+key,rtol=1e-5 if key.endswith(('_l2','_mse')) else 1e-8)


def boot(values,matches):
    values=np.asarray(values,dtype=float)
    draws=np.random.default_rng(20260907).integers(0,len(matches),(500,len(matches)))
    return {'mean':float(values.mean()),'match_bootstrap_ci95':np.quantile(values[draws].mean(1),[.025,.975]).tolist(),
            'match_values':dict(zip(matches,values.tolist()))}


def check_summary(actual,reported,name):
    require(set(actual['match_values'])==set(reported['match_values']),name+'/match coverage')
    for key in ('mean','match_bootstrap_ci95'):close(actual[key],reported[key],name+'/'+key,rtol=1e-5,atol=1e-8)
    for match,value in actual['match_values'].items():close(value,reported['match_values'][match],name+'/match value',rtol=1e-5,atol=1e-8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('pilot','full'),required=True)
    p.add_argument('--directory',type=Path,default=ROOT/'results/sparse_causal_fidelity_v1')
    args=p.parse_args();pilot=args.phase=='pilot';started=time.monotonic();torch.set_num_threads(8)
    output=args.directory/('pilot_audit.json' if pilot else 'fidelity_audit.json')
    require(not output.exists(),'Completed fidelity audit is immutable')
    result_path=args.directory/('pilot.json' if pilot else 'fidelity_results.json')
    reg_path=args.directory/'registration.json';reg=read(reg_path);result=read(result_path)
    frozen={}
    def bind(path,expected=None):
        path=Path(path).resolve();actual=sha(path);require(expected is None or actual==expected,'Changed bound artifact '+str(path));frozen[path]=actual;return actual
    bind(Path(__file__));bind(ROOT/'scripts/package_development.py');bind(ROOT/'scripts/audit_sparse_development.py')
    reg_hash=bind(reg_path);result_hash=bind(result_path)
    require(result['status']==('passed' if pilot else 'passed_development_sparse_causal_fidelity') and result['registration_sha256']==reg_hash,
            'Execution has not passed matching protocol')
    require(reg['status']=='registered_before_sparse_causal_execution' and reg['site_index']==16 and reg['descriptor_identity']=='spatial/block_15_output'
            and reg['conditions']==list(CONDITIONS) and reg['control_id']==MEAN and reg['conditions_per_pair']==62 and reg['dictionary_count']==30,
            'Wrong registered experiment')
    for relative,digest in reg['code_sha256'].items():bind(ROOT/relative,digest)
    for entry in reg['causal_bindings'].values():bind(entry['path'],entry['sha256'])
    feature=reg['feature_bindings'];bind(feature['audit']['path'],feature['audit']['sha256']);sparse_audit=read(feature['audit']['path'])
    require(sparse_audit['status']=='passed' and sparse_audit['all_reconstruction_metrics_recomputed'],'Sparse audit prerequisite failed')
    require(bind(ROOT/'scripts/audit_sparse_development.py')==sparse_audit['audit_script_sha256'],'Independent reconstruction code changed')
    for entry in feature['reports']+feature['checkpoints']:bind(entry['path'],entry['sha256'])
    pairset=lambda entries:{(str(Path(e['path']).resolve()),e['sha256']) for e in entries}
    require(pairset(feature['checkpoints'])==pairset(sparse_audit['checkpoints']) and len(feature['checkpoints'])==30,'Checkpoint audit coverage differs')
    original=read(reg['causal_bindings']['registration.json']['path'])
    for entry in original['probe_bindings'].values():bind(entry['path'],entry['sha256'])
    selection=read(original['probe_bindings']['selection']['path'])
    selected=[(i,m) for i,m in enumerate(selection['models']) if m['name']==original['output_probe']]
    require(len(selected)==1 and original['output_probe']==reg['output_probe']=='codec_spatial_X/absolute30/position','Wrong frozen output objective')
    index,metadata=selected[0]
    with np.load(original['probe_bindings']['archive']['path'],allow_pickle=False) as saved:
        probe={key:saved[f'site{index}_main_{key}'] for key in ('x_mean','x_scale','y_mean','y_scale','coefficient')}
    require(fingerprint(metadata['selected_alpha'],probe)==metadata['model_sha256'],'Output model fingerprint differs')
    require(probe['coefficient'].shape==(4608,15),'Output probe dimensions differ')
    q=np.linalg.qr(np.random.default_rng(20260907).standard_normal((2048,128)))[0].astype(np.float32)
    bundles={}
    for item in reg['dictionaries']:
        path=item['checkpoint']['path'];bind(path,item['checkpoint']['sha256'])
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['report']==item['training'],'Registered checkpoint training report differs')
        state={key:value.numpy() for key,value in saved['state_dict'].items()}
        mean,scale=[saved['normalizer'][key].numpy() for key in ('mean','scale')]
        require(hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()==sparse_audit['normalizer_sha256'],'Normalizer differs')
        bundles[item['id']]=(state,mean,scale,saved['report'])
    require(len(bundles)==30 and MEAN not in bundles and pairset([item['checkpoint'] for item in reg['dictionaries']])==pairset(feature['checkpoints']),'Wrong dictionary identity/checkpoint set')
    identifiers=[MEAN,*bundles];conditions={(identifier,condition) for identifier in identifiers for condition in CONDITIONS}
    if pilot:
        require(result['n_pairs_seeds']==1 and result['conditions_checked']==62,'Incomplete pilot')
        original_pilot=read(reg['causal_bindings']['pilot.json']['path'])
        pairs={original['pilot_record']['match_id']:{'recipient':original['pilot_record'],'donor':original['pilot_record']}}
        expected={(next(iter(pairs)),original['paired_noise_seeds'][0])}
        entries=[result['pair']];original_entries=[original_pilot['pair']]
    else:
        prior=read(args.directory/'pilot_audit.json');bind(args.directory/'pilot_audit.json')
        require(prior['status']=='passed' and prior['registration_sha256']==reg_hash and prior['pilot_sha256']==bind(args.directory/'pilot.json'),'Reserved pilot audit differs')
        pairs={p['match_id']:p for p in original['pairs'] if p['split']=='selection'}
        expected={(match,seed) for match in pairs for seed in original['paired_noise_seeds']}
        require(len(expected)==22 and result['n_pairs_seeds']==22 and result['n_matches']==11 and result['conditions_checked']==1364,'Incomplete full fidelity')
        entries=result['pairs'];original_entries=reg['pairs']
    originals={}
    for entry in original_entries:
        bind(entry['path'],entry['sha256']);row=read(entry['path']);originals[row['match_id'],row['seed']]=(entry,row)
    require(set(originals)==expected and len(entries)==len(expected),'Original paired cohort differs')
    rows=[];audited=[];seen=set();max_descriptor_error=0.;max_metric_error=0.
    for entry in entries:
        bind(entry['path'],entry['sha256']);row=read(entry['path']);key=(row['match_id'],row['seed'])
        require(key in expected and key not in seen and row['split']==('pilot' if pilot else 'selection') and row['status']=='passed'
                and row['registration_sha256']==reg_hash,'Duplicate/forbidden/incorrect fidelity pair');seen.add(key)
        source_entry,old=originals[key]
        require(row['source_causal_pair']==source_entry and row['input_sha256']==old['input_sha256'] and row['paired_noise_sha256']==old['paired_noise_sha256'],'Paired input lineage differs')
        for name in ('native_baseline_replay_bitwise_equal','native_donor_replay_bitwise_equal','zero_lift_replay_bitwise_equal'):require(row[name] is True,'Execution replay control failed')
        pair=pairs[key[0]]
        for side in ('recipient','donor'):
            record=pair[side];bind(record['artifact_path'],record['sha256'])
            require(row[side+'_clip_id']==record['clip_id'],'Wrong source clip identity')
            with np.load(record['artifact_path'],allow_pickle=False) as source:
                require(float(source['targets'][0,15,2])==old[side+'_source_ball_z']
                        and float(source['timestamps'][0,15])==old[side+'_target_timestamp']
                        and source['player_ids'].astype(str).tolist()==old['canonical_player_ids'],'Source view/time/entity mismatch')
        bind(old['artifact_path'],old['artifact_sha256']);bind(row['artifact_path'],row['artifact_sha256'])
        reported={(v['dictionary_id'],v['condition']):v for v in row['conditions']}
        require(len(row['conditions'])==62 and set(reported)==conditions,'Missing/duplicate fidelity conditions')
        with np.load(old['artifact_path'],allow_pickle=False) as source,np.load(row['artifact_path'],allow_pickle=False) as saved:
            expected_arrays={'baseline_flow','native_donor_flow','recipient_tile','donor_tile','recipient_z_t','recipient_clean_codec','native_descriptors'}
            expected_arrays|={identifier+'_'+suffix for identifier in identifiers for suffix in ('reconstructed_descriptors','codes','recipient_reconstruction_flow','donor_reconstruction_flow')}
            require(set(saved.files)==expected_arrays,'Unexpected/missing fidelity arrays')
            for key2 in saved.files:
                value=saved[key2]
                require(value.dtype==np.float32 and np.isfinite(value).all(),'Non-FP32 or nonfinite saved fidelity arrays')
            baseline,donor,z,clean=[saved[name] for name in ('baseline_flow','native_donor_flow','recipient_z_t','recipient_clean_codec')]
            require(baseline.shape==donor.shape==z.shape==(1,8,36,16,32) and clean.shape==(4,8,9,16,32),'Flow layout differs')
            for a,b in [('baseline_flow','baseline_flow'),('native_donor_flow','site16_exact_donor_flow'),('recipient_z_t','recipient_z_t'),('recipient_clean_codec','recipient_clean_codec'),('recipient_tile','site16_recipient_tile'),('donor_tile','site16_donor_tile')]:
                require(np.array_equal(saved[a],source[b]),'Native reference replay artifact differs')
            require(array_hash(z)==row['input_sha256']['z_t'],'Saved interpolant hash differs')
            clean_joint=clean.transpose(1,0,2,3,4).reshape(1,8,36,16,32)
            require(np.array_equal(z,.5*clean_joint+.5*source['recipient_noise']),'Original paired noise interpolation differs')
            tiles=[saved['recipient_tile'],saved['donor_tile']];descriptors=saved['native_descriptors']
            independent=np.stack([descriptor(t,q) for t in tiles])
            close(independent,descriptors,'Independent original view/bin descriptor differs',atol=3e-6,rtol=2e-5)
            native_metrics=[flow_metrics(f,baseline,z,clean_joint,probe,old['donor_source_ball_z']) for f in (baseline,donor)]
            for condition,metrics in zip(CONDITIONS,native_metrics):compare_metrics(metrics,row['native_reference_metrics'][condition],'native reference '+condition)
            computed={}
            for identifier in identifiers:
                if identifier==MEAN:
                    mean=next(iter(bundles.values()))[1];recon=np.repeat(mean.astype(np.float32)[None],2,axis=0);codes=np.empty((2,0),np.float32)
                else:
                    state,mean,scale,training=bundles[identifier]
                    normalized=((descriptors.astype(float)-mean)/scale).astype(np.float32)
                    decoded,codes=numpy_reconstruct(normalized,state,training['kind'],training['group_size'],training['active_scalar_budget'])
                    recon=(decoded.astype(float)*scale+mean).astype(np.float32)
                actual_recon=saved[identifier+'_reconstructed_descriptors'];actual_codes=saved[identifier+'_codes']
                require(actual_recon.shape==(2,1536) and actual_codes.shape==codes.shape,'Dictionary output dimensions differ')
                close(recon,actual_recon,'Independent checkpoint reconstruction differs',atol=3e-6,rtol=2e-5)
                close(codes,actual_codes,'Independent checkpoint sparse code differs',atol=3e-6,rtol=2e-5)
                max_descriptor_error=max(max_descriptor_error,float(np.max(np.abs(recon-actual_recon))))
                for i,condition in enumerate(CONDITIONS):
                    measured=reported[identifier,condition];flow=saved[identifier+'_'+condition+'_flow'];reference=(baseline,donor)[i]
                    require(flow.shape==baseline.shape and np.array_equal(flow[:,:7],baseline[:,:7]) and measured['all_earlier_flow_frames_bitwise_unchanged'],'Earlier output changed')
                    require(measured['is_discovery_mean_control']==(identifier==MEAN),'Control identity differs')
                    actual=flow_metrics(flow,baseline,z,clean_joint,probe,old['donor_source_ball_z']);compare_metrics(actual,measured,identifier+'/'+condition)
                    gap=(np.asarray(actual['output_position_proxy'])-native_metrics[i]['output_position_proxy'])/probe['y_scale']
                    difference=flow.astype(float)-reference
                    delta=actual_recon[i].astype(float)-descriptors[i]
                    actual.update(standardized_error_gain=actual['objective']-native_metrics[0]['objective'],
                        gain_difference_from_native_reference=actual['objective']-native_metrics[i]['objective'],
                        flow_distance_from_native_reference_l2=norm(difference),flow_mse_from_native_reference=float(np.square(difference).mean()),
                        endpoint_distance_from_native_reference_l2=norm(difference*.5),
                        proxy_distance_from_native_reference_standardized_rms=float(np.sqrt(np.square(gap).mean())),
                        proxy_difference_from_native_reference_standardized=gap.tolist(),descriptor_reconstruction_rmse=float(np.sqrt(np.square(delta).mean())),
                        lifted_delta_l2=norm(lifted(delta,q)))
                    for name,value in actual.items():
                        if name in ('prediction_sha256','output_position_proxy','proxy_shift_raw','proxy_shift_standardized','flow_delta_l2','endpoint_delta_l2','endpoint_recipient_source_mse','view_final_position_proxies'):continue
                        close(value,measured[name],identifier+'/'+condition+'/'+name,rtol=2e-5 if name.endswith(('_l2','_mse','_rmse')) else 1e-8,atol=1e-7)
                    expected_tile=(tiles[i].astype(float)+lifted(delta,q)).astype(np.float32)
                    roundtrip_error=norm(descriptor(expected_tile,q)-actual_recon[i])
                    require(roundtrip_error<=1e-4*(1+norm(actual_recon[i])) and measured['descriptor_roundtrip_max_abs_error']<=1e-4*(1+norm(actual_recon[i])), 'Lift roundtrip exceeds frozen bound')
                    max_metric_error=max(max_metric_error,abs(actual['standardized_error_gain']-measured['standardized_error_gain']))
                    computed[identifier,condition]=actual
            rows.append({'match_id':row['match_id'],'seed':row['seed'],'values':computed})
        audited.append({'match_id':row['match_id'],'seed':row['seed'],'pair_report':entry,'artifact_path':row['artifact_path'],'artifact_sha256':row['artifact_sha256'],'conditions_checked':62})
        print(json.dumps({'checked_pairs':len(audited),'expected_pairs':len(entries),'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
    require(seen==expected,'Missing fidelity pairs')
    summaries=[];paired=[]
    if not pilot:
        matches=sorted(pairs)
        for identifier in identifiers:
            for condition in CONDITIONS:
                published=[v for v in result['comparisons'] if v['dictionary_id']==identifier and v['condition']==condition]
                require(len(published)==1,'Missing/duplicate published comparison')
                summary={}
                for metric in SUMMARY:
                    value=[np.mean([r['values'][identifier,condition][metric] for r in rows if r['match_id']==match]) for match in matches]
                    summary[metric]=boot(value,matches);check_summary(summary[metric],published[0]['metrics'][metric],identifier+'/'+condition+'/'+metric)
                summaries.append({'dictionary_id':identifier,'condition':condition,'metrics':summary})
                if identifier==MEAN:continue
                published=[v for v in result['paired_discovery_mean_comparisons'] if v['dictionary_id']==identifier and v['condition']==condition]
                require(len(published)==1 and published[0]['control_id']==MEAN,'Missing/duplicate mean-control comparison')
                by_match={name:[] for name in ('donor_objective_gain_above_mean_control','absolute_native_gain_error_improvement_over_mean_control')}
                for match in matches:
                    matchrows=[r['values'] for r in rows if r['match_id']==match];require(len(matchrows)==2,'Unpaired seeds')
                    by_match['donor_objective_gain_above_mean_control'].append(np.mean([v[identifier,condition]['standardized_error_gain']-v[MEAN,condition]['standardized_error_gain'] for v in matchrows]))
                    by_match['absolute_native_gain_error_improvement_over_mean_control'].append(np.mean([abs(v[MEAN,condition]['gain_difference_from_native_reference'])-abs(v[identifier,condition]['gain_difference_from_native_reference']) for v in matchrows]))
                values={name:boot(v,matches) for name,v in by_match.items()}
                for name,value in values.items():check_summary(value,published[0]['metrics'][name],identifier+'/'+condition+'/'+name)
                paired.append({'dictionary_id':identifier,'condition':condition,'control_id':MEAN,'metrics':values})
        require(len(result['comparisons'])==62 and len(result['paired_discovery_mean_comparisons'])==60,'Extra summary comparisons')
    require(all(sha(path)==digest for path,digest in frozen.items()),'Evidence changed during audit')
    audit={'status':'passed','phase':args.phase,'registration_sha256':reg_hash,('pilot_sha256' if pilot else 'fidelity_results_sha256'):result_hash,
           'feature_audit_sha256':feature['audit']['sha256'],'conditions_checked':len(audited)*62,'n_pairs_seeds':len(audited),
           'all_dictionary_reconstructions_recomputed':True,'all_discovery_mean_controls_verified':True,'all_native_reference_arrays_exact':True,
           'all_saved_output_metrics_recomputed':True,'all_earlier_flow_frames_bitwise_unchanged':True,
           'paired_match_summaries_recomputed':not pilot,'source_target_time_player_joins_exact':True,
           'max_descriptor_abs_difference':max_descriptor_error,'max_objective_gain_abs_difference':max_metric_error,
           'lift_verification':'Independent FP64 native+right-inverse reconstruction/norm; execution FP32 roundtrip bounded. Replacement tiles not separately saved.',
           'no_new_model_execution_or_fits':True,'physical_control_established':False,'confirmation_used':False,
           'pairs':audited,'comparisons':summaries,'paired_discovery_mean_comparisons':paired,
           'audit_script_sha256':sha(Path(__file__)),'artifact_bindings':[{'path':str(path),'sha256':digest} for path,digest in frozen.items()],
           'elapsed_seconds':round(time.monotonic()-started,3)}
    output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','output':str(output),'sha256':sha(output),'elapsed_seconds':audit['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
