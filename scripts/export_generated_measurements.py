#!/usr/bin/env python3
"""Export all audited generated-measurement paths without selection or fitting."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='8'
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PATHS=('probe_linear','affine_forward','quadratic_forward','random_norm','wrong_variable_ball_x')
DOSES=(-600.,-300.,0.,300.,600.)
LABELS={'probe_linear':'Probe linear','affine_forward':'Affine map','quadratic_forward':'Quadratic map',
        'random_norm':'Norm-matched random','wrong_variable_ball_x':'Wrong variable: ball x'}
COLORS=('#0072B2','#D55E00','#009E73','#777777','#CC79A7')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def close(a,b,message):
    require(np.shape(a)==np.shape(b) and np.allclose(a,b,rtol=1e-10,atol=1e-8),message)


def csv_write(path,rows):
    require(rows,'Empty CSV table')
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({key:'' if value is None else value for key,value in row.items()})
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream);saved=list(reader)
        require(reader.fieldnames==fields and len(saved)==len(rows),'CSV coverage changed')
        for before,after in zip(rows,saved):
            for field in fields:
                expected=before.get(field);actual=after[field]
                if expected is None:require(actual=='','CSV missing-value differs')
                elif isinstance(expected,(bool,str)):require(actual==str(expected),'CSV literal differs')
                elif isinstance(expected,(int,float,np.number)):require(float(actual)==float(expected),'CSV number differs')
                else:raise ValueError('Nested CSV value not flattened')
    return {'rows':len(rows),'columns':fields,'sha256':sha(path),'bytes':Path(path).stat().st_size}


def flatten(value,prefix='',result=None):
    result={} if result is None else result
    if isinstance(value,dict):
        for key,item in value.items():flatten(item,prefix+'.'+key if prefix else key,result)
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):flatten(item,prefix+'.'+str(i),result)
    else:result[prefix]=value
    return result


def bands(values,draws):
    a=np.asarray(values,float);require(np.isfinite(a).all(),'Nonfinite bootstrap values')
    sampled=a[draws].mean(axis=1)
    return a.mean(axis=0),np.quantile(sampled,.025,axis=0),np.quantile(sampled,.975,axis=0)


def make_tables(report,protocol):
    """Use every already frozen pair; two seeds per match, then equal matches."""
    pairs=report['per_pair'];summary=report['summary'];matches=sorted({p['match_id'] for p in pairs})
    require(len(pairs)==len(matches)*2*len(PATHS) and len(summary)==len(PATHS)*4,'Incomplete path/time comparisons')
    grouped={};pair_keys=set()
    for row in pairs:
        key=(row['match_id'],row['seed'],row['path'])
        require(key not in pair_keys and row['path'] in PATHS,'Duplicate/unknown pair');pair_keys.add(key)
        grouped.setdefault((row['path'],row['match_id']),[]).append(row)
    for path in PATHS:
        for match in matches:
            rows=grouped[path,match]
            require(len(rows)==2 and sorted(r['seed'] for r in rows)==[2026090701,2026090702],'Incorrect paired seeds')
    drawn=np.random.default_rng(protocol['bootstrap_seed']).integers(len(matches),size=(protocol['bootstrap_draws'],len(matches)))
    table={'summary':[flatten(s) for s in summary],'per_pair_time':[],'per_pair_dose':[],
           'per_match_time':[],'per_match_dose':[],'dose_response':[]}
    curve={};by_summary={(s['path'],s['time_index']):s for s in summary}
    require(set(by_summary)=={(path,t) for path in PATHS for t in range(4)},'Missing/duplicate summary')
    for row in pairs:
        for t in range(4):
            ordering=row['ordering_by_time'][t]
            table['per_pair_time'].append({'match_id':row['match_id'],'seed':row['seed'],'path':row['path'],
                'time_index':t,'seconds_after_context':protocol['seconds_after_context'][t],
                **{name:row[name][t] for name in ('view0_extreme_contrast','allviews_extreme_contrast','view0_nontarget_role_rms')},
                **ordering,'any_clipped_dose':row['any_clipped_dose'],'decoded_context_changed':row['decoded_context_changed']})
            for d,dose in enumerate(DOSES):
                table['per_pair_dose'].append({'match_id':row['match_id'],'seed':row['seed'],'path':row['path'],
                    'time_index':t,'seconds_after_context':protocol['seconds_after_context'][t],'requested_dose_uu':dose,
                    'effective_dose_uu':row['effective_doses'][d],'dose_clipped':row['effective_doses'][d]!=dose,
                    'view0_height_change_uu':row['view0_height_changes'][d][t],
                    'allviews_mean_height_change_uu':row['allviews_mean_height_changes'][d][t]})
    for path in PATHS:
        match_curves=[];match_effective=[];all_effective=[]
        for match in matches:
            rows=grouped[path,match]
            heights=np.mean([r['view0_height_changes'] for r in rows],axis=0)
            across=np.mean([r['allviews_mean_height_changes'] for r in rows],axis=0)
            doses=np.asarray([r['effective_doses'] for r in rows],float)
            require(heights.shape==(5,4) and doses.shape==(2,5) and np.isfinite(heights).all() and np.isfinite(doses).all(),'Malformed pair curve')
            match_curves.append(heights);match_effective.append(doses.mean(0));all_effective.append(doses)
            for t in range(4):
                metrics={'match_id':match,'path':path,'time_index':t,'seconds_after_context':protocol['seconds_after_context'][t],
                         'seeds':2,'any_clipped_seed_count':sum(r['any_clipped_dose'] for r in rows),
                         'decoded_context_changed_seed_count':sum(r['decoded_context_changed'] for r in rows)}
                for name in ('view0_extreme_contrast','allviews_extreme_contrast','view0_nontarget_role_rms'):
                    metrics[name]=float(np.mean([r[name][t] for r in rows]))
                for name in ('spearman','slope','positive_adjacent_fraction'):
                    values=[r['ordering_by_time'][t][name] for r in rows if r['ordering_by_time'][t][name] is not None]
                    metrics[name+'_finite_seed_mean']=float(np.mean(values)) if values else None
                    metrics[name+'_defined_seed_count']=len(values)
                metrics['nondecreasing_nonzero_seed_fraction']=float(np.mean([r['ordering_by_time'][t]['nondecreasing_with_nonzero_span'] for r in rows]))
                metrics['inconsistent_duplicate_seed_count']=sum(not r['ordering_by_time'][t]['duplicate_effective_responses_consistent'] for r in rows)
                for control in ('random_norm','wrong_variable_ball_x'):
                    metrics['view0_contrast_minus_'+control]=metrics['view0_extreme_contrast']-float(np.mean([r['view0_extreme_contrast'][t] for r in grouped[control,match]]))
                table['per_match_time'].append(metrics)
                for d,dose in enumerate(DOSES):
                    table['per_match_dose'].append({'match_id':match,'path':path,'time_index':t,
                        'seconds_after_context':protocol['seconds_after_context'][t],'requested_dose_uu':dose,
                        'effective_dose_seed_mean_uu':float(doses[:,d].mean()),'effective_dose_seed_min_uu':float(doses[:,d].min()),
                        'effective_dose_seed_max_uu':float(doses[:,d].max()),'clipped_seed_count':int(np.count_nonzero(doses[:,d]!=dose)),
                        'view0_height_change_seed_mean_uu':float(heights[d,t]),'allviews_height_change_seed_mean_uu':float(across[d,t])})
        values=np.asarray(match_curves);effective=np.asarray(match_effective);raw_doses=np.concatenate(all_effective)
        mean,low,high=bands(values,drawn)
        curve[path]={'mean':mean,'low':low,'high':high,'effective_mean':effective.mean(0),
                     'effective_min':raw_doses.min(0),'effective_max':raw_doses.max(0)}
        for t in range(4):
            close(mean[:,t],by_summary[path,t]['view0_mean_height_by_requested_dose'],'Dose means differ from audited report')
            extreme=values[:,4,t]-values[:,0,t]
            a,b,c=bands(extreme,drawn)
            close(a,by_summary[path,t]['view0_extreme_contrast']['mean'],'Extreme contrast mean differs')
            close([b,c],by_summary[path,t]['view0_extreme_contrast']['match_bootstrap_ci95'],'Extreme contrast interval differs')
            for d,dose in enumerate(DOSES):
                table['dose_response'].append({'path':path,'time_index':t,'seconds_after_context':protocol['seconds_after_context'][t],
                    'requested_dose_uu':dose,'view0_estimated_height_change_mean_uu':float(mean[d,t]),
                    'pointwise_match_bootstrap_low_uu':float(low[d,t]),'pointwise_match_bootstrap_high_uu':float(high[d,t]),
                    'mean_effective_dose_uu':float(effective[:,d].mean()),'minimum_effective_dose_uu':float(raw_doses[:,d].min()),
                    'maximum_effective_dose_uu':float(raw_doses[:,d].max()),'clipped_pair_fraction':float(np.mean(raw_doses[:,d]!=dose)),
                    'matches':len(matches),'paired_seeds_per_match':2})
    return table,curve,matches


def validate_prediction_joins(report,predictions,record_ids,records):
    require(record_ids==[r['record_id'] for r in records],'Saved estimate record order differs')
    lookup={r['record_id']:i for i,r in enumerate(records)}
    require(len(lookup)==len(records) and predictions.shape==(len(records),4,4,12) and np.isfinite(predictions).all(),'Prediction coverage/axes differ')
    baseline_by_pair={(r['match_id'],r['seed']):r for r in records if r['record_id']==r['baseline_record_id']}
    absolute=predictions[...,:6]+predictions[...,6:]
    for pair in report['per_pair']:
        baseline=baseline_by_pair[pair['match_id'],pair['seed']];b=lookup[baseline['record_id']]
        branches=[r for r in records if r['baseline_record_id']==baseline['record_id'] and r['intervention_type']==pair['path']]
        branches.append(baseline);branches.sort(key=lambda r:r['dose'])
        require(len(branches)==5 and [r['dose'] for r in branches]==list(DOSES),'Dose record join differs')
        indices=[lookup[r['record_id']] for r in branches]
        close(pair['effective_doses'],[r['height']['effective_dose'] for r in branches],'Effective doses differ from generation metadata')
        close(pair['view0_height_changes'],absolute[indices,0,:,2]-absolute[b,0,:,2],'Audited response differs from saved predictions')
        close(pair['allviews_mean_height_changes'],absolute[indices,:,:,2].mean(1)-absolute[b,:,:,2].mean(0),'All-view response differs')


def save_figure(fig,folder,stem,outputs):
    for suffix in ('png','pdf'):
        path=folder/(stem+'.'+suffix);require(not path.exists(),'Prior figure is immutable')
        fig.savefig(path,dpi=170)
        if suffix=='png':
            from PIL import Image
            with Image.open(path) as image:image.verify()
        else:require(path.read_bytes().startswith(b'%PDF-'),'PDF header invalid')
        outputs[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
    plt.close(fig)


def lines(ax,x,values,*,bands_values=None):
    for i,path in enumerate(PATHS):
        style='--' if i>=3 else '-'
        ax.plot(x[path] if isinstance(x,dict) else x,values[path],style,marker='o',ms=4,lw=1.5,color=COLORS[i],label=LABELS[path])
        if bands_values is not None:
            lo,hi=bands_values[path]
            ax.fill_between(x[path] if isinstance(x,dict) else x,lo,hi,color=COLORS[i],alpha=.10)
    ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.17);ax.set_axisbelow(True)


def render(report,protocol,curve,matches,folder):
    outputs={};phase=report['phase'];scope=f'{phase.capitalize()}: {len(matches)} matches, two paired seeds per match'
    note='Learned video estimates; generated 3D accuracy is unvalidated. Bands: pointwise match bootstrap, not simultaneous guarantees.'
    for effective in (False,True):
        fig,axes=plt.subplots(2,2,figsize=(12.8,9.2),constrained_layout=True)
        for t,ax in enumerate(axes.flat):
            x={path:curve[path]['effective_mean'] for path in PATHS} if effective else np.asarray(DOSES)
            lines(ax,x,{p:curve[p]['mean'][:,t] for p in PATHS},bands_values={p:(curve[p]['low'][:,t],curve[p]['high'][:,t]) for p in PATHS})
            ax.set_title(f'{protocol["seconds_after_context"][t]:.1f} s after context')
            ax.set_ylabel('Estimated ball height change (uu)')
            ax.set_xlabel('Mean effective dose (uu; fixed requested-dose groups)' if effective else 'Requested dose (uu; clipped within each rollout)')
            if not effective:ax.set_xticks(DOSES)
        axes[0,0].legend(fontsize=8,loc='best')
        fig.suptitle(scope+'\nView 0 dose response: all five registered paths',fontsize=14)
        fig.supxlabel(note,fontsize=9)
        save_figure(fig,folder,'dose_response_effective' if effective else 'dose_response_requested',outputs)
    summaries={(r['path'],r['time_index']):r for r in report['summary']};timepoints=np.asarray(protocol['seconds_after_context'])
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.2),constrained_layout=True)
    specifications=[('view0_extreme_contrast','Estimated high-minus-low dose contrast','Estimated height contrast (uu)'),
                    ('view0_contrast_minus_random_norm','Contrast above norm-matched random','Paired contrast difference (uu)'),
                    ('view0_contrast_minus_wrong_variable_ball_x','Contrast above wrong-variable control','Paired contrast difference (uu)'),
                    ('view0_nontarget_role_rms','Change in 11 non-target role coordinates','Standardized RMS change')]
    for ax,(metric,title,ylabel) in zip(axes.flat,specifications):
        values={p:[summaries[p,t][metric]['mean'] for t in range(4)] for p in PATHS}
        bands_values={p:([summaries[p,t][metric]['match_bootstrap_ci95'][0] for t in range(4)],
                         [summaries[p,t][metric]['match_bootstrap_ci95'][1] for t in range(4)]) for p in PATHS}
        lines(ax,timepoints,values,bands_values=bands_values);ax.set_title(title);ax.set_ylabel(ylabel)
        ax.set_xlabel('Seconds after context');ax.set_xticks(timepoints)
    axes[0,0].legend(fontsize=8,loc='best');fig.suptitle(scope+'\nSigned responses, controls and collateral changes',fontsize=14);fig.supxlabel(note,fontsize=9)
    save_figure(fig,folder,'contrasts_controls_nuisance',outputs)
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.2),constrained_layout=True)
    specs=[('nondecreasing_nonzero_pair_fraction','Nondecreasing response with nonzero span','Fraction of paired runs'),
           ('mean_match_spearman','Rank correlation, equal-weight eligible matches','Mean within-match Spearman'),
           ('undefined_rank_pairs','Undefined rank pairs (including constants)',f'Count out of {2*len(matches)} paired runs'),
           ('inconsistent_duplicate_dose_pairs','Contradictory responses at duplicate doses',f'Count out of {2*len(matches)} paired runs')]
    for ax,(metric,title,ylabel) in zip(axes.flat,specs):
        values={p:[np.nan if summaries[p,t][metric] is None else summaries[p,t][metric] for t in range(4)] for p in PATHS}
        lines(ax,timepoints,values);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xlabel('Seconds after context');ax.set_xticks(timepoints)
        if metric=='nondecreasing_nonzero_pair_fraction':ax.set_ylim(-.03,1.03)
        if metric=='mean_match_spearman':ax.set_ylim(-1.05,1.05)
        if metric.endswith('_pairs'):ax.set_ylim(-.2,max(1,2*len(matches))+.2)
    axes[0,0].legend(fontsize=8,loc='best');fig.suptitle(scope+'\nOrdering and missing/contradictory responses',fontsize=14)
    fig.supxlabel('All paths and negative cases retained. Undefined ranks remain missing; constants never count as successful ordering.',fontsize=9)
    save_figure(fig,folder,'ordering_and_undefined',outputs)
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('selection','confirmation'),required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/generated_evaluation_v1.json')
    p.add_argument('--evaluation-dir',type=Path,default=ROOT/'results/generated_evaluation_v2')
    p.add_argument('--generation-dir',type=Path,default=ROOT/'results/rollout_steering_v2')
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args();started=time.monotonic()
    folder=args.output_dir or args.evaluation_dir/(args.phase+'_figures')
    require(not folder.exists(),'Completed/partial export directories are immutable')
    audit_path=args.evaluation_dir/(args.phase+'_audit.json');audit=read(audit_path)
    require(audit['status']=='passed_generated_measurement_audit','Independent measurement audit must pass before reading results')
    result_path=args.evaluation_dir/(args.phase+'.json');require(sha(result_path)==audit['evaluation_sha256'],'Audited measurement report changed')
    report=read(result_path);protocol=read(args.protocol);frozen={}
    def bind(path,expected=None):
        path=Path(path).resolve();actual=sha(path);require(expected is None or actual==expected,'Input changed: '+str(path));frozen[path]=actual;return actual
    bind(Path(__file__));bind(audit_path);bind(result_path,audit['evaluation_sha256'])
    bind(args.protocol,report['protocol_sha256'])
    require(report['status']=='passed_frozen_generated_video_evaluation' and report['phase']==args.phase
            and report['protocol_sha256']==audit['protocol_sha256'] and audit['all_summary_metrics_independently_recomputed'], 'Completed full measurement phase required')
    require(protocol['paths']==list(PATHS) and protocol['requested_doses']==list(DOSES) and protocol['bootstrap_draws']==1000
            and protocol['bootstrap_seed']==2026090703 and protocol['seconds_after_context']==[.1,.2,.3,.4],'Registered display inputs changed')
    for name,digest in protocol['code_sha256'].items():bind(ROOT/name,digest)
    manifest_path=args.generation_dir/(args.phase+'.json');bind(manifest_path,report['generation_manifest_sha256']);manifest=read(manifest_path)
    generation_audit_path=args.generation_dir/(args.phase+'_audit.json');bind(generation_audit_path,report['generation_audit_sha256'])
    generation_audit=read(generation_audit_path)
    require(generation_audit['status']=='passed_rollout_generation_audit' and generation_audit['manifest_sha256']==sha(manifest_path),'Generation source gate differs')
    artifact=report['prediction_artifact'];bind(artifact['path'],artifact['sha256'])
    require(artifact['sha256']==audit['prediction_artifact_sha256'],'Prediction artifact audit differs')
    with np.load(artifact['path'],allow_pickle=False) as saved:
        predictions=saved['role_predictions'];record_ids=saved['record_ids'].astype(str).tolist()
        close(saved['absolute_ball_predictions'],predictions[...,:6]+predictions[...,6:],'Saved absolute-ball estimate differs')
    validate_prediction_joins(report,predictions,record_ids,manifest['records'])
    table,curve,matches=make_tables(report,protocol)
    require(len(matches)=={'selection':10,'confirmation':22}[args.phase] and report['matches']==len(matches),'Incomplete registered phase')
    folder.mkdir(parents=True);csv_outputs={}
    for name,rows in table.items():csv_outputs[name+'.csv']=csv_write(folder/(name+'.csv'),rows)
    figures=render(report,protocol,curve,matches,folder)
    require(all(sha(path)==digest for path,digest in frozen.items()),'Bound inputs changed during export')
    exported={'status':'passed_audited_generated_measurement_export','phase':args.phase,'measurement_audit_sha256':sha(audit_path),
              'evaluation_sha256':sha(result_path),'protocol_sha256':report['protocol_sha256'],'generation_manifest_sha256':sha(manifest_path),
              'prediction_artifact_sha256':artifact['sha256'],'all_five_paths_four_times_retained':True,
              'requested_dose_grouping_unchanged':True,'effective_dose_axis':'Mean effective dose within each fixed requested-dose group; no regrouping or dose fitting',
              'uncertainty':'Registered1000 whole-match bootstrap draws after paired-seed averaging; per-dose bands are descriptive pointwise display, not simultaneous inference',
              'negative_cases_omitted':False,'undefined_ranks_imputed':False,'model_or_probe_fitting_performed':False,
              'generated_video_accuracy_established':False,'physical_control_established':False,'csv':csv_outputs,'figures':figures,
              'script_sha256':sha(Path(__file__)),'elapsed_seconds':round(time.monotonic()-started,3),
              'artifact_bindings':[{'path':str(path),'sha256':digest} for path,digest in frozen.items()]}
    (folder/'export.json').write_text(json.dumps(exported,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':exported['status'],'folder':str(folder),'elapsed_seconds':exported['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
