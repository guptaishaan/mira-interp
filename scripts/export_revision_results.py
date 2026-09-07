#!/usr/bin/env python3
"""Format existing development/fresh-confirmation results without fitting."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def development():
    report=json.loads((ROOT/'results/development_probes_v3/development_report.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ri,definition in enumerate(('absolute30','role12')):
        for ci,group in enumerate(('all','position','velocity')):
            ax=axes[ri,ci]
            for prefix,color in [('mean/','#2469a0'),('spatial/','#c76724')]:
                rows=[r for r in report['stage1'] if r['name'].startswith(prefix)]
                loss=[r['targets'][definition]['groups'][group]['main']['selection']['normalized_mse'] for r in rows]
                ax.plot(np.arange(-1,16),loss,'o-',color=color,ms=3,label=prefix[:-1])
            for key,label,color,style in [('codec_mean_X','Codec mean','#aaa','--'),('codec_spatial_X','Codec full grid','#555','-.'),('RGB_X','RGB grid','#8b64a0',':')]:
                row=next(r for r in report['stage1'] if r['name']==key)
                ax.axhline(row['targets'][definition]['groups'][group]['main']['selection']['normalized_mse'],color=color,ls=style,lw=1.2,label=label)
            ax.set(title=f'{"Absolute state" if ri==0 else "Ego and ball-relative state"}: {group}',xlabel='Block (−1 = input)',ylabel='Standardized selection MSE')
            if ri==0 and ci==0:
                ax.legend(fontsize=8)
    fig.suptitle('Development comparison — choices use these selection matches',fontsize=14)
    for ext in ('png','pdf'):
        fig.savefig(ROOT/f'figures/development_readout_comparison.{ext}',dpi=180)
    plt.close(fig)

def confirmation():
    folder=ROOT/'results/fresh_confirmation_v3'
    report=json.loads((folder/'confirmation.json').read_text())
    records=[]
    for name,row in report['metrics'].items():
        for variant in ('main','shuffled','mean_baseline'):
            m=row[variant]
            for i,target in enumerate(row['target_names']):
                records.append(dict(model=name,variant=variant,target=target,r2=m['r2'][i],mae=m['mae'][i],rmse=m['rmse'][i]))
    with (folder/'target_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,group in zip(axes,('position','velocity')):
        name=report['comparisons']['role12'][group]['choices']['primary']
        item=report['metrics'][name];m=item['main']
        vals=np.asarray(m['r2'],float)
        # The frozen bootstrap schema stores one [lower,upper] pair per target.
        bounds=np.asarray(m['r2_ci95'],float)
        if bounds.shape==(2,len(vals)):
            bounds=bounds.T
        labels=[n.replace('ball_minus_ego.','Ball−ego ').replace('ego.','Ego ').replace('location.','').replace('velocity.','') for n in item['target_names']]
        ax.bar(np.arange(len(vals)),vals,color=['#3978a8']*3+['#568a64']*3)
        ax.errorbar(np.arange(len(vals)),vals,yerr=np.maximum(np.stack([vals-bounds[:,0],bounds[:,1]-vals]),0),fmt='none',ecolor='black',capsize=3,lw=1)
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(np.arange(len(vals)),labels,rotation=25,ha='right')
        ax.set(title=group.capitalize(),ylabel='Fresh-match R²',ylim=(min(-.15,float(np.nanmin(bounds))-.05),1.02))
    fig.suptitle(f'Frozen role-relative probes: {report["n_matches"]} new matches\nDescriptive whole-match bootstrap intervals; observational decoding',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(ROOT/f'figures/fresh_role_decoding.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','confirmation'),required=True);args=p.parse_args()
    development() if args.phase=='development' else confirmation()
