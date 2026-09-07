#!/usr/bin/env python3
"""Plot every registered sparse variant/budget; ranges are across three seeds."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='8'
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from package_development import ROOT, read, require, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=ROOT/'results/feature_development_v1/sparse_audit.json')
    args=p.parse_args(); source_hash=sha(args.audit); report=read(args.audit)
    require(report['status']=='passed' and len(report['models'])==30,'Completed sparse audit required')
    labels=['ReLU','Signed','Block','Temporal\nblock','Shuffled\ntime block']
    def variant(m):
        if not m['temporal_weight']: return {'relu':0,'signed':1,'block':2}[m['kind']]
        return 4 if m['temporal_control']=='shuffled_within_match' else 3
    panels=[('Selection reconstruction\n(lower is better)',lambda m:m['metrics']['selection']['normalized_reconstruction_mse'],'Normalized MSE'),
            ('Discovery reconstruction\n(lower is better)',lambda m:m['metrics']['discovery']['normalized_reconstruction_mse'],'Normalized MSE'),
            ('Selection adjacent support\n(same camera view only)',lambda m:m['metrics']['selection']['adjacent_support_jaccard'],'Jaccard'),
            ('Selection frozen ball-z probe drift\n(lower is better)',lambda m:m['frozen_probe']['selection']['standardized_prediction_drift_per_target'][2],'Standardized prediction MSE'),
            ('Height-bin variation retained\n(1 = native variation)',lambda m:m['geometry_retention']['height']['reconstructed_over_original_centered_variance'],'Reconstructed / native variance'),
            ('Heading-bin distances changed\n(0 = preserved)',lambda m:m['geometry_retention']['heading']['pairwise_squared_distance_relative_l2'],'Relative L2 distortion')]
    fig,axes=plt.subplots(2,3,figsize=(14,8.6),constrained_layout=True)
    summary=[]
    for ax,(title,value,ylabel) in zip(axes.flat,panels):
        for j,budget in enumerate((32,64)):
            means,lows,highs=[],[],[]
            for v in range(5):
                selected=[m for m in report['models'] if m['active_scalar_budget']==budget and variant(m)==v]
                require(sorted(m['seed'] for m in selected)==[0,1,2],'Wrong full comparison grid')
                a=np.array([value(m) for m in selected]);means.append(a.mean());lows.append(a.mean()-a.min());highs.append(a.max()-a.mean())
                summary.append({'metric':title.replace('\n',' '),'active_budget':budget,'variant':labels[v].replace('\n',' '),'values':a.tolist()})
            ax.bar(np.arange(5)+(j-.5)*.36,means,.34,yerr=np.array([lows,highs]),capsize=3,label=f'{budget} active coordinates',color=('#386cb0','#ef8a33')[j])
        ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel);ax.set_xticks(np.arange(5),labels,fontsize=9)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        if title.startswith('Height-bin'):ax.axhline(1,color='black',lw=.8,ls='--')
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Sparse descriptor reconstruction: all 30 registered final checkpoints\nBars = three-seed means; whiskers = seed range, not confidence intervals',fontsize=14)
    folder=args.audit.parent; outputs={}
    for suffix in ('png','pdf'):
        out=folder/('sparse_comparison.'+suffix);require(not out.exists(),'Existing plot is immutable')
        fig.savefig(out,dpi=170);outputs[out.name]=sha(out)
    plt.close(fig)
    require(sha(args.audit)==source_hash,'Sparse audit changed during export')
    (folder/'plot_audit.json').write_text(json.dumps({'status':'passed','sparse_audit_sha256':source_hash,'script_sha256':sha(Path(__file__)),
        'all30_models_plotted':True,'seed_whiskers_are_confidence_intervals':False,'outputs':outputs,'values':summary},indent=2)+'\n')
    print(json.dumps({'status':'passed','outputs':outputs}))
if __name__=='__main__':main()
