#!/usr/bin/env python3
"""Package audited causal or sparse derived artifacts; never upload or publish."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='8'
import numpy as np
from package_development import ROOT, read, require, sha, fingerprint

FORBIDDEN = {'y','targets','actions','frames','physics','position','velocity',
             'recipient_source_targets_final','donor_source_targets_final'}
PRIVATE_FIELDS={'recipient_source_ball_z','donor_source_ball_z','recipient_source_targets_final','donor_source_targets_final'}


def public_json(value):
    if isinstance(value,dict):return {k:public_json(v) for k,v in value.items() if k not in PRIVATE_FIELDS}
    if isinstance(value,list):return [public_json(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('causal','sparse'),required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--report',type=Path)
    args=p.parse_args(); stage=args.stage
    folder=ROOT/'results'/('causal_development_v1' if stage=='causal' else 'feature_development_v1')
    args.output=args.output or Path('/data2/ishaangp/mira-interp/releases')/(stage+'-development-v1-2026-09-07.tar.gz')
    args.report=args.report or ROOT/'results'/(stage+'_development_archive.json')
    partial=args.output.with_suffix(args.output.suffix+'.partial')
    require(not any(path.exists() for path in (args.output,partial,args.report)),'Archive/report is immutable')
    started=time.monotonic(); frozen={}; specifications=[]
    def bind(path,expected=None):
        path=Path(path).resolve(); actual=sha(path)
        require(expected is None or expected==actual,'Artifact changed: '+str(path));frozen[path]=actual;return actual
    def add(path,name,*,arrays=None,transform_json=False,expected=None):
        path=Path(path);bind(path,expected)
        specifications.append(dict(path=path,name=name,arrays=arrays,transform_json=transform_json))
    bind(Path(__file__));bind(ROOT/'scripts/package_development.py')
    audit_path=folder/('selection_audit.json' if stage=='causal' else 'sparse_audit.json')
    audit=read(audit_path);require(audit['status']=='passed','Independent completed audit required');bind(audit_path)
    if stage=='causal':
        result=read(folder/'selection_results.json'); discovery=read(folder/'discovery_selection.json');pilot=read(folder/'pilot.json')
        require(audit['selection_results_sha256']==bind(folder/'selection_results.json') and audit['conditions_checked']==660,'Causal completion differs')
        require(audit['discovery_selection_sha256']==bind(folder/'discovery_selection.json'),'Discovery lock differs')
        discovery_audit=read(folder/'discovery_audit.json');require(discovery_audit['status']=='passed','Discovery audit failed')
        require(pilot['status']=='passed','Pilot failed')
        reg=read(folder/'registration.json');require(bind(folder/'registration.json')==audit['registration_sha256'],'Causal registration differs')
        for path,expected in reg['code_sha256'].items():bind(ROOT/path,expected)
        entries=[pilot['pair']]+discovery['pairs']+result['pairs']
        require(len(entries)==85 and len({e['path'] for e in entries})==85,'Missing/duplicate causal pairs')
        base={'baseline_flow','donor_flow','recipient_clean_codec','donor_clean_codec','recipient_noise','recipient_z_t'}
        pattern=re.compile(r'^site(?:[0-9]|1[0-6])_(?:recipient_tile|donor_tile|objective_gradient_tile|ball_z_pullback|(?:self|exact_donor|component_transfer|ablate_to_discovery_mean|restore|norm_matched_random|norm_matched_random_full|norm_matched_random_ablation|wrong_variable_ball_x|wrong_view_component)_(?:flow|endpoint_tile|replacement_tile))$')
        for entry in entries:
            bind(entry['path'],entry['sha256']);row=read(entry['path'])
            require(row['status']=='passed' and row['registration_sha256']==audit['registration_sha256'] and row['split'] in ('pilot','discovery','selection'),'Wrong causal pair')
            path=Path(row['artifact_path']);bind(path,row['artifact_sha256'])
            with np.load(path,allow_pickle=False) as saved:
                keep=sorted(k for k in saved.files if k in base or pattern.fullmatch(k))
                require(set(saved.files)-set(keep)=={'recipient_source_targets_final','donor_source_targets_final'},'Unexpected causal arrays')
            stem=row['split']+'/'+path.stem
            add(path,'pairs/'+stem+'.npz',arrays=keep,expected=row['artifact_sha256'])
            add(entry['path'],'pairs/'+stem+'.json',transform_json=True,expected=entry['sha256'])
        methods={'pair_archives':85,'exact_selection_conditions':660,'removed_arrays':sorted(FORBIDDEN & {'recipient_source_targets_final','donor_source_targets_final'}),
                 'removed_pair_report_fields':sorted(PRIVATE_FIELDS)}
    else:
        require(audit['all_reconstruction_metrics_recomputed'] and len(audit['checkpoints'])==30 and len(audit['feature_reports'])==3,'Sparse gate incomplete')
        protocol_path=ROOT/'configs/feature_development_v1.json';protocol=read(protocol_path)
        require(bind(protocol_path)==audit['protocol_sha256'],'Sparse protocol differs')
        for entry in audit['artifact_bindings']:bind(entry['path'],entry['sha256'])
        for entry in audit['checkpoints']:
            path=Path(entry['path']);add(path,'checkpoints/'+path.parent.name+'/'+path.name,expected=entry['sha256'])
        for entry in audit['feature_reports']:
            path=Path(entry['path']);add(path,'checkpoints/'+path.parent.name+'/'+path.name,expected=entry['sha256'])
        data_path=Path(protocol['input_path']);bind(data_path,audit['input_npz_sha256'])
        with np.load(data_path,allow_pickle=False) as source:
            roles=source['split'].astype(str);require(set(roles)=={'discovery','selection'},'Confirmation excluded')
            keep=sorted(set(source.files)-{'position','velocity'})
            require(not set(keep)&FORBIDDEN,'Source labels remain in sparse input')
        add(data_path,'features/selected_descriptors.npz',arrays=keep,expected=audit['input_npz_sha256'])
        add(protocol_path,'provenance/feature_development_v1.json')
        for entry in read(ROOT/'results/geometry_development_v1/geometry_audit.json')['comparisons']:
            for key in ('report','export','maps'):
                path=Path(entry[key]['path']);add(path,'geometry/'+entry['variable']+'/'+path.name,expected=entry[key]['sha256'])
            for name,digest in entry['figures'].items():add(Path(entry['maps']['path']).parent/name,'geometry/'+entry['variable']+'/'+name,expected=digest)
        for name in ('input_audit.json','selected_probe.json','gate.json','geometry_audit.json'):
            add(ROOT/'results/geometry_development_v1'/name,'geometry/'+name)
        methods={'checkpoints':30,'feature_reports':3,'descriptor_rows':8064,'descriptor_dimensions':1536,'removed_arrays':['position','velocity']}
    # Include exact completed gate/figure bytes; JSON redaction is limited to raw source height fields.
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix in ('.json','.png','.pdf','.npz'):
            if path.suffix=='.npz':
                require(stage=='sparse' and path.name=='retention_arrays.npz','Unexpected top-level array file')
                with np.load(path,allow_pickle=False) as saved:keep=sorted(saved.files)
                add(path,'results/'+path.name,arrays=keep)
            else:add(path,'results/'+path.name,transform_json=path.suffix=='.json')
    require(len({s['name'] for s in specifications})==len(specifications),'Duplicate archive names')
    documentation=('This archive contains audited '+stage+' development artifacts.\n'
        'Raw Rocket Science videos, actions and per-observation simulator state labels are excluded.\n'
        'Causal outputs and codec latents are model-derived. Sparse checkpoints and descriptors are study outputs.\n'
        'JSON source-height fields and source physics arrays are removed where present; original source hashes remain in manifest provenance.\n'
        'An audit pass establishes reproducible computation, not physical control. Source-label reproduction requires accepting Rocket Science terms.\n'
        'All tensors, report transforms, byte hashes and array fingerprints are read back and verified. No upstream model weights are included.\n')
    expected={};args.output.parent.mkdir(parents=True,exist_ok=True)
    def record(archive,name,content,**metadata):
        info=tarfile.TarInfo(name);info.size=len(content);info.mtime=0;info.mode=0o644
        archive.addfile(info,io.BytesIO(content));expected[name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),**metadata}
    with partial.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=6,mtime=0,filename='') as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for index,spec in enumerate(specifications):
                path=spec['path'];metadata={'source_path':str(path.resolve()),'source_sha256':frozen[path.resolve()]}
                if spec['arrays'] is not None:
                    with np.load(path,allow_pickle=False) as saved:arrays={key:saved[key] for key in spec['arrays']}
                    require(not set(arrays)&FORBIDDEN,'Forbidden raw source array')
                    metadata['arrays']={key:fingerprint(value) for key,value in arrays.items()}
                    stream=io.BytesIO();np.savez(stream,**arrays);content=stream.getvalue()
                elif spec['transform_json']:
                    content=(json.dumps(public_json(read(path)),indent=2,allow_nan=False)+'\n').encode();metadata['source_height_fields_removed']=True
                else:content=path.read_bytes()
                record(archive,spec['name'],content,**metadata)
                if (index+1)%20==0:print(json.dumps({'stage':'write','members':index+1,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
            record(archive,'README.txt',documentation.encode())
            manifest={'stage':stage,'scope':'development_only','physical_control_established':False,'source_labels_video_actions_included':False,
                      'independent_audit':{'path':str(audit_path),'sha256':sha(audit_path)},'methods':methods,'members':expected.copy()}
            record(archive,'manifest.json',(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode())
    require(partial.stat().st_size<2_000_000_000,'Archive exceeds conservative GitHub asset size bound')
    seen=set();array_count=0
    with tarfile.open(partial,'r|gz') as archive:
        for member in archive:
            require(member.isfile() and member.name in expected and member.name not in seen,'Unsafe/unexpected/duplicate member');seen.add(member.name)
            content=archive.extractfile(member).read();recorded=expected[member.name]
            require(len(content)==recorded['bytes'] and hashlib.sha256(content).hexdigest()==recorded['sha256'],'Archived bytes differ')
            if 'arrays' in recorded:
                with np.load(io.BytesIO(content),allow_pickle=False) as saved:
                    require(set(saved.files)==set(recorded['arrays']) and not set(saved.files)&FORBIDDEN,'Archived array set differs')
                    for key,digest in recorded['arrays'].items():require(fingerprint(saved[key])==digest,'Archived array fingerprint differs');array_count+=1
            if len(seen)%20==0:print(json.dumps({'stage':'readback','members':len(seen),'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
    require(seen==set(expected),'Missing archive members')
    require(all(sha(path)==digest for path,digest in frozen.items()),'Sources changed during packaging')
    partial.rename(args.output)
    report={**manifest,'status':'passed','archive_path':str(args.output),'archive_sha256':sha(args.output),'archive_bytes':args.output.stat().st_size,
            'all_archive_members_verified':True,'archive_members_verified':len(seen),'array_fingerprints_verified':array_count,
            'packaging_script_sha256':sha(Path(__file__)),'packaging_helper_sha256':sha(ROOT/'scripts/package_development.py'),
            'elapsed_seconds':round(time.monotonic()-started,3),'publication_state':'local_only'}
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('status','archive_path','archive_bytes','archive_sha256','elapsed_seconds')}),flush=True)
if __name__=='__main__':main()
