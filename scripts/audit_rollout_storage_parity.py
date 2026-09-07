#!/usr/bin/env python3
"""Require exact model/array parity for the prospective lossless storage amendment."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = {'frames','latents','native_edit_descriptor','edited_descriptor','native_edit_tile','edited_tile'}
ROW_IGNORED = {'generated_artifact_path','generated_sha256','generated_bytes','registration_sha256','record_report','elapsed_seconds'}
REPORT_IGNORED = {'records','registration_sha256','controls','elapsed_seconds','peak_gpu_allocated_bytes',
                  'generated_artifact_bytes','mean_saved_condition_bytes'}
REGISTRATION_IGNORED = {'created_utc','code_sha256','bindings','storage'}
EXTRA_BINDINGS = {'storage_benchmark','prior_successful_pilot','prior_successful_pilot_audit'}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def equal_except(old,new,ignored,description):
    require({k:v for k,v in old.items() if k not in ignored} ==
            {k:v for k,v in new.items() if k not in ignored},description)


def bitwise_array_equal(old,new):
    return old.dtype == new.dtype and old.shape == new.shape and old.tobytes(order='C') == new.tobytes(order='C')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-directory',type=Path,default=ROOT/'results/rollout_steering_v1')
    parser.add_argument('--new-directory',type=Path,default=ROOT/'results/rollout_steering_v2')
    parser.add_argument('--review',type=Path,default=ROOT/'results/rollout_storage_v2_review.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();started=time.perf_counter()
    output=args.output or args.new_directory/'pilot_parity_audit.json'
    require(not output.exists(),'Completed parity audits are immutable')
    frozen={}
    def bind(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        require(expected is None or actual == expected,f'Bound file changed:{path}')
        frozen[path]=actual
        return actual
    def checked(entry):
        bind(entry['path'],entry['sha256'])
        return read(entry['path'])
    bind(Path(__file__))
    review=read(args.review);bind(args.review)
    require(review['status']=='passed' and review['model_actions_seeds_paths_doses_and_arithmetic_unchanged'] is True
            and review['new_provenance_bindings_rechecked_by_verify'] is True,'Storage source diff was not independently reviewed')
    for name,digest in review['code_sha256'].items():bind(ROOT/name,digest)
    registrations=[];reports=[];audits=[]
    for directory in (args.old_directory,args.new_directory):
        registration=read(directory/'registration.json');registration_hash=bind(directory/'registration.json')
        report=read(directory/'pilot.json');manifest_hash=bind(directory/'pilot.json')
        audit=read(directory/'pilot_audit.json');bind(directory/'pilot_audit.json')
        require(registration['status']=='registered_before_rollout_steering' and report['status']=='passed_rollout_steering_pilot'
                and report['registration_sha256']==registration_hash,'Registered successful pilot required')
        require(audit['status']=='passed_rollout_generation_audit' and audit['manifest_sha256']==manifest_hash
                and audit['registration_sha256']==registration_hash and audit['n_records']==2,'Independent pilot audit does not bind exact outputs')
        require(report['full_inferences']==5 and report['n_records']==2 and report['match_count']==1 and report['seed_count']==1,'Wrong engineering coverage')
        for name,digest in registration['code_sha256'].items():bind(ROOT/name,digest)
        for entry in registration['bindings'].values():bind(entry['path'],entry['sha256'])
        for entry in registration['probe_bindings'].values():bind(entry['path'],entry['sha256'])
        registrations.append(registration);reports.append(report);audits.append(audit)
    old,new=registrations
    require(new['storage']=='Lossless NPZ DEFLATE level1; same arrays and model arithmetic as successful v1 pilot', 'Unexpected storage amendment description')
    equal_except(old,new,REGISTRATION_IGNORED,'Scientific registration fields differ')
    require(set(new['bindings'])-set(old['bindings'])==EXTRA_BINDINGS and set(old['bindings']).issubset(new['bindings']),
            'Unexpected new prerequisite binding')
    equal_except(old['bindings'],new['bindings'],EXTRA_BINDINGS,'Original prerequisite bindings changed')
    require(new['bindings']['prior_successful_pilot']['sha256']==bind(args.old_directory/'pilot.json')
            and new['bindings']['prior_successful_pilot_audit']['sha256']==bind(args.old_directory/'pilot_audit.json'),
            'Version2 does not bind the exact successful version1 pilot')
    benchmark=checked(new['bindings']['storage_benchmark'])
    require(benchmark['status']=='passed_storage_benchmark' and benchmark['pilot_manifest_sha256']==bind(args.old_directory/'pilot.json')
            and review['benchmark_sha256']==new['bindings']['storage_benchmark']['sha256'],'Storage benchmark lineage differs')
    original_code='scripts/rollout_steering.py';new_code='scripts/rollout_steering_v2.py';helper='src/mira_interp/array_storage.py'
    require(original_code in old['code_sha256'] and new_code not in old['code_sha256'] and helper not in old['code_sha256']
            and new_code in new['code_sha256'] and helper in new['code_sha256'] and original_code not in new['code_sha256'],
            'Storage amendment code identity differs')
    equal_except(old['code_sha256'],new['code_sha256'],{original_code,new_code,helper},'Shared scientific implementation hashes changed')
    require(old['code_sha256'][original_code]==review['code_sha256'][original_code]
            and new['code_sha256'][new_code]==review['code_sha256'][new_code]
            and new['code_sha256'][helper]==review['code_sha256'][helper],'Producer/helper no longer match storage-only diff review')
    old_report,new_report=reports
    equal_except(old_report,new_report,REPORT_IGNORED,'Pilot non-storage metadata/model loading differs')
    equal_except(old_report['controls'],new_report['controls'],{'four_control_inference_seconds'},'Control outcomes or raw output hashes changed')
    old_rows={r['record_id']:r for r in old_report['records']};new_rows={r['record_id']:r for r in new_report['records']}
    require(len(old_rows)==len(new_rows)==2 and set(old_rows)==set(new_rows),'Pilot condition identity differs')
    results=[]
    for identifier,old_row in old_rows.items():
        new_row=new_rows[identifier]
        equal_except(old_row,new_row,ROW_IGNORED,'Model/action/trace/probe-effect metadata changed:'+identifier)
        for row in (old_row,new_row):
            require(checked(row['record_report'])=={k:v for k,v in row.items() if k!='record_report'},'Per-condition metadata differs')
            bind(row['generated_artifact_path'],row['generated_sha256'])
            require(Path(row['generated_artifact_path']).stat().st_size==row['generated_bytes'],'Archive byte count differs')
        with np.load(old_row['generated_artifact_path'],allow_pickle=False) as a, np.load(new_row['generated_artifact_path'],allow_pickle=False) as b:
            require(set(a.files)==set(b.files)==ARRAYS,'Pilot saved array schema differs')
            arrays={}
            for key in sorted(ARRAYS):
                first,second=a[key],b[key]
                require(bitwise_array_equal(first,second) and np.isfinite(first).all(),'Pilot array is not bitwise identical:'+key)
                arrays[key]={'shape':list(first.shape),'dtype':str(first.dtype),
                             'sha256':hashlib.sha256(first.tobytes(order='C')).hexdigest()}
        results.append({'record_id':identifier,'old_archive_sha256':old_row['generated_sha256'],
                        'new_archive_sha256':new_row['generated_sha256'],'old_bytes':old_row['generated_bytes'],
                        'new_bytes':new_row['generated_bytes'],'arrays':arrays,'all_six_arrays_bitwise_equal':True})
    require(all(sha(path)==digest for path,digest in frozen.items()),'Bound evidence changed during parity audit')
    result={'status':'passed_storage_only_pilot_parity','records':results,
        'old_manifest_sha256':bind(args.old_directory/'pilot.json'),'new_manifest_sha256':bind(args.new_directory/'pilot.json'),
        'old_audit_sha256':bind(args.old_directory/'pilot_audit.json'),'new_audit_sha256':bind(args.new_directory/'pilot_audit.json'),
        'old_registration_sha256':bind(args.old_directory/'registration.json'),'new_registration_sha256':bind(args.new_directory/'registration.json'),
        'review_sha256':bind(args.review),'all_model_output_arrays_bitwise_equal':True,
        'all_actions_traces_raw_hashes_sites_probe_effects_exact':True,'same_frozen_science_and_implementation_except_storage':True,
        'only_encoding_paths_provenance_and_runtime_metadata_differ':True,'physical_control_established':False,
        'gpu_used':False,'source_video_or_simulator_labels_loaded':False,'script_sha256':sha(Path(__file__)),
        'elapsed_seconds':time.perf_counter()-started,
        'artifact_bindings':[{'path':str(path),'sha256':digest} for path,digest in frozen.items()]}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'output':str(output),'sha256':sha(output),'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
