#!/usr/bin/env python3
"""Publish all audited generated tensors, excluding observed context and source labels."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mira_interp.array_storage import write_npz

LIMIT=1_800_000_000
RESERVE=25*1024**3
SHAPES={'frames':(4,24,3,288,512),'latents':(4,12,9,16,32),
        'native_edit_tile':(9,16,2048),'edited_tile':(9,16,2048),
        'native_edit_descriptor':(1,1536),'edited_descriptor':(1,1536)}
PUBLIC_SHAPES={**SHAPES,'frames':(4,8,3,288,512),'latents':(4,4,9,16,32)}
PREDICTION_KEYS={'record_ids','role_predictions','absolute_ball_predictions','role_target_scale'}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def encoded_json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def array_fingerprint(value):
    # Canonical logical C order; hashes never redistribute private context values.
    return {'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.nbytes,
            'sha256':hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def public_arrays(source):
    require(set(source)==set(SHAPES),'Unexpected source arrays: reject labels/actions/extra payloads')
    for key,value in source.items():
        require(value.shape==SHAPES[key] and value.dtype==np.float32 and np.isfinite(value).all(),'Invalid generated source:'+key)
    require(source['frames'].min()>=0 and source['frames'].max()<=1,'Generated video outside0..1')
    return {key:value[:,16:] if key=='frames' else value[:,8:] if key=='latents' else value
            for key,value in source.items()}


def safe_name(name):
    path=PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and path.parts,'Unsafe archive member')
    return name


def tar_member_bytes(size):return 512+((size+511)//512)*512


def tar_total_bytes(body):return ((body+1024+10239)//10240)*10240


def public_entry(entry):return {key:value for key,value in entry.items() if key!='staged_path'}


def part_manifest(entries,index,phase,bindings):
    return encoded_json({'schema_version':1,'phase':phase,'part_index':index,'source_bindings':bindings,
        'members':[public_entry(e) for e in sorted(entries,key=lambda e:e['member'])],
        'observed_context_actions_simulator_labels_included':False})


def part_size(entries,index,phase,bindings):
    body=sum(tar_member_bytes(e['bytes']) for e in entries)
    return tar_total_bytes(body+tar_member_bytes(len(part_manifest(entries,index,phase,bindings))))


def partition(entries,phase,bindings,limit):
    require(10240<=limit<=LIMIT,'Part limit must be between10KiB and1.8GB')
    groups=[];current=[]
    for entry in sorted(entries,key=lambda e:e['member']):
        if current and part_size(current+[entry],len(groups)+1,phase,bindings)>limit:
            groups.append(current);current=[]
        require(part_size([entry],len(groups)+1,phase,bindings)<=limit,'One encoded artifact exceeds part limit')
        current.append(entry)
    if current:groups.append(current)
    return groups


def make_info(name,size):
    info=tarfile.TarInfo(safe_name(name));info.size=size;info.mtime=0;info.mode=0o644
    info.uid=info.gid=0;info.uname=info.gname=''
    return info


def write_part(path,entries,manifest):
    with path.open('xb') as stream,tarfile.open(fileobj=stream,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        tar.addfile(make_info('PART_MANIFEST.json',len(manifest)),io.BytesIO(manifest))
        for entry in sorted(entries,key=lambda e:e['member']):
            with Path(entry['staged_path']).open('rb') as source:
                tar.addfile(make_info(entry['member'],entry['bytes']),source)


def verify_part(path,entries,manifest):
    expected={e['member']:e for e in entries};seen=set()
    with tarfile.open(path,'r:') as tar:
        for member in tar:
            safe_name(member.name)
            require(member.name not in seen and member.isfile() and member.mtime==member.uid==member.gid==0
                    and member.mode==0o644 and not member.pax_headers,'Unsafe, duplicate or noncanonical tar member')
            seen.add(member.name);stream=tar.extractfile(member)
            if member.name=='PART_MANIFEST.json':
                require(stream.read()==manifest,'Part manifest readback differs');continue
            require(member.name in expected and member.size==expected[member.name]['bytes'],'Unexpected member or byte length')
            entry=expected[member.name];h=hashlib.sha256()
            # Public NPZs are small enough to check one at a time; never load a whole part.
            if entry['kind']=='generated_future_npz':
                raw=stream.read();h.update(raw)
                with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                    require(set(z.files)==set(PUBLIC_SHAPES),'Archive NPZ contains forbidden/extra arrays')
                    for key in z.files:
                        require(z[key].shape==PUBLIC_SHAPES[key] and array_fingerprint(z[key])==entry['arrays'][key]['public'],
                                'Tar NPZ array fingerprint differs')
            else:
                for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
            require(h.hexdigest()==entry['sha256'],'Tar member hash differs')
    require(seen==set(expected)|{'PART_MANIFEST.json'},'Missing tar members')


def main():
    initial_code_hashes={path:sha(path) for path in (Path(__file__).resolve(),(ROOT/'src/mira_interp/array_storage.py').resolve())}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('pilot','selection','confirmation'))
    parser.add_argument('--generation-dir',type=Path,default=ROOT/'results/rollout_steering_v2')
    parser.add_argument('--evaluation-dir',type=Path,default=ROOT/'results/generated_evaluation_v2')
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/generated_evaluation_v1.json')
    parser.add_argument('--output-dir',type=Path,default=Path('/data2/ishaangp/mira-interp/publication/generated_v2_complete'))
    parser.add_argument('--report',type=Path)
    parser.add_argument('--max-part-bytes',type=int,default=LIMIT)
    args=parser.parse_args();started=time.perf_counter()
    report_path=args.report or ROOT/'results'/f'generated_publication_{args.phase}_complete.json'
    require(not report_path.exists(),'Completed package reports are immutable')
    phase_dir=args.output_dir/args.phase
    require(not phase_dir.exists(),'Use a new package directory; existing packages are immutable')
    bindings={};frozen=dict(initial_code_hashes)
    def bind(name,path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        require(expected is None or digest==expected,'Changed source:'+str(path))
        frozen[path]=digest;bindings[name]={'path':str(path),'sha256':digest}
        return read(path)
    gen=bind('generation_manifest',args.generation_dir/(args.phase+'.json'))
    audit=bind('generation_audit',args.generation_dir/(args.phase+'_audit.json'))
    evaluation=bind('evaluation_manifest',args.evaluation_dir/(args.phase+'.json'))
    measured=bind('measurement_audit',args.evaluation_dir/(args.phase+'_audit.json'))
    registration=bind('generation_registration',args.generation_dir/'registration.json',gen['registration_sha256'])
    protocol=bind('evaluation_protocol',args.protocol,evaluation['protocol_sha256'])
    gen_sha=bindings['generation_manifest']['sha256'];audit_sha=bindings['generation_audit']['sha256']
    eval_sha=bindings['evaluation_manifest']['sha256']
    require(gen['status']==('passed_rollout_steering_pilot' if args.phase=='pilot' else 'passed_rollout_generation')
            and audit['status']=='passed_rollout_generation_audit' and audit['manifest_sha256']==gen_sha
            and audit['registration_sha256']==gen['registration_sha256'],'Generation completion gate failed')
    require(evaluation['status']=='passed_frozen_generated_video_evaluation' and evaluation['phase']==args.phase
            and evaluation['generation_manifest_sha256']==gen_sha and evaluation['generation_audit_sha256']==audit_sha,
            'Frozen measurement gate failed')
    require(measured['status']=='passed_generated_measurement_audit' and measured['phase']==args.phase
            and measured['evaluation_sha256']==eval_sha and measured['generation_manifest_sha256']==gen_sha
            and measured['generation_audit_sha256']==audit_sha and measured['protocol_sha256']==evaluation['protocol_sha256'],
            'Independent measurement audit binding failed')
    expected_count={'pilot':2,'selection':420,'confirmation':924}[args.phase]
    expected_matches={'pilot':1,'selection':10,'confirmation':22}[args.phase]
    records=gen['records'];ids=[r['record_id'] for r in records]
    require(len(records)==gen['n_records']==evaluation['records']==measured['records_checked']==expected_count
            and len(set(ids))==expected_count and len({r['match_id'] for r in records})==expected_matches,
            'Incomplete or duplicate phase coverage')
    require(all(r['split']==args.phase for r in records),'Wrong phase role')
    require({(r['record_id'],r['generated_sha256']) for r in records}==
            {(r['record_id'],r['generated_sha256']) for r in audit['records']},'Generation audit record coverage differs')
    caches={r['record_id']:r for r in measured['cache_records']}
    require(set(caches)==set(ids) and len(measured['cache_records'])==len(ids),'Measurement cache coverage differs')
    expected_sources={r['clip_id']:r for r in registration['records'] if r['role']==args.phase}
    require({r['clip_id'] for r in records}==set(expected_sources),'Registered match cohort differs')
    for row in records:
        source=expected_sources[row['clip_id']]
        require(row['match_id']==source['match_id'] and row['source_sha256']==source['sha256']
                and re.fullmatch(r'[A-Za-z0-9_+.-]+',row['record_id']) and '..' not in row['record_id'],'Invalid record/source identity')
    if args.phase!='pilot':
        expected={(s['clip_id'],seed,name,float(dose)) for s in expected_sources.values() for seed in registration['seeds']
                  for name,dose in registration['condition_grid']}
        require({(r['clip_id'],r['seed'],r['intervention_type'],r['dose']) for r in records}==expected,'Complete dose/seed grid required')
        require(measured['all_summary_metrics_independently_recomputed'] is True,'Full-phase metric audit is incomplete')
    experiment_code={}
    for table in (registration['code_sha256'],protocol['code_sha256']):
        for name,digest in table.items():
            require(sha(ROOT/name)==digest,'Registered source code changed:'+name)
            require(name not in experiment_code or experiment_code[name]==digest,'Conflicting experiment source hashes')
            experiment_code[name]=digest
    args.output_dir.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(args.output_dir).free>RESERVE+2*sum(r['generated_bytes'] for r in records),
            'Insufficient free space for staged/public archives plus25GiB reserve')
    phase_dir.mkdir(parents=True)
    stage=phase_dir/'staging';stage.mkdir()
    entries=[];names=set();published=[]
    def add_file(name,path,kind,**extra):
        safe_name(name);require(name not in names,'Duplicate archive member');names.add(name)
        path=Path(path);entries.append({'member':name,'staged_path':str(path),'bytes':path.stat().st_size,
            'sha256':sha(path),'kind':kind,**extra})
        return entries[-1]
    def add_original(name,path,expected=None,kind='metadata_json'):
        path=Path(path).resolve();digest=sha(path)
        require(expected is None or digest==expected,'Changed original:'+str(path));frozen[path]=digest
        return add_file(name,path,kind,source_path=str(path),source_sha256=digest)
    # Include exact completed manifests, aggregate results, and per-condition metadata.
    for name,binding in bindings.items():add_original('metadata/'+name+'.json',binding['path'],binding['sha256'])
    for index,worker in enumerate(gen.get('workers',[])):
        add_original(f'metadata/generation_worker{index}.json',worker['path'],worker['sha256'])
    for name,digest in sorted(experiment_code.items()):
        add_original('experiment_code/'+name,ROOT/name,digest,'source_code')
    add_original('THIRD_PARTY.md',ROOT/'THIRD_PARTY.md',kind='attribution')
    add_original('licenses/MIRA_APACHE_2_0.txt',ROOT/'external/mira/LICENSE',kind='attribution')
    add_original('licenses/VIDEOMAE_MODEL_CARD.md',Path('/data2/ishaangp/mira-interp/models/videomae-base/README.md'),kind='attribution')
    loading=gen.get('model_loading') or read(gen['workers'][0]['path'])['model_loading']
    provenance={'generator':{key:loading[key] for key in ('model_repo','model_revision','source_revisions','verified_assets')},
        'frozen_video_evaluator':evaluation['evaluator'],'registered_experiment_code_sha256':experiment_code,
        'generator_and_evaluator_weights_included':False}
    provenance_path=stage/'model_provenance.json';provenance_path.write_bytes(encoded_json(provenance))
    add_file('metadata/model_provenance.json',provenance_path,'derived_provenance')
    attribution=('''# Model and source attribution

Generator: Alakazam MIRA Mini 4P, an independent 1B reproduction of MIRA.
https://huggingface.co/alakazamworld/mira-mini-4p
The project records CC BY-NC-SA4.0 terms; generated visual artifacts retain that
attribution and license notice. See THIRD_PARTY.md and
https://creativecommons.org/licenses/by-nc-sa/4.0/.

MIRA implementation: General Intuition, Kyutai and collaborators.
https://github.com/mira-wm/mira ; Apache2.0 license included in licenses/.
Rocket Science source data: Kyutai, https://huggingface.co/datasets/kyutai/rocket-science.
No gated source video, actions or simulator-label arrays are bundled.
Rocket League copyright Psyonix LLC/Epic Games; no endorsement is implied.

Independent video evaluator: VideoMAE by Tong et al.,
https://arxiv.org/abs/2203.12602 and https://github.com/MCG-NJU/VideoMAE.
Original model https://huggingface.co/MCG-NJU/videomae-base, with CC-BY-NC4.0
model-card attribution included in licenses/. No encoder weights are bundled.

metadata/model_provenance.json records exact model revisions, checkpoint hashes,
source revisions, evaluator configuration and registered experiment-code hashes.
experiment_code/ contains exact copies of those source files; storage code is in
code/. The copied scientific implementation is unchanged from the audited runs.
''').encode()
    attribution_path=stage/'MODEL_ATTRIBUTION.md';attribution_path.write_bytes(attribution)
    add_file('MODEL_ATTRIBUTION.md',attribution_path,'attribution')
    for name in ('pilot_parity_audit.json',):
        path=args.generation_dir/name
        if path.exists():add_original('metadata/'+name,path)
    if args.phase=='pilot' and (args.evaluation_dir/'pilot_measurement_parity.json').exists():
        add_original('metadata/pilot_measurement_parity.json',args.evaluation_dir/'pilot_measurement_parity.json')
    prediction=evaluation['prediction_artifact']
    require(prediction['sha256']==measured['prediction_artifact_sha256'],'Prediction audit hash differs')
    with np.load(prediction['path'],allow_pickle=False) as z:
        require(set(z.files)==PREDICTION_KEYS and z['record_ids'].astype(str).tolist()==ids,'Prediction row/schema mismatch')
        require(z['role_predictions'].shape==(len(records),4,4,12) and np.isfinite(z['role_predictions']).all(),
                'Invalid derived role predictions')
        require(np.array_equal(z['absolute_ball_predictions'],z['role_predictions'][...,:6]+z['role_predictions'][...,6:]),
                'Absolute ball prediction mismatch')
    add_original('predictions/all_records.npz',prediction['path'],prediction['sha256'],'derived_predictions_npz')
    record_lookup={r['record_id']:r for r in records}
    for number,row in enumerate(sorted(records,key=lambda r:r['record_id']),1):
        identifier=row['record_id'];source=Path(row['generated_artifact_path'])
        require(sha(source)==row['generated_sha256'],'Private generated source changed')
        frozen[source.resolve()]=row['generated_sha256']
        metadata=read(row['record_report']['path'])
        require(metadata=={k:v for k,v in row.items() if k!='record_report'},'Condition report differs from manifest')
        add_original('metadata/conditions/'+identifier+'.json',row['record_report']['path'],row['record_report']['sha256'])
        cache=caches[identifier]
        require(cache['generated_sha256']==row['generated_sha256'],'Per-record prediction cache source mismatch')
        with np.load(cache['path'],allow_pickle=False) as z:
            require(set(z.files)=={'role_predictions','generated_sha256','protocol_sha256'}
                    and str(z['generated_sha256'])==row['generated_sha256']
                    and str(z['protocol_sha256'])==evaluation['protocol_sha256'], 'Unexpected or unbound prediction cache')
        add_original('predictions/conditions/'+identifier+'.npz',cache['path'],cache['sha256'],'derived_predictions_npz')
        with np.load(source,allow_pickle=False) as z:original={key:z[key] for key in z.files}
        future=public_arrays(original)
        arrays={key:{'source':array_fingerprint(original[key]),'public':array_fingerprint(value),
                      'slice':':,16:' if key=='frames' else ':,8:' if key=='latents' else 'all'}
                for key,value in future.items()}
        output=stage/(identifier+'.npz')
        require(shutil.disk_usage(phase_dir).free>RESERVE+2*sum(a.nbytes for a in future.values()),'Disk reserve reached')
        with output.open('xb') as stream:write_npz(stream,future,compresslevel=1)
        with np.load(output,allow_pickle=False) as restored:
            require(set(restored.files)==set(PUBLIC_SHAPES),'Public NPZ contains forbidden arrays')
            for key in restored.files:
                value=restored[key]
                require(value.shape==PUBLIC_SHAPES[key] and array_fingerprint(value)==arrays[key]['public']
                        and value.tobytes(order='C')==future[key].tobytes(order='C'),'Public tensor is not the exact source slice')
        entry=add_file('tensors/'+identifier+'.npz',output,'generated_future_npz',record_id=identifier,
            source_path=str(source.resolve()),source_sha256=row['generated_sha256'],arrays=arrays,
            match_id=row['match_id'],clip_id=row['clip_id'],split=row['split'],seed=row['seed'],
            intervention_type=row['intervention_type'],dose=row['dose'],baseline_record_id=row['baseline_record_id'])
        published.append(public_entry(entry));del original,future
        if number%20==0 or number==len(records):print(json.dumps({'phase':args.phase,'staged_records':number,'total':len(records)}),flush=True)
    require({r['record_id'] for r in published}==set(record_lookup),'A generated condition was omitted')
    readme=('''# Complete generated-output release\n\nAll conditions of this audited phase are included. Each tensor NPZ preserves\nframes[:,16:] as FP32[4,8,3,288,512], latents[:,8:] as FP32[4,4,9,16,32],\nand all native/edited residual tiles and descriptors. No observed video context,\nactions, or simulator-label arrays are included. Frame0 in this public array is\noriginal generated frame16; latent0 is original generated latent8. Four views\nretain their original order. Values are unchanged, with no image quantization.\n\nPART_MANIFEST.json binds every member to its source hash and array fingerprints.\nOriginal metadata paths identify private sources; their raw data are not bundled.\nPrediction files are frozen VideoMAE estimates, not generated-video ground truth.\nTheir16-frame evaluator windows used decoded context that is deliberately absent\nfrom this release. Reproducing those exact estimates requires authorized context\ninputs. The included estimates and complete dose data allow reproduction of\nreported dose-response summaries without redistributing those context inputs.\n\nAll parts are deterministic USTAR archives (fixed owner/mode/zero timestamps),\nwith lossless level1 DEFLATE NPZ members. The external archive_manifest.json and\npart reports provide file SHA256 and exact byte lengths for uploader/readback\nverification. Extract all uniquely named tensor/metadata members into one folder;\nPART_MANIFEST.json is part-specific, so inspect it per archive.\n\nMIRA source: https://github.com/mira-wm/mira\nVideoMAE: https://huggingface.co/MCG-NJU/videomae-base\nFrozen model/source revisions and all experiment bindings are in the metadata.\nNo encoder checkpoint or gated source dataset is bundled here. Generated-video\naccuracy and reliable physical control are not established merely by completing\nthese archives. All negative/zero/reversed intervention outputs are retained.\n''').encode()
    (stage/'README.md').write_bytes(readme);add_file('README.md',stage/'README.md','release_documentation')
    add_original('code/package_generated_outputs.py',Path(__file__),initial_code_hashes[Path(__file__).resolve()],kind='source_code')
    add_original('code/array_storage.py',ROOT/'src/mira_interp/array_storage.py',initial_code_hashes[(ROOT/'src/mira_interp/array_storage.py').resolve()],kind='source_code')
    groups=partition(entries,args.phase,bindings,args.max_part_bytes);parts=[]
    for index,group in enumerate(groups,1):
        manifest=part_manifest(group,index,args.phase,bindings)
        filename=f'mira-generated-v2-{args.phase}-part{index:04d}.tar';path=phase_dir/filename
        write_part(path,group,manifest)
        require(path.stat().st_size==part_size(group,index,args.phase,bindings)<=args.max_part_bytes,'Actual tar size exceeds exact partition budget')
        verify_part(path,group,manifest)
        part={'status':'passed_generated_artifact_part','phase':args.phase,'completed':True,'part_index':index,'filename':filename,'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
              'manifest_sha256':hashlib.sha256(manifest).hexdigest(),'member_count':len(group)+1,
              'record_count':sum(e['kind']=='generated_future_npz' for e in group),'readback_verified':True}
        part_report_path=phase_dir/(filename+'.json')
        part_report_path.write_bytes(encoded_json({**part,'members':[public_entry(e) for e in group]}))
        part['report']={'path':str(part_report_path),'sha256':sha(part_report_path)}
        parts.append(part)
        print(json.dumps({'phase':args.phase,'part_complete':index,'parts':len(groups),'bytes':part['bytes']}),flush=True)
    require(all(sha(path)==digest for path,digest in frozen.items()),'Frozen evidence changed during publication preparation')
    require(sum(p['record_count'] for p in parts)==expected_count,'Archive record coverage differs')
    archive_manifest={'schema_version':1,'phase':args.phase,'source_bindings':bindings,'parts':parts,
        'records':published,'members':[public_entry(e) for e in entries],
        'all_registered_conditions_included':True,'observed_context_actions_simulator_labels_included':False,
        'max_part_bytes':args.max_part_bytes,'part_layout':'deterministic sorted members, actual512-byte tar padding and10240-byte closing alignment'}
    manifest_path=phase_dir/'archive_manifest.json';manifest_path.write_bytes(encoded_json(archive_manifest))
    result={'status':'passed_generated_artifact_package','phase':args.phase,'source_bindings':bindings,
        'generation_manifest_sha256':gen_sha,'generation_audit_sha256':audit_sha,'evaluation_sha256':eval_sha,
        'measurement_audit_sha256':bindings['measurement_audit']['sha256'],'registration_sha256':gen['registration_sha256'],
        'protocol_sha256':evaluation['protocol_sha256'],'expected_records':expected_count,'packaged_records':len(published),
        'parts':parts,'archive_manifest':{'path':str(manifest_path),'sha256':sha(manifest_path)},
        'all_members_readback_verified':True,'all_exported_arrays_exact_source_slices':True,'all_registered_conditions_included':True,
        'source_context_actions_simulator_labels_included':False,'private_sources_unchanged':True,
        'script_sha256':initial_code_hashes[Path(__file__).resolve()],'array_storage_sha256':initial_code_hashes[(ROOT/'src/mira_interp/array_storage.py').resolve()],
        'packaging_code_hashes_captured_at_start_and_unchanged':True,
        'gpu_used':False,'uploaded_to_github':False,'max_part_bytes':args.max_part_bytes,
        'elapsed_seconds':time.perf_counter()-started,'free_bytes_after':shutil.disk_usage(phase_dir).free}
    report_path.write_bytes(encoded_json(result));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
