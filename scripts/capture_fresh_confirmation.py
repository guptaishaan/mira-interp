#!/usr/bin/env python3
"""Capture reserved fresh observations only after development choices are locked."""
import argparse
import json
from pathlib import Path
import sys
import time
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'external/mira/src'),str(ROOT/'scripts')]
from mira_interp.model_loading import load_pretrained_four_player,sha256
from mira_interp.development_capture import channel_projection,capture
from capture_observations import clip_seed
from prepare_development import save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    lock_path=ROOT/'configs/fresh_evaluation_v3.json'
    lock=json.loads(lock_path.read_text())
    prep_path=ROOT/'results/fresh_confirmation_float.json'
    prep=json.loads(prep_path.read_text())
    if lock['status']!='frozen_before_fresh_capture' or prep['status']!='passed':
        raise ValueError('Registered choices and complete source preparation required')
    qualified_path=ROOT/'data/qualified_fresh_confirmation_clip_manifest.json'
    qualified=json.loads(qualified_path.read_text())
    if prep['qualified_manifest_sha256']!=sha256(qualified_path) or qualified['registered_manifest_sha256']!=lock['candidate_manifest_sha256'] or qualified['status']!='passed':
        raise ValueError('Fresh source qualification/reservation lineage failed')
    if {r['clip_id'] for r in prep['records']}!={r['clip_id'] for r in qualified['records']} or len(prep['records'])!=len(qualified['records']):
        raise ValueError('Fresh prepared cohort differs from qualification')
    d=ROOT/'results/development_probes_v3'
    if sha256(d/'development_selection.json')!=lock['development_selection_sha256'] or sha256(d/'development_models.npz')!=lock['model_archive_sha256']:
        raise ValueError('Development choices changed')
    if sha256(ROOT/'configs/development_v3.json')!=lock['feature_capture_protocol_sha256']:
        raise ValueError('Capture protocol changed')
    code_hashes={str(f.relative_to(ROOT)):sha256(f) for f in [Path(__file__),ROOT/'src/mira_interp/development_capture.py',ROOT/'src/mira_interp/model_loading.py']}
    if code_hashes!=lock['capture_code_sha256']:
        raise ValueError('Fresh capture code changed after registration')
    selection=prep['records'][args.worker::args.workers]
    out=Path('/data2/ishaangp/mira-interp/captures/fresh_confirmation_v3')
    report_path=ROOT/'results'/f'fresh_capture_worker{args.worker}.json'
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha256(lock_path),preparation_sha256=sha256(prep_path),code_sha256=code_hashes,
                expected=len(selection),records=[],errors=[],scope='fresh observational confirmation; no fitting')
    save(report_path,report)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model,loading=load_pretrained_four_player(Path('/data2/ishaangp/mira-interp/assets'))
    model.to('cuda')
    projection=channel_projection('cuda')
    report['model_loading']=loading
    save(report_path,report)
    start=time.monotonic()
    for rec in selection:
        try:
            if rec['split']!='confirmation':
                raise ValueError('Wrong cohort role')
            path=out/(rec['clip_id']+'.npz')
            if path.exists():
                raise ValueError('Fresh output exists; inspect it instead of silently repeating')
            inp=Path(rec['artifact_path'])
            if sha256(inp)!=rec['sha256']:
                raise ValueError('Prepared source hash changed')
            with np.load(inp,allow_pickle=False) as source:
                features,info=capture(model,source['frames'],source['actions'],seed=clip_seed(rec['clip_id']),projection=projection)
                features.update(y=source['targets'][:,1::2].reshape(32,30),target_names=source['target_names'],
                                sites=np.asarray(['block_0_input']+[f'block_{i}_output' for i in range(16)]),
                                view_index=np.repeat(np.arange(4),8),latent_frame_index=np.tile(np.arange(8),4),
                                source_frame_index=np.tile(source['source_frame_indices'][1::2],4),
                                timestamps=source['timestamps'][:,1::2].reshape(32),canonical_player_ids=np.tile(source['player_ids'],(32,1)),
                                match_ids=np.repeat(rec['match_id'],32),split=np.repeat('confirmation',32),clip_ids=np.repeat(rec['clip_id'],32))
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**features)
            side=dict(clip_id=rec['clip_id'],match_id=rec['match_id'],split='confirmation',path=str(path),sha256=sha256(path),input_sha256=rec['sha256'],
                      registration_sha256=report['registration_sha256'],code_sha256=code_hashes,**info)
            save(path.with_suffix('.json'),side)
            report['records'].append(side)
            save(report_path,report)
            print(json.dumps(dict(completed=len(report['records']),expected=len(selection))),flush=True)
        except Exception as e:
            report['status']='failed'
            report['errors'].append(dict(clip_id=rec['clip_id'],type=type(e).__name__,message=str(e)))
            save(report_path,report)
            raise
    report.update(status='passed',elapsed_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save(report_path,report)

if __name__=='__main__':
    main()
