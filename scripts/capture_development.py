#!/usr/bin/env python3
"""Capture corrected development observations after preparation/pilot gates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'external/mira/src')]
from mira_interp.model_loading import load_pretrained_four_player, sha256
from mira_interp.development_capture import channel_projection, capture
sys.path.insert(0, str(ROOT/'scripts'))
from capture_observations import clip_seed
from prepare_development import save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--aggregate', action='store_true')
    args=p.parse_args()
    out=Path('/data2/ishaangp/mira-interp/captures/development_v3')
    prep_path=ROOT/'results'/('development_prepare_pilot.json' if args.pilot else 'development_prepare.json')
    prep=json.loads(prep_path.read_text())
    if prep['status'] != 'passed':
        raise ValueError('Preparation gate must pass first')
    records=prep['records']
    code_hashes={str(f.relative_to(ROOT)):sha256(f) for f in [Path(__file__), ROOT/'src/mira_interp/development_capture.py', ROOT/'src/mira_interp/model_loading.py']}
    protocol=ROOT/'configs/development_v3.json'
    registration=sha256(protocol)
    if not args.pilot:
        pilot=json.loads((ROOT/'results/development_capture_pilot.json').read_text())
        if pilot['status']!='passed' or pilot['code_sha256']!=code_hashes or pilot['registration_sha256']!=registration:
            raise ValueError('Passed pilot for exactly this code and protocol required')
    if args.aggregate:
        gathered=[]
        for rec in records:
            path=out/rec['split']/(rec['clip_id']+'.npz')
            side=json.loads(path.with_suffix('.json').read_text())
            if side['input_sha256']!=rec['sha256'] or side['sha256']!=sha256(path) or side['code_sha256']!=code_hashes or side['registration_sha256']!=registration:
                raise ValueError('Capture provenance failed')
            with np.load(path,allow_pickle=False) as d:
                if set(d['split'].astype(str)) != {rec['split']} or set(d['match_ids'].astype(str)) != {rec['match_id']} or 'confirmation' in d['split']:
                    raise ValueError('Role/identity contamination')
                if d['mean_X'].shape!=(32,17,2048) or d['spatial_X'].shape!=(32,17,1536) or not all(np.isfinite(d[k]).all() for k in ('mean_X','spatial_X','codec_spatial_X','y')):
                    raise ValueError('Invalid arrays')
                original=next(r for r in json.loads((ROOT/'data/qualified_clip_manifest.json').read_text())['records'] if r['clip_id']==rec['clip_id'])
                with np.load(original['artifact_path'],allow_pickle=False) as source:
                    if not np.array_equal(d['y'],source['targets'][:,1::2].reshape(32,30)):
                        raise ValueError('Targets not exactly source-aligned')
            gathered.append(side)
        counts={s:len({r['match_id'] for r in gathered if r['split']==s}) for s in ('discovery','selection')}
        if counts!={'discovery':31,'selection':11} or len(gathered)!=336:
            raise ValueError('Incomplete development cohort')
        result=dict(status='passed',registration_sha256=registration,code_sha256=code_hashes,records=gathered,
                    counts=counts,rows=32*len(gathered),confirmation_used=False,scope='development only')
        save(ROOT/'results/development_capture_manifest.json', result)
        print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
        return
    report_path=ROOT/'results'/('development_capture_pilot.json' if args.pilot else f'development_capture_worker{args.worker}.json')
    selected=records[args.worker::args.workers]
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),code_sha256=code_hashes,
                registration_sha256=registration,input_manifest_sha256=sha256(prep_path),expected=len(selected),completed=[],errors=[])
    save(report_path,report)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model, loading=load_pretrained_four_player(Path('/data2/ishaangp/mira-interp/assets'))
    model=model.to('cuda')  # preserve released FP32 parameters and buffers
    if any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Released FP32 checkpoint dtype not preserved')
    projection=channel_projection('cuda')
    report['model_loading']=loading
    save(report_path,report)
    start=time.monotonic()
    for rec in selected:
        try:
            if rec['split'] not in (['pilot'] if args.pilot else ['discovery','selection']):
                raise ValueError('Forbidden role')
            path=out/rec['split']/(rec['clip_id']+'.npz')
            side_path=path.with_suffix('.json')
            if side_path.exists():
                cached=json.loads(side_path.read_text())
                if cached['input_sha256']!=rec['sha256'] or cached['sha256']!=sha256(path) or cached['code_sha256']!=code_hashes or cached['registration_sha256']!=registration:
                    raise ValueError('Stale capture cache')
                report['completed'].append(cached)
                save(report_path,report)
                continue
            input_path=Path(rec['artifact_path'])
            if sha256(input_path)!=rec['sha256']:
                raise ValueError('Prepared input hash mismatch')
            with np.load(input_path,allow_pickle=False) as d:
                # Physical labels are accessed only AFTER the model forward.
                arrays, info=capture(model,d['frames'],d['actions'],seed=clip_seed(rec['clip_id']),projection=projection,pilot=args.pilot)
                arrays.update(y=d['targets'][:,1::2].reshape(32,30), target_names=d['target_names'],
                              sites=np.asarray(['block_0_input']+[f'block_{i}_output' for i in range(16)]),
                              view_index=np.repeat(np.arange(4),8), latent_frame_index=np.tile(np.arange(8),4),
                              source_frame_index=np.tile(d['source_frame_indices'][1::2],4),
                              timestamps=d['timestamps'][:,1::2].reshape(32), canonical_player_ids=np.tile(d['player_ids'],(32,1)),
                              match_ids=np.repeat(rec['match_id'],32), split=np.repeat(rec['split'],32), clip_ids=np.repeat(rec['clip_id'],32))
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**arrays)
            side=dict(clip_id=rec['clip_id'],match_id=rec['match_id'],split=rec['split'],path=str(path),artifact_path=str(path),
                      sha256=sha256(path),input_sha256=rec['sha256'],registration_sha256=registration,code_sha256=code_hashes,**info)
            save(side_path,side)
            report['completed'].append(side)
            print(json.dumps(dict(completed=len(report['completed']),expected=len(selected),elapsed_seconds=time.monotonic()-start)),flush=True)
            save(report_path,report)
        except Exception as e:
            report['errors'].append(dict(clip_id=rec['clip_id'],type=type(e).__name__,message=str(e)))
            report['status']='failed'
            save(report_path,report)
            raise
    report.update(status='passed',elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    save(report_path,report)

if __name__=='__main__':
    main()
