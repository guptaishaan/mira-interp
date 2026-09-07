#!/usr/bin/env python3
"""Prepare fractional RGB for already-qualified fresh matches without model evaluation."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from mira_interp.development_capture import decode_float_video
from mira_interp.model_loading import sha256
from prepare_development import save
import torch
torch.set_num_threads(1)
source=ROOT/'data/qualified_fresh_confirmation_clip_manifest.json'
manifest=json.loads(source.read_text())
if manifest['status']!='passed':
    raise ValueError('Fresh qualification must pass')
report_path=ROOT/'results/fresh_confirmation_float.json'
report=dict(status='running',qualified_manifest_sha256=sha256(source),code_sha256=sha256(Path(__file__)),
            helper_sha256=sha256(ROOT/'src/mira_interp/development_capture.py'),records=[],errors=[],expected=len(manifest['records']))
save(report_path,report)
out=Path('/data2/ishaangp/mira-interp/prepared_fresh_float')
def work(row):
    if row['role']!='confirmation':
        raise ValueError('Fresh confirmation only')
    p=Path(row['artifact_path'])
    if sha256(p)!=row['sha256']:
        raise ValueError('Source prepared bytes changed')
    with np.load(p,allow_pickle=False) as old:
        kept={k:old[k] for k in old.files if k!='frames'}
        pixels=old['frames']
    selected=list(range(row['plan']['local_start'],row['plan']['local_start']+16))
    key=f"{row['match_id']}_c{row['chunk_index']:05d}"
    frames,pts=[],[]
    with tarfile.open(Path('/data2/ishaangp/mira-interp/rocket-science/dev')/row['source_shard']) as tar:
        for v in range(4):
            b=tar.extractfile(f'{key}.p{v}.mp4').read()
            if hashlib.sha256(b).hexdigest()!=row['source_component_sha256'][f'p{v}.mp4']:
                raise ValueError('Source video changed')
            f,t=decode_float_video(b,selected,row['video_audit'][v]['decoded_frame_count'])
            frames.append(f);pts.append(t)
    frames=np.stack(frames)
    if not np.array_equal(np.rint(frames).clip(0,255).astype(np.uint8),pixels) or not np.array_equal(pts,kept['video_pts']):
        raise ValueError('Source pixels or PTS disagree')
    out.mkdir(parents=True,exist_ok=True)
    path=out/(row['clip_id']+'.npz')
    np.savez_compressed(path,frames=frames,**kept)
    r=dict(clip_id=row['clip_id'],match_id=row['match_id'],split='confirmation',artifact_path=str(path),
           sha256=sha256(path),parent_sha256=row['sha256'],round_to_previous_pixels_exact=True,source_pts_exact=True)
    save(path.with_suffix('.json'),r)
    return r
with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(work,r) for r in manifest['records']]
    for row,f in zip(manifest['records'],futures):
        try:
            report['records'].append(f.result())
        except Exception as e:
            report['errors'].append(dict(clip_id=row['clip_id'],type=type(e).__name__,message=str(e)))
        save(report_path,report)
        print(f'{len(report["records"])}/{report["expected"]}',flush=True)
report['status']='passed' if not report['errors'] and len(report['records'])==report['expected'] else 'failed'
save(report_path,report)
if report['status']!='passed':
    raise SystemExit(1)
