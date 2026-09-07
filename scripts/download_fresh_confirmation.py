#!/usr/bin/env python3
"""Fetch only metadata-registered fresh-confirmation shards, with publisher hashes."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mira_interp.model_loading import sha256
from huggingface_hub import HfApi,hf_hub_download
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_development import save

manifest=ROOT/'data/fresh_confirmation_split_manifest.json'
m=json.loads(manifest.read_text())
dest=Path('/data2/ishaangp/mira-interp/rocket-science')
names={'dev/'+r['shard'] for r in m['matches'] if r['split']=='confirmation'}|{'dev/index.json'}
api=HfApi().dataset_info('kyutai/rocket-science',revision=m['source_revision'],files_metadata=True)
selected=sorted([f for f in api.siblings if f.rfilename in names],key=lambda f:f.rfilename)
if len(selected)!=len(names) or api.sha!=m['source_revision']:
    raise ValueError('Pinned source inventory mismatch')
total=sum(f.size for f in selected)
if total>100*2**30 or shutil.disk_usage(dest).free<total+25*2**30:
    raise ValueError('100GiB budget or25GiB disk reserve exceeded')
report_path=ROOT/'results/fresh_confirmation_download.json'
report=dict(status='RUNNING',revision=m['source_revision'],manifest_sha256=sha256(manifest),expected_bytes=total,files=[])
save(report_path,report)
for f in selected:
    p=Path(hf_hub_download('kyutai/rocket-science',f.rfilename,repo_type='dataset',revision=m['source_revision'],local_dir=dest))
    digest=sha256(p)
    expected=f.lfs.sha256 if f.lfs else None
    if p.stat().st_size!=f.size or (expected and digest!=expected):
        raise ValueError('Publisher hash/size failed')
    if not expected and digest not in m['input_index_sha256']:
        raise ValueError('Registered index mismatch')
    report['files'].append(dict(path=f.rfilename,bytes=f.size,sha256=digest,publisher_sha256=expected,verified=True))
    save(report_path,report)
    print(f'{len(report["files"])} / {len(selected)} verified: {f.rfilename}',flush=True)
report['status']='PASS'
save(report_path,report)
