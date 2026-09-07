#!/usr/bin/env python3
"""Download the pinned public VideoMAE encoder bundle; no GPU or installation."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = 'MCG-NJU/videomae-base'
REVISION = 'dc740ceda42fce44faed2ea03c6d447db72f6af9'
FILES = ('config.json', 'preprocessor_config.json', 'README.md', 'model.safetensors')
WEIGHT_SHA = 'bc053ca2840a038b1068269a4eec06ca569689e9a1ed9376a5b2b8a111be5290'
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('/data2/ishaangp/mira-interp/models/videomae-base'))
    p.add_argument('--report', type=Path, default=ROOT/'results/videomae_access.json')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Model revision mismatch')
    indexed = {item.rfilename: item for item in info.siblings}
    total = sum(indexed[name].size for name in FILES)
    if total > 512 * 1024**2 or shutil.disk_usage(args.output_dir).free < total + 25 * 1024**3:
        raise ValueError('Download budget or disk reserve exceeded')
    files = []
    for name in FILES:
        path = Path(hf_hub_download(REPO, name, revision=REVISION, token=False, local_dir=args.output_dir))
        actual = sha256(path)
        if path.stat().st_size != indexed[name].size or (name == 'model.safetensors' and actual != WEIGHT_SHA):
            raise ValueError(f'Publisher integrity mismatch: {name}')
        files.append({'name': name, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual})
    report = {'status': 'passed_download_integrity', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'model_repo': REPO, 'revision': REVISION, 'source': f'https://huggingface.co/{REPO}/tree/{REVISION}',
              'files': files, 'total_bytes': total, 'weight_lfs_sha256_verified': True,
              'gpu_used': False, 'dependencies_installed': False,
              'scientific_scope': 'asset access only; evaluator accuracy and generated-domain reliability untested',
              'script_sha256': sha256(Path(__file__))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
