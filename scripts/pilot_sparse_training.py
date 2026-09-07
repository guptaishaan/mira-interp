#!/usr/bin/env python3
"""Full-width synthetic GPU feasibility check after the geometry audit."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from mira_interp.feature_geometry import train_dictionary
from analyze_feature_geometry import registered_context


def main():
    protocol_path = ROOT / 'configs/feature_development_v1.json'
    p = json.loads(protocol_path.read_text())
    registered_context(protocol_path, Path(p['input_path']), Path(p['selected_probe_path']), Path(p['gate_path']))
    audit_path = ROOT / 'results/geometry_development_v1/geometry_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed':
        raise ValueError('Geometry audit must pass before sparse training')
    out = ROOT / 'results/feature_development_v1/synthetic_pilot.json'
    if out.exists():
        raise ValueError('Do not overwrite a completed synthetic pilot')
    if not torch.cuda.is_available():
        raise ValueError('Explicit authorized CUDA device required')
    torch.set_num_threads(4)
    rng = np.random.default_rng(20260907)
    data = dict(X=rng.standard_normal((128, 1536)).astype(np.float32), velocity=rng.standard_normal((128, 3)),
                match_ids=np.repeat(['synthetic_a','synthetic_b','synthetic_c','synthetic_d'], 32),
                split=np.repeat(['discovery','discovery','selection','selection'], 32),
                clip_ids=np.repeat(['a0','b0','c0','d0'], 32), entity_ids=np.full(128,'synthetic_label'),
                view_index=np.zeros(128, int), frame_index=np.tile(np.arange(32), 4), timestamps=np.tile(np.arange(32)*.1, 4))
    started = time.monotonic(); reports = []
    for kind, temporal, control in [('relu',0.,'adjacent'), ('signed',0.,'adjacent'), ('block',0.,'adjacent'),
                                     ('block',.1,'adjacent'), ('block',.1,'shuffled_within_match')]:
        model, normalization, report = train_dictionary(data, kind=kind, width=p['sparse']['width'], active=64,
            steps=20, seed=20260907, device='cuda', temporal_weight=temporal, temporal_control=control,
            view_identity_verified=True)
        if any(not torch.isfinite(value).all() for value in model.parameters()):
            raise ValueError('Nonfinite trained parameters')
        torch.testing.assert_close(model.decoder.norm(dim=1), torch.ones(3072), rtol=1e-6, atol=1e-6)
        reports.append(report)
    result = dict(status='passed', scope='synthetic full-width feasibility only; no research examples, models or results',
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  geometry_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), elapsed_seconds=time.monotonic()-started,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(), synthetic_models_discarded=True, reports=reports)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status','elapsed_seconds','peak_gpu_bytes')}))


if __name__ == '__main__':
    main()
