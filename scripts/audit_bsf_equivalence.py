#!/usr/bin/env python3
"""Compare our descriptor adapter against pinned official VanillaBSF algebra."""
import os
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'external/block-sparse-featurizer'
PIN = '219f121ea82d2b19200d1dac918396e6058d7eb9'
sys.path[:0] = [str(ROOT / 'src'), str(EXTERNAL)]
from mira_interp.feature_geometry import TopKDictionary
from bsf.vanilla import VanillaBSF


def main():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=EXTERNAL, text=True).strip()
    if revision != PIN:
        raise ValueError('Official BSF revision changed')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=EXTERNAL, text=True).strip():
        raise ValueError('Official BSF source tree has local changes')
    torch.set_num_threads(1)
    torch.manual_seed(20260907)
    reports = []
    for group in (1, 8):
        ours = TopKDictionary(32, width=64, active=16, kind='signed' if group == 1 else 'block', group_size=group).double()
        reference = VanillaBSF(32, n_groups=64 // group, group_size=group, l0=16 // group).double()
        with torch.no_grad():
            reference.W_enc.copy_(ours.encoder.weight.T)
            reference.b_enc.copy_(ours.encoder.bias)
            reference.W_dec.copy_(ours.decoder)
        x = torch.randn(19, 32, dtype=torch.float64)
        a, az = ours(x)
        b, bz = reference(x)
        torch.testing.assert_close(az, bz.flatten(1), rtol=0, atol=0)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        (a - x).square().mean().backward()
        reference.loss(x)[0].backward()
        comparisons = [(ours.encoder.weight.grad.T, reference.W_enc.grad),
                       (ours.encoder.bias.grad, reference.b_enc.grad),
                       (ours.decoder.grad, reference.W_dec.grad)]
        maximum = max(float((a - b).abs().max()) for a, b in comparisons)
        for a, b in comparisons:
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
        ours.normalize_decoder(); reference.normalize_decoder()
        torch.testing.assert_close(ours.decoder, reference.W_dec, rtol=0, atol=0)
        reports.append(dict(group_size=group, code_and_output_exact=True,
                            maximum_gradient_error=maximum, normalized_decoder_exact=True,
                            trainable_parameters=sum(p.numel() for p in ours.parameters())))
    result = dict(status='passed', official_revision=PIN, scope='signed scalar and VanillaBSF forward, loss-gradient and decoder-normalization equivalence at matched weights; not a published training-result replication',
                  checks=reports, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [EXTERNAL/'bsf/base.py', EXTERNAL/'bsf/vanilla.py', ROOT/'src/mira_interp/feature_geometry.py']})
    path = ROOT / 'results/bsf_equivalence.json'
    if path.exists():
        raise ValueError('Preserve completed BSF equivalence audits')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
