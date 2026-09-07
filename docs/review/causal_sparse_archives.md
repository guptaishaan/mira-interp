# Causal and sparse publication archives

`scripts/package_research_artifacts.py` produces separate archives for the completed causal and sparse development stages. It performs no Git, release, or network operation.

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/package_research_artifacts.py --stage causal
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/package_research_artifacts.py --stage sparse
```

Default outputs are under `/data2/ishaangp/mira-interp/releases/`, named `causal-development-v1-2026-09-07.tar.gz` and `sparse-development-v1-2026-09-07.tar.gz`. The corresponding manifests are `results/causal_development_archive.json` and `results/sparse_development_archive.json`. Existing archives and reports are immutable. A conservative 2,000,000,000-byte limit is enforced for each asset.

The causal archive retains all 85 pilot/discovery/selection pair tensor archives, including the complete 660 selection intervention outputs and their derived codec/activation tensors. Per-observation simulator-state arrays are excluded. Pair reports remove exact recipient/donor source-height fields; all experimental measurements and original source hashes remain. Report bytes in the archive can therefore differ from private originals, with both hashes recorded.

The sparse archive retains all 30 trained dictionaries, the 8,064 selected descriptors and their row identities, frozen geometry maps and figures, independent audits, and derived retention arrays. The per-observation source position and velocity arrays are excluded. Dictionary checkpoints are the study's learned featurizers, not upstream world-model weights. Aggregate physical-bin coordinates and model-derived probe predictions remain explicit derived study results.

The archive writer pins each input's hash, normalizes tar metadata, and compresses once. Every finished member is streamed back and checked against its exact byte hash. All NPZ arrays are additionally checked against source fingerprints, and every original file is rehashed at completion. Reproducing analyses that require source labels still requires obtaining Rocket Science through its normal access process.

Both local archives completed readback verification:

| Archive | Bytes | SHA256 |
|---|---:|---|
| causal-development-v1-2026-09-07.tar.gz | 1,116,295,253 | `2eef5d5a14d876c9bbcb02767ac999a3d47586643a91018586f84d820d96a2ba` |
| sparse-development-v1-2026-09-07.tar.gz | 1,090,009,905 | `9fe4e9f01e9aa212aab37fb14c3414f66d13cc430a7aadeb15f5ded39aaa7998` |

These manifests describe locally prepared files. Separate publication records must verify any subsequent remote release size and digest.
