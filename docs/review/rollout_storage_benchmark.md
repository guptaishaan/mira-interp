# Pilot output storage benchmark

Both completed pilot NPZ files were rewritten using three lossless ZIP storage choices, with two repetitions per condition and rotated format order. All twelve outputs were reopened. Every array's keys, shape, dtype, byte hash, and values exactly matched its original; the original files and manifest remained unchanged. Default compression reproduced the original archive SHA256 values exactly. No model execution, fitting, or protocol change occurred.

| Format | Median write including flush/fsync | Median warm read | Mean bytes per condition | Projected bytes for 1,344 conditions |
|---|---:|---:|---:|---:|
| NumPy default DEFLATE | 8.08174 s | 0.36060 s | 36,229,672 | 48,692,679,168 |
| NumPy uncompressed | 0.15359 s | 0.12457 s | 173,127,204 | 232,682,962,176 |
| DEFLATE level 1 | 1.44582 s | 0.45699 s | 45,248,577 | 60,814,087,488 |

Level 1 reduces median write time by 6.64 seconds per output while adding about 9.02 MB. For 1,344 outputs, the measured rates imply approximately 149 minutes less serial writing, or 74 minutes less wall time with two equal-throughput writers. Uncompressed output would save another approximately 14.5 minutes across two writers but consume an additional 171.87 GB. The parent workflow selected level 1 as a prospective storage-only amendment, with a new registered pilot and exact parity checks before the full run.

These are timing projections from two reserved conditions, not a guarantee for other videos or concurrent workloads. Source arrays were already in memory. Write timing includes ZIP closure, flush, and `fsync`; hashing and verification are excluded. Reads happened after file hashing and therefore used a warm filesystem cache. No system caches were dropped. Available disk space before the benchmark was 607,292,297,216 bytes; all three projected totals preserve the 25 GiB reserve under that snapshot.

Evidence: `results/rollout_storage_benchmark.json`, SHA256 `327f2ee15af51fa555829352eac5949e7bafd355d9747292cca1669ed20ca798`. It contains exact source and output paths/hashes, every array hash, all twelve timings, free-space observations, and projection assumptions. Temporary benchmark outputs are confined to `/data2/ishaangp/mira-interp/storage_benchmark_v1/`.

Reproduce into new destinations:

```bash
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  scripts/benchmark_rollout_storage.py \
  --output-dir /data2/your-new-benchmark-directory \
  --report /data2/your-new-benchmark-report.json
```

The script refuses to replace a completed report or existing trial files. Changing ZIP compression does not change loaded array content; the parent must separately verify this invariant through the new complete inference pilot before changing the main rollout writer.

## Complete inference parity passed

The separate version2 producer passed seven CPU writer tests and an independent source-diff review (`results/rollout_storage_v2_review.json`, SHA256 `3fea6ab6159badb1b498245c2e56f785595b54576f4b1284cc96b25d2e253683`). Its changes are restricted to storage, associated provenance, and default output directories. Rewriting the full original baseline through the new helper also exactly reproduced the independently benchmarked level-1 archive hash; see `results/rollout_storage_helper_check.json`.

A newly registered five-inference version2 pilot then passed its generation audit. The independent parity auditor compared all six arrays in both saved conditions against version1 and found every bit unchanged. Action hashes, all sampler calls, raw decoder hashes, intervention site, probe shifts, and common scientific registration/code hashes also matched exactly. Both independent generation audits were prerequisites. The parity report is `results/rollout_steering_v2/pilot_parity_audit.json`, status `passed_storage_only_pilot_parity`, SHA256 `0ecc2952d76256eff9e83b1f40eeee137f5d09721b86a5ebee3c5d9a609a2e5e`.

The version2 inference/save loop took 33.91 seconds versus 46.95 seconds for version1, and its two saved files total 90.50 MB versus 72.46 MB. The difference is consistent with the isolated write benchmark. These checks establish lossless storage and unchanged pilot outputs; they add no physical-steering evidence.

Run the independent parity audit only after both complete generation audits pass:

```bash
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  scripts/audit_rollout_storage_parity.py
```

The auditor permits only explicitly named encoding, path, registration, and runtime metadata differences. Its CPU corruption tests reject altered action hashes, model-call records, doses, unknown metadata, array dtypes/shapes, and even a changed sign bit on zero.
