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
