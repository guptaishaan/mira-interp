# Fresh-confirmation feature archive

This archive preserves all 184 audited feature captures from 23 fresh matches,
eight clips per match. Each clip contains 32 rows, view-major then latent0..7,
for 5,888 captured rows. The frozen evaluation scores latent2..7, giving 4,416
rows. No coefficients are fitted and no hypothesis is selected during packaging.

Each `captures/{clip_id}.npz` retains these exact arrays:

- `mean_X[32,17,2048]` and `spatial_X[32,17,1536]`: all17 residual sites.
- `codec_spatial_X[32,4608]`, `codec_mean_X[32,32]`, and `RGB_X[32,1536]`.
- All original row metadata: site and target names, canonical player IDs,
  match/clip/split, view and latent index, source frame index, and timestamp.

The feature arrays retain FP16 storage. `RGB_X` deliberately includes derived
3x16x32 downsampled RGB values flattened per row. Original-resolution frames,
source videos, actions, and simulator-label arrays (`y`/`targets`) are excluded.
Target names only describe the omitted columns. Per-view source timestamps
remain distinct; these are not assumed to be identical camera times.

`provenance/` contains the fixed protocols and candidate cohort, quality
qualification, original source audit, preparation/worker/capture reports,
confirmation metrics, independent metrics audit, and model publication record.
The original source audit remains **failed** because some candidate matches
failed alignment. The unchanged quality gates admitted23 matches. The qualified
cohort and completed independent confirmation audit are separate passed reports;
packaging does not relabel the failed original audit.

The fitted model archive is already a separate development-release asset. Its
verified URL, size and GitHub SHA256 digest are recorded in `manifest.json`, so
this archive does not duplicate its bytes. `THIRD_PARTY.md` supplies attribution.
Access to omitted Rocket Science source labels requires accepting the dataset's
terms and reproducing the registered preparation.

`scripts/package_fresh_confirmation.py` requires passed independent capture and
metrics audits bound to the exact frozen choices. It hashes every source capture
before and after reading, uses an explicit array allowlist, and never loads `y`.
Inner NPZ files use ZIP_STORED; gzip compresses the outer tar once. Every completed
archive member is read back and hashed, and every array is compared against its
source fingerprint, shape and dtype. Final checks rehash all sources and reports.

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -u \
  scripts/package_fresh_confirmation.py
```

The local package report is `results/fresh_confirmation_archive.json`. This
script never creates a GitHub release or uploads files. Fresh-match decoding
remains teacher-forced annotation decoding, and negative velocity results remain
visible in the included metrics; no physical-control claim is established.
