# Revised development feature archive

This archive contains all 336 audited development feature captures: 31 discovery
matches and 11 selection matches, eight clips per match, 32 view/time rows per
clip. It also contains all derived arrays from the reserved numerical pilot.
Source physics labels, source videos, and actions are excluded. The fitted
`development_models.npz` archive is a separate release asset; its hash and size
are recorded in this archive's manifest.

Each `captures/{split}/{clip_id}.npz` retains the exact saved arrays:

- `mean_X[32,17,2048]`: all spatial means, block0 input and 16 block outputs.
- `spatial_X[32,17,1536]`: twelve 3x4 spatial bins, each projected to 128 channels.
- `codec_spatial_X[32,4608]`, `codec_mean_X[32,32]`, `RGB_X[32,1536]`: baselines.
  `RGB_X` deliberately retains derived 3x16x32 downsampled RGB values, flattened;
  it is not an original-resolution frame or source video.
- The complete original site names and row metadata: match/clip/split, view,
  latent index, source frame, timestamp, canonical player IDs, and target names.

Features use FP16 storage. Rows are view-major, then latent0..7; analysis scores
latent2..7. Timestamp and source frame refer to the final frame in each codec
pair, with each view's own alignment. Target names describe the omitted `y`
columns and contain no simulator values. No feature or metadata is rounded or
recomputed during publication.

`numerics_pilot.npz` retains FP32 arrays for three runtime variants:
`original_full_bf16`, `fp32_model_legacy_mix`, and `publisher_precision`. Each has
`codec[4,8,9,16,32]`, `interpolant[1,8,36,16,32]`,
`prediction[1,8,36,16,32]`, and `means[32,17,2048]`, plus the shared site names.
The codec axes are view,time,height,width,channel; interpolants and predictions
use batch,time,tiled-view-height,width,channel. These are model-derived values.
The numerical pilot uses the earlier uint8-prepared reserved clip, whereas the
336 development captures use unrounded resized RGB, as documented in the
included provenance reports.

The package requires passed capture and saved-probe audits bound to the exact
protocol, capture manifest, and model archive. It verifies every source NPZ hash,
uses an explicit permitted-array list, and never accesses source `y`. Inner NPZ
files use uncompressed ZIP entries; the outer tar uses gzip once. After writing,
every member is read back, every member byte hash is checked, and every NumPy
array is compared with its audited source fingerprint, shape and dtype.

`manifest.json` records every member and array fingerprint. Reports and
attribution are in `provenance/`; retain `THIRD_PARTY.md` when sharing. Reproducing
omitted source labels requires authorized access to Rocket Science and the
registered preparation scripts. These captures support observational decoding;
they do not establish physical control.

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -u scripts/package_development.py
```

The packaging report is `results/development_archive.json`. Existing final or
partial archives and existing reports are never silently overwritten.

The completed run passed in 60.925 seconds, reading back all 345 archive members
and comparing all 5,053 arrays with their source fingerprints. The archive is
1,340,553,456 bytes, below the 2GB asset limit, with SHA256
`a2b575b03c229292999f9632c62ec87f7d30c2ace776b18a4609152a412182ef`.
The archive contains the pre-completion version of this README; the audit
manifest records its exact bytes. Independent code review found no material
integrity or source-array exclusion gap.
