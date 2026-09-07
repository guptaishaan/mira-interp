# Independent development-capture audit

`scripts/audit_development_capture.py` checks the corrected development capture
without running a model or fitting probes. It verifies the exact original 31
discovery and 11 selection matches, eight clips per match, before opening any
feature or label NPZ. Old confirmation files are excluded.

The audit checks the current registered protocol, completed worker records,
strict model-loading evidence, reserved-pilot controls, source and prepared-file
hashes, feature dimensions/dtypes/finite values, all 17 sites, and every saved
label, timestamp, source frame and canonical player identity against both the
original and unrounded prepared inputs. It rejects duplicate source-view/frame
rows and checks the exact two-worker modulo assignment.

The spatial descriptor receives a separate mathematical check: a small
nonresearch fixture is processed by the implementation and by explicit nested
view/time/bin slices. It checks 3x4 bins, each covering three rows and four columns
of a 9x16 view grid. The projection is a fixed Gaussian QR map (seed 20260907),
independent of observations and labels. Its orthogonality and reconstructed
matrix hash are recorded. Workers did not save the exact matrix, so the audit
does not claim that the recorded hash was directly measured inside each worker.

Every captured clip also checks two redundant identities: full codec tokens
average to the saved codec mean, and the average of equal-area projected bins
agrees with the projected global mean. These checks allow recorded FP16 storage
rounding (`rtol=0.001, atol=0.00002` for codec). For the projected identity the
auditor propagates the FP16 rounding bound `u/(1-u)*abs(saved)+2^-25`, with
`u=2^-11`, through `abs(Q)` and the bin average using FP64 reference arithmetic.
The `2^-25` term bounds half of the FP16 subnormal spacing. This handles cancellation in
near-zero projected coordinates. RGB descriptors are independently recomputed on the
reserved pilot and first development clip; other clips retain the verified
preparation/hash chain. Per-clip reports explicitly mark this coverage.

The preparation helper originally had a strict RGB range check. The later
tolerance repair affected only `make_inputs`, which preparation does not call.
The original whole-file hash is reproduced by the preserved source snapshot in
`results/development_attempts/helper_before_rgb_tolerance.py.txt`. The auditor
verifies that old/current ASTs are identical after excluding that one function;
the decoder, imports and globals are unchanged.

The reserved descriptor pilot passed in 2.2 seconds. Explicit-slice mean error
was zero, maximum projected-bin discrepancy was 2.15e-6, and maximum QR
orthogonality discrepancy was 2.39e-7. Its RGB descriptor matched exactly.
See [the pilot audit](../../results/review/development_descriptor_pilot.json).
This pilot alone cannot authorize development fitting.

The first full audit stopped on a fixed-tolerance check at clip 13. A single
coordinate differed by 0.001138 while its propagated FP16 rounding budget was
0.003880; cancellation made a relative tolerance on the final value misleading.
The original failure is preserved in
`results/development_attempts/capture_audit01_fixed_tolerance.json`. The corrected
audit uses the analytic storage-rounding bound above; capture code and outputs
were unchanged.

The complete audit passed all 336 clips and 10,752 source-view/time rows in
44.968 seconds. Every source join and required array passed. The independent
reviewer agreed that a propagated rounding bound is the appropriate engineering
check; no research labels or performance measurements set that bound.

```bash
NUMPY_MADVISE_HUGEPAGE=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_development_capture.py --pilot-only \
  --report results/review/development_descriptor_pilot.json

# After both complete workers and the combined manifest pass:
NUMPY_MADVISE_HUGEPAGE=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_development_capture.py \
  --manifest results/development_capture_manifest.json \
  --report results/development_capture_audit.json
```

The full report binds `capture_manifest_sha256`, `registration_sha256` and
`split_manifest_sha256` to the exact artifacts. It reports exact source joins
and all-layer completion only after every required record passes. It makes no
physical-causality claim and does not reinterpret old confirmation data as new.
