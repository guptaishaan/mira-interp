# Final execution and publication audit

`scripts/audit_registered_rollout_completion.py` checks whether the fixed
selection and confirmation runs have finished and their complete generated
artifacts are available from GitHub. Its successful status is
`passed_execution_and_publication_audit`. Physical control and generated-video
3D accuracy remain explicitly unestablished.

Run this only after both phases finish generation, measurement, export,
packaging, independent package review, and publication:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/audit_registered_rollout_completion.py \
  --report results/rollout_steering_v2/execution_and_publication_audit.json
```

The script requires exactly eight ordered passed stages and their current
report hashes. It joins each generation grid to its registration, generation
audit, measurement audit, and packaged record IDs: 420 selection conditions
from 10 matches and 924 confirmation conditions from 22 matches, with the same
two paired seeds and 21 assignments per match/seed. It checks all five paths
and four measurement times remain in the saved reports.

For each phase it verifies the eight PNG/PDF files and six CSVs against the
export manifest, the package's exact source/report hashes, every part sidecar,
and the independent package review. Source array slices and metric formulas
were checked by the earlier bound audits; this final script does not decode
arrays or rerun those scientific calculations. Visual inspection of all four
plots is a separate review, not something this script claims to perform.

GitHub calls use `gh api --method GET` only. The auditor resolves the actual
release tag through annotated tags to a commit, requires the recorded target,
and compares the exact remote asset set, IDs, uploaded status, sizes, SHA256
digests, and release notes with the verified local publication metadata. It
performs no upload, release creation, Git changes, or model execution.

By default, local multipart TAR files are checked for existence and size.
Their bytes have already been hashed by the packager, independent reviewer,
and publisher; current remote digests are checked again. Small metadata and
export files are rehashed, including a final check for changes during the
audit. `--rehash-parts` optionally repeats full local archive hashing. This
explicit distinction avoids presenting prior byte verification as a new local
rehash.

The default publication reports are
`results/generated_selection_v2_publication.json` and
`results/generated_confirmation_v2_publication.json`. Their paths can be
provided through `--selection-publication` and `--confirmation-publication`.
The README must retain the two figures and 113 source-whitespace words bound
by the independent initial-progress presentation review. Other presentation
and scientific claims are not inferred from execution completion.

A running, failed, missing, or stale prerequisite raises an error. No report
is written on failure; an existing report is never overwritten. Focused tests
use temporary metadata and mocked GitHub responses, including incomplete
stages, reordered or duplicated stages, changed source hashes, partial or
substituted assets, incorrect tag commits, and stale package reviews.
