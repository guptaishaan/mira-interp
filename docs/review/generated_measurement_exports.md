# Audited generated-measurement exports

`scripts/export_generated_measurements.py` exports every registered path and time from a completed, independently audited measurement phase. It performs no fitting, intervention selection, or new model inference. The full study supervisor owns measurement and audit execution; do not start a second copy of either stage.

The exporter checks the passed measurement audit **before reading the result report**, then binds the report, measurement protocol, frozen execution code, generation manifest/audit, and saved prediction archive by SHA256. It independently joins each saved prediction to its generation record, shared baseline, requested dose, and clipped effective dose. Its final report rechecks those bindings. Existing output directories are immutable; use a new directory when reproducing an export.

After the corresponding `selection_audit.json` or `confirmation_audit.json` has passed:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/export_generated_measurements.py \
  --phase selection \
  --protocol configs/generated_evaluation_v1.json \
  --evaluation-dir results/generated_evaluation_v2 \
  --generation-dir results/rollout_steering_v2
```

Use `--phase confirmation` for the separately audited confirmation phase. Outputs default to `results/generated_evaluation_v2/<phase>_figures/`; `export.json` records every output hash and input binding. A successful export does not replace either independent audit.

`scripts/run_generated_exports.py` is a separate CPU watcher for the already running study. It waits for the study supervisor to record each matching measurement audit as a completed stage, checks that audit's hash and full-phase flags, then runs the exporter once. It never launches or repeats generation, measurement, or audits. It disables CUDA, checks the source hashes in `results/generated_exporter_review.json`, stops before a new export if the study has failed or its own reviewed source changed, and has a six-hour total wait/execution budget. Termination or timeout cleans up only its owned exporter process group and records failure. Its status and logs live under `results/generated_evaluation_v2/export_watcher/`; that directory must not already exist when starting a new watcher.

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/run_generated_exports.py
```

## Figures and tables

All five paths appear in all four time points: probe linear, affine map, quadratic map, norm-matched random, and wrong-variable ball x. The exporter averages the two paired seeds within each match, then weights matches equally. It uses the registered 1,000 whole-match bootstrap draws and seed for the display bands, reproducing the already audited extreme-contrast means and intervals. Dose-response bands are descriptive pointwise intervals, with no simultaneous-coverage or evaluator-accuracy guarantee.

| PNG/PDF pair | Contents |
| --- | --- |
| `dose_response_requested` | View 0 estimated ball-height change versus all five requested doses, separately at 0.1, 0.2, 0.3 and 0.4 seconds. |
| `dose_response_effective` | The same responses and bands, positioned at the mean effective dose within each original requested-dose group. |
| `contrasts_controls_nuisance` | Signed high-minus-low dose contrast; paired differences from random and wrong-variable controls; standardized RMS change in the 11 non-target role coordinates. |
| `ordering_and_undefined` | Nondecreasing responses with nonzero span, eligible-match Spearman means, undefined rank counts, and inconsistent duplicate-dose responses. |

The effective-dose plot preserves the five original assignment groups. Different pairs can receive different effective doses within a group, and groups can overlap after clipping. It is not a fitted relationship at one common effective dose. The accompanying CSV reports the group mean, minimum, maximum, and clipping fraction.

Six CSVs retain the full summary, pair/time, pair/dose, match/time, match/dose, and aggregate dose-response tables. CSV values are read back and compared exactly; line endings are LF. Missing rank statistics remain empty fields, and their counts remain explicit. Negative responses and unsuccessful paths are retained. PNG files are decoded for verification, PDF headers are checked, and each completed phase's figures should also be visually inspected.

Every plot describes **learned video estimates**. The frozen VideoMAE evaluator's accuracy on generated videos has not been established; these figures alone cannot establish physical control or reliable preservation of unrelated physical properties.

## Engineering validation

Only synthetic cohorts were used while the registered rollout study was running. Eight tests cover positive, negative and constant responses; equal-match bootstrap bands; clipped-dose assignment groups; complete CSV round trips; refusal of a failed audit before reading results; detection of a corrupted saved-prediction join; and the watcher's completed-stage, hash, phase, failure, and full-audit gates.

```bash
NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src .venv/bin/python -m pytest \
  -q tests/test_generated_export.py
```

The initial five-test log is `results/generated_exporter_tests.log`; the final reviewed eight-test log is `results/generated_exporter_tests_reviewed.log`. The intermediate log is retained as `results/generated_exporter_tests_final.log`. Synthetic figure and CSV hashes are in `/data2/ishaangp/mira-interp/engineering/generated_exporter_synthetic_v1/synthetic_manifest.json`; all four PNGs were visually inspected. These artifacts use two artificial matches with positive, negative, and constant paths and clipped doses. They contain no real generated outcomes and are not research findings.

To reproduce the synthetic figures and CSVs in a fresh directory:

```bash
NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src:scripts:tests .venv/bin/python - <<'PY'
from pathlib import Path
import json
from test_generated_export import synthetic
from export_generated_measurements import make_tables, render, csv_write

folder = Path('/data2/ishaangp/mira-interp/engineering/generated_exporter_synthetic_reproduction')
folder.mkdir(parents=True, exist_ok=False)
report, protocol, _, _ = synthetic(clipped=True)
tables, curves, matches = make_tables(report, protocol)
figures = render(report, protocol, curves, matches, folder)
csvs = {name + '.csv': csv_write(folder / (name + '.csv'), rows)
        for name, rows in tables.items()}
(folder / 'synthetic_manifest.json').write_text(json.dumps({
    'engineering_only': True, 'synthetic_data': True,
    'real_generated_outcomes_used': False, 'figures': figures, 'csv': csvs,
}, indent=2, allow_nan=False) + '\n')
PY
```

Numerical tables and plot content are reproducible. PDF metadata can change byte hashes on regeneration; the export manifest binds the actual files from each execution.
