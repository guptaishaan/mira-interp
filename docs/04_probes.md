# Observational all-layer physical-state probes

**The registered observational execution is complete.** Data, capture, selection,
independent selection audit and confirmation all passed their execution gates.
See [actual results and limitations](07_observational_results.md). The mathematical
software tests use synthetic arrays; the reported research results use real data.

The registered analysis measures **decoding of contemporaneously indexed
physical-state annotations from observed videos at diffusion time tau=0.5**. The model sees noisy latent encodings of
the target frames and clean previous-frame latents. Good decoding would demonstrate
accessible state information in this setting. It would not establish future-state
prediction, causal use, a physical manifold, or successful steering.

The active registration is [`observational_probe_v2.json`](../configs/observational_probe_v2.json),
SHA256 `e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2`,
registered at 2026-09-07 00:05:44 UTC, before research capture or fitting. It follows
the original [`v1 registration`](../configs/observational_probe_v1.json), SHA256
`2437f915ceb14a10c33a6b003b7273ee6ba135e8495d4b15b03ff72f0edbd6a3`, which
**failed its original 61-match data gate**. Both exact protocol hashes are accepted
for reproducibility, but the input capture audit must match the actually selected
protocol. Altered hashes and cross-protocol audit substitution are rejected.
Bootstrap details below complete the numerical implementation before
any research fit: 500 whole-match resamples, seed 20260906, percentile 95% intervals.

## Input and prerequisites

An aggregated NPZ must contain:

| Array | Shape | Meaning |
| --- | --- | --- |
| `X` | `[N,17,2048]` | Within-view spatial means at block-0 input and all 16 block outputs |
| `y` | `[N,30]` | Ball and four canonical players: XYZ location and XYZ velocity |
| `match_ids`, `split` | `[N]` | Complete-match identity and discovery/selection/confirmation assignment |
| `sites` | `[17]` | Exact site names and order from the registration |
| `target_names` | `[30]` | Exact registered target names and order |
| `codec_X` | `[N,32]` | Within-view codec spatial mean, on the same rows |
| `RGB_X` | `[N,192]`, optional | Within-view RGB descriptor, 3 channels x 8 x 8 cells |

Eight 16-frame clips per match give 256 rows: eight clips x four views x eight
latent pairs. Rows retain clip order, then view-major order, then latent time.
The target for latent pair `t` is that view's own source frame `2*t+1`, because
view timestamps are slightly different. The four views and their frames are
repeated observations of a match, not independent samples. The same ridge readout
is applied to all views; separate probes are not fitted to each viewpoint.

The v2 cohort has 31 discovery, 11 selection, and 11 confirmation matches, with
the engineering pilot excluded. It contains all 53 original-role matches that
passed unchanged quality gates, with all eight prescribed clips each. Seven
matches failed clock consistency and one had fewer than three observed goals;
the latter has insufficient calibration evidence, not demonstrated corruption.
The original roles were retained with no replacements or rebalancing. The original
37/12/12 plan and failed audit remain preserved in the v2 lineage.

The minimum remains 30/10/10. Fewer matches produce
`blocked_underpowered` before fitting. Counts above the minimum must still agree
with the exact registered cohort. All matches must remain within one partition;
every expected site and target must be present and finite.

The analyzer requires a separate JSON audit tied to the input NPZ's SHA256:

```json
{
  "status": "passed",
  "analysis_npz_sha256": "SHA256_OF_THE_ACTUAL_NPZ",
  "registration_sha256": "e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2",
  "real_data": true,
  "video_alignment_checked": true,
  "physics_alignment_checked": true,
  "all_layers_complete": true,
  "match_splits_disjoint": true,
  "checkpoint_integrity_checked": true
}
```

These flags must summarize completed upstream audits, including clip/source hashes,
timing, player identities, source actions, pretrained loading and capture controls.
They must not be filled merely to make the analyzer run. Their evidence remains in
the underlying data/capture reports.

## Fitting and selection

1. Assign each match total weight `1 / number_of_matches`; divide that weight
   equally over its rows. Consequently all row weights sum to one.
2. Compute feature and target means/scales on **discovery only**, using these
   weights. Constant or numerically constant columns use scale one. Selection and
   confirmation never change the normalization statistics.
3. Fit linear ridge with an unpenalized intercept. With standardized features `Z`
   and standardized targets `T`, minimize
   `sum_i w_i ||T_i - Z_i B||² + alpha ||B||²`.
   Use exactly `[0.1, 1, 10, 100, 1000]`. Each site's covariance is decomposed once
   and reused for all alphas and the label-shuffled control.
4. Choose one alpha per site by equal-match selection MSE, averaged over the 30
   discovery-standardized targets. Then choose the global residual site using
   that same selection score. Ties use smaller alpha, then registered site order.
   Codec and optional RGB baselines use the same cohort, weights and alpha grid.
5. Save coefficients, normalization arrays, selected alphas, the global winner,
   complete selection curves, label-shuffle mapping and content hashes. Write the
   selection lock **before** evaluating confirmation. Never refit on selection.
6. Independently audit the saved coefficients and selected losses before evaluating
   confirmation. The audit recomputes discovery normalization and checks the ridge
   normal equations, selected alpha minima and the global winner. It decodes only
   discovery/selection target rows; it does not refit models. Unselected-alpha
   curves are checked for completeness and choice consistency, not independently
   refitted.
7. Reload the locked coefficients from disk and verify their fingerprints. Evaluate
   every registered residual site, preserving the selection-chosen global winner.
   No confirmation winner is selected.

The selection lock also hashes the analysis script and probe implementation.
Confirmation requires a passed independent selection audit tied to every frozen
artifact. It rejects changed code, data, audit, registration or coefficient files;
a code repair after selection must be documented rather than silently changing
the measurement procedure.

Feature dimensionality differs between residual, codec and RGB readouts. They are
comparative information benchmarks on the same examples, not capacity-matched
representations. Residual and shuffled-label models have matched dimensionality
and tuning budgets.

## Controls and uncertainty

The mean baseline predicts the equal-match discovery target mean in original
physical units. The codec baseline tests whether a linear readout can recover the
same targets directly from the input representation. The optional RGB baseline
provides a simple visual comparison.

The label-shuffled control permutes complete discovery-match label blocks, with a
fixed random ordering and a cyclic shift so that no match receives its own labels.
It preserves label marginals and the within-match clip/view/time ordering. Equal
row counts per discovery match are required; there is no label interpolation or
resampling. Its alpha is independently selected from the same grid using the real
selection targets. One registered shuffle is a negative control, not a randomization
test or a calibrated p-value.

Report per-target MAE, RMSE and R² in the original Unreal-unit coordinate system,
plus mean standardized MSE for comparison across targets. Metrics give each match
equal weight. R² uses the pooled target variance under those same match weights;
undefined R² for constant confirmation targets is saved as JSON `null`.

Confidence intervals resample **whole confirmation matches** with replacement.
They preserve all repeated rows and use identical bootstrap draws across models.
Paired whole-match intervals also report standardized-MSE improvements against
the discovery mean, codec readout and that site's shuffled-label control. Positive
gain means lower error than the stated comparator; no confirmation-based choice
is made from these comparisons.
Only 11 confirmation matches are available under v2, so intervals are descriptive and may
be unstable. They do not include uncertainty from retraining, alternate noise
seeds, or a larger population. Per-site/target intervals are not corrected for
all-site multiplicity, and the study makes no family-wide significance claim.

## Run only after the gates pass

Selection, its audit, and confirmation must run as separate processes:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/analyze_probes.py \
  --registration configs/observational_probe_v2.json \
  --data /path/to/audited_observations.npz \
  --audit /path/to/aggregate_capture_audit.json \
  --output-dir results/observational_probes --phase select

NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/audit_probe_selection.py \
  --registration configs/observational_probe_v2.json \
  --data /path/to/audited_observations.npz \
  --audit /path/to/aggregate_capture_audit.json \
  --selection-dir results/observational_probes

NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/analyze_probes.py \
  --registration configs/observational_probe_v2.json \
  --data /path/to/audited_observations.npz \
  --audit /path/to/aggregate_capture_audit.json \
  --output-dir results/observational_probes --phase confirm
```

The default phase is `select`; there is no automatic combined phase. A missing,
failed, mismatched or changed selection audit blocks confirmation.
Existing locks/results are not overwritten.
CPU BLAS thread limits are set to eight; no GPU is used by this analysis.

The first selection attempt spent most CPU time in host-kernel memory compaction.
It was interrupted before any model archive or selection lock was written; its
record and log are preserved in [`results/probe_attempts`](../results/probe_attempts/).
The restart uses the identical analysis code and protocol, with
`NUMPY_MADVISE_HUGEPAGE=0` disabling NumPy's huge-page requests only for that
process. A shape-matched synthetic timing check reduced the complete matrix
calculation to about 1.3 seconds; these timings are engineering diagnostics,
not research results. Mathematical operations, hyperparameters, split roles and
selection rules are unchanged. No host-wide huge-page settings were changed.

Expected outputs after an actual completed run:

- `frozen_selection.json`: all selection curves, per-site choices, global winner,
  and data/audit/registration/model hashes.
- `frozen_models.npz`: reloadable coefficients and discovery normalization arrays.
- `selection_audit.json`: independently checked selected models, losses, choices,
  and artifact/code hashes; required before confirmation.
- `confirmation.json`: all 17 residual profiles, 30 targets, matched controls,
  baselines, descriptive match-bootstrap intervals, and explicit claim limits.
- `layer_profile.png`: full confirmation profile with the selection-chosen site
  marked; target-pixel access is stated in the title.
- `status.json`: execution status. `passed_observational_analysis` means the
  registered analysis finished; it does not mean the hypothesis passed.

## Software verification and remaining scientific gates

Twelve tests pass in [`test_probes.py`](../tests/test_probes.py):
primal and dual ridge agree with an independent direct linear-system solution;
constant columns remain well behaved; duplicating correlated rows in one match
does not change the weighted fit; arbitrary confirmation changes cannot alter
normalization, selected models, curves or hashes; frozen coefficients round-trip
with fingerprint checks; whole-match shuffles preserve their contract; metrics
retain physical units and bootstrap match identity; split leakage is rejected.
The v1/v2 gate tests reject a mismatched capture protocol, changed data hash, and
unapproved protocol, while accepting each explicitly approved matching pair.
Additional tests verify excluded confirmation labels are skipped by the audit
reader, wrong discovery coefficients/normalization are rejected, and confirmation
cannot proceed with a missing, stale or incomplete independent selection audit.
All test inputs are synthetic and temporary. No synthetic research curves are
written as results.

The reproduction's model-training match manifest is unavailable. These splits
establish separation for this interpretability analysis; they do not prove the
matches were unseen during world-model training. Negative observational results
are valid outcomes. Positive observational decoding still cannot open the causal
stage without controlled-pair audits and an independent generated-video physical
evaluator. Geometry, sparse features and steering remain gated.

The v2 population is restricted to quality-qualified matches and eligible live
windows, with 8/61 (13.1%) original research matches excluded. Requiring at least
three goals and consistent clock metadata can favor particular matches/clients;
frozen-state and demolition exclusions further change the sampled population.
31/11/11 meets operational minima but is not a prospective power calculation.
Players can recur across matches, so this is not a player-disjoint study.

Own-view frame/state indexing follows the publisher's one-to-one export contract,
supported by count, PTS and identity audits plus a limited pilot HUD spotcheck.
It does not independently establish zero video/state latency or precise 3D camera
alignment. The report therefore describes contemporaneously indexed annotation
decoding and carries the original failed-plan lineage and all v2 population limits.
