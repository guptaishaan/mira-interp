# Confirmation result exports

The exporter reads only the completed v2 confirmation report. It performs no
fitting, reselection, bootstrap resampling, or loading of raw clips/activations.
The confirmation report remains the authoritative numeric result.

Run from the repository root after confirmation has passed:

```bash
NUMPY_MADVISE_HUGEPAGE=0 python scripts/export_probe_results.py
```

This process-local NumPy setting avoids huge-page allocation stalls observed on
this node. It does not change the recorded analysis or its mathematics.

- [`target_metrics.csv`](../results/probes_v2/target_metrics.csv) contains 1,140
  rows: 19 sites × true/shuffled training labels × 30 targets. MAE and RMSE use
  the source units (uu for positions, uu/s for velocities); R² is dimensionless.
  Every confidence interval is copied from the completed report.
- [`site_summary.csv`](../results/probes_v2/site_summary.csv) contains all 19
  sites, normalized MSE and intervals, and paired gains against the mean, codec,
  and corresponding shuffled control. Positive gain means lower main-model error.
- [`observational_layers.png`](../figures/observational_layers.png) and its
  [PDF](../figures/observational_layers.pdf) show all 17 residual sites, true and
  shuffled label curves, whole-match bootstrap intervals, three baselines, and
  the site already chosen on selection.
- [`observational_targets.png`](../figures/observational_targets.png) and its
  [PDF](../figures/observational_targets.pdf) show all 17 × 30 main-model R² values.
  Entity boundaries separate the ball and four canonical player slots. The
  diverging color scale centers on zero, spans the complete negative observed
  range, and extends to 1; negative and positive halves have different scales.
  The original selected row is outlined. CSVs retain exact numeric values.

Both figures describe contemporaneously indexed state-annotation decoding on
11 confirmation matches, with target pixels present. These observations do not
establish future prediction, physical causal coordinates, or steering. Published
frame/state row correspondence is used; independent zero-latency calibration of
every 3D annotation has not been established.

The executed export passed schema and coverage checks. A separate exact-copy
check compared every target metric and interval to the confirmation JSON, checked
all site normalized MSE and paired gains, and verified all six output hashes.
Both PNG layouts were visually inspected. The heatmap has no undefined R² cells
and displays the observed range −0.1899976154 to 0.8711162022 without clipping.

[`export_manifest.json`](../results/probes_v2/export_manifest.json) records the
source confirmation, frozen selection, registration, exporter, and output SHA256
hashes. Confirmation SHA256 is
`4bd750fa2a3ee297a1f64baed79971a2f90136bcbfc9a0d861714c4e8f134cb4`.
The completed export-manifest SHA256 is
`91bf27cdae9bd6b5fb70f96fa1a2495faaa8477952215a20123211657bdadb73`.
