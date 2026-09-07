# Revised development readouts

This is development evidence from the original31 discovery and11 selection
matches, after the literature-led repair. No old confirmation examples were read.
The separately frozen 23-match confirmation subsequently passed; see
[the fresh results](11_fresh_confirmation_results.md). The numbers below retain
their original development status.

All336 corrected captures passed independent audits, including the complete17
sites, source labels/timestamps/identities, spatial descriptor algebra and FP16
storage bounds. Each GPU used10.52GiB peak allocated memory. Analysis scores the
last six latent steps, giving8,064 rows with matches as the independent units.

## What improved

Changing the player convention is the strongest development result. The same
visual view follows one car, while the old absolute player0–3 labels depend on
canonical sorted IDs. New role targets name the viewing car (ego) and the ball
relative to that car. Inputs contain no physical labels. Coordinates remain
world-axis XYZ; this does not rotate into camera-heading coordinates.

| Target family | Best current mean | Best current spatial | Best tested temporal | Mean baseline |
| --- | ---: | ---: | ---: | ---: |
| Absolute30, all |0.9117|1.0132|0.9201|1.1055|
| Role12, all |0.4422|0.6491|0.4514|1.0627|
| Role positions |0.2240|0.4744|0.2389|1.1900|
| Role velocities |0.6481|0.7680|0.6381|0.9354|

Numbers are equal-match selection MSE standardized using discovery scales.
Comparisons across different target definitions are not a like-for-like error
reduction. Within role positions, the selected model is about81% below its own
mean baseline; role velocities are about32% below theirs.

The role-position winner is mean block5 output. Ego XYZ R² are0.727/0.884/0.939;
ball-minus-ego XYZ are0.573/0.679/0.922. These positions still have substantial
raw error; high R² does not mean centimeter-level physical accuracy.

Velocity gains are concentrated in the vertical direction. With mean block6
plus a same-view previous-latent difference, ego XYZ velocity R² are
0.159/0.080/0.882 and ball-minus-ego velocity R² are0.002/−0.012/0.540. Horizontal
velocity remains weak and is not described as solved.

## Which hypotheses were not supported

The spatial mean has a proven information bottleneck, but the tested3x4-bin,
128-channel projected descriptor did **not** beat it on these aggregate target
families. This compressed spatial construction is not an exhaustive test of raw
token information. Its negative comparison is retained. Added temporal differences
helped role velocity modestly, while current means won the all-target families.
The stronger full-codec descriptor also remained below the residual readout.

These outcomes favor a representation/target convention explanation for much of
the player-state weakness. They do not prove that pooling caused the original
score, that every state is decoded, or that the represented state is causal.

## Validation and next gate

Every mean/spatial site, seven ridge penalties, both target definitions, three
target families, stronger baselines and whole-match label controls are retained
in `results/development_probes_v3/development_report.json`. Models were fitted
on discovery only. The two-stage temporal comparison reuses development selection,
so an independent confirmation is required. An independent saved-coefficient
audit precedes any causal execution or fresh evaluation.

`development_models.npz` is too large for a Git blob and is published as a
[GitHub release asset](https://github.com/guptaishaan/mira-interp/releases/tag/development-v3-2026-09-07)
with a verified remote SHA256. Reports, plots, code and frozen choices
remain in Git. Source videos and physical-label arrays remain local.
