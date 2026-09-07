# Generated steering: audited selection results

The tested paths did **not demonstrate reliable, ordered ball-height steering**
in this selection cohort. Generation and measurement completed successfully:
420 conditions from 10 matches, two paired seeds per match, all five paths and
all four measurement times. Some individual contrasts are positive, but their
signs, dose ordering, control comparisons, and later responses are inconsistent.
This is a result about the frozen video estimator under the registered edits;
generated physical accuracy has not been established.

The primary outcome is view 0 estimated absolute ball height, computed as
estimated ego height plus estimated ball-minus-ego height. Each condition shares
the same input context, actions and seed with its baseline. One block-15 residual
edit is applied during the first generated latent; subsequent generation proceeds
without further edits. The estimator examines windows ending 0.1–0.4 seconds
after the observed context. See the
[generation protocol](rollout_steering.md) and
[frozen measurement definitions](generated_measurements.md).

## All registered primary contrasts

The contrast is the estimate at requested dose +600 minus the estimate at
requested dose −600, in simulator units. Doses refer to internal probe/map
targets; they are not measured physical displacements. Intervals are the
registered descriptive 95% whole-match bootstrap intervals, after averaging
the two seeds within each match. They do not provide simultaneous coverage over
paths/times or include evaluator error.

“Positive” counts positive two-seed match means out of 10. “Ordered” counts
nondecreasing five-dose responses with nonzero span out of 20 match/seed pairs;
it is a descriptive pair count, not 20 independent matches. Ordering uses actual
clipped doses and rejects inconsistent responses at tied doses.

| Path | Time (s) | Extreme contrast [95% interval] | Positive matches | Ordered pairs |
| --- | ---: | ---: | ---: | ---: |
| Linear probe | 0.1 | +0.958 [+0.225, +1.670] | 8/10 | 2/20 |
| Linear probe | 0.2 | +0.582 [-0.333, +1.597] | 7/10 | 1/20 |
| Linear probe | 0.3 | +0.278 [-1.024, +1.566] | 6/10 | 1/20 |
| Linear probe | 0.4 | -1.293 [-2.819, +0.129] | 3/10 | 0/20 |
| Affine map | 0.1 | -0.075 [-0.611, +0.490] | 3/10 | 0/20 |
| Affine map | 0.2 | -0.980 [-2.260, +0.021] | 4/10 | 1/20 |
| Affine map | 0.3 | +1.319 [+0.282, +2.295] | 8/10 | 1/20 |
| Affine map | 0.4 | -0.359 [-1.337, +0.545] | 5/10 | 1/20 |
| Quadratic map | 0.1 | +0.163 [-1.199, +1.680] | 5/10 | 2/20 |
| Quadratic map | 0.2 | -0.019 [-0.611, +0.630] | 4/10 | 2/20 |
| Quadratic map | 0.3 | -0.172 [-0.771, +0.438] | 5/10 | 0/20 |
| Quadratic map | 0.4 | -0.730 [-2.212, +0.796] | 4/10 | 1/20 |
| Random control | 0.1 | +0.413 [-0.456, +1.243] | 8/10 | 1/20 |
| Random control | 0.2 | -0.006 [-0.859, +0.736] | 6/10 | 0/20 |
| Random control | 0.3 | +0.156 [-1.215, +1.173] | 7/10 | 1/20 |
| Random control | 0.4 | -0.110 [-1.028, +0.816] | 5/10 | 2/20 |
| Wrong-ball-x control | 0.1 | -0.884 [-1.808, +0.018] | 4/10 | 0/20 |
| Wrong-ball-x control | 0.2 | +0.087 [-0.877, +1.139] | 5/10 | 3/20 |
| Wrong-ball-x control | 0.3 | +0.208 [-0.678, +1.152] | 4/10 | 0/20 |
| Wrong-ball-x control | 0.4 | -0.327 [-1.360, +0.775] | 3/10 | 0/20 |

The linear probe's early contrast is +0.958 [0.225, 1.670], positive in 8/10
matches. It falls to −1.293 [−2.819, 0.129] at 0.4 seconds, positive in only 3/10.
Its full dose ordering occurs in only 2/20 pairs initially and 0/20 at the last
time. A positive pooled endpoint contrast therefore does not demonstrate reliable
individual dose ordering or persistence.

The affine path has one positive interval at 0.3 seconds:
+1.319 [0.282, 2.295], with 8/10 positive matches. Its mean contrasts at the other
three times are negative. Quadratic contrasts are near zero or negative, with
every interval including zero. No candidate path has consistently positive,
ordered responses across the four times. The registered comparisons are
retained without selecting a new path or endpoint from these results.

## Controls and corroborating measurements

All 12 candidate-path/time contrast differences against the random control have
intervals including zero. For the linear probe at 0.1 seconds, the paired
difference from random is +0.546 [−0.627, 1.895]. Random itself has a +0.413
[−0.456, 1.243] contrast and 8/10 positive matches at that time. The linear
probe does exceed the wrong-ball-x control at this one time:
+1.842 [0.378, 3.297]. This is the only candidate-versus-wrong-x interval excluding
zero; it does not establish specificity against both controls or across time.
The affine 0.3-second difference is +1.163 [−0.505, 2.680] versus random and
+1.110 [−0.228, 2.407] versus wrong x.

The controls match the quadratic path's requested raw lift norm. Under the
registered mixed-precision inference, random/quadratic norm ratios span
0.9956–1.0052 and wrong-x/quadratic
ratios span 0.9962–1.0046 across the 80 nonzero conditions per control. They do
not match the linear or affine path's norms. The random direction is isotropic
in the 1536-coordinate descriptor; covariance- or feature-subspace-matched
controls were not included. Thus norm matching alone does not control every
geometric difference between directions.

The secondary all-view mean does not corroborate the two positive primary
intervals: the linear 0.1-second contrast is −0.088 [−0.424, 0.271], and the affine
0.3-second contrast is −0.016 [−0.464, 0.432]. At 0.4 seconds, the linear all-view
contrast is −0.830 [−1.485, −0.185]. These remain secondary learned measurements;
view differences are not simulator-ground-truth evidence.

Mean standardized nuisance RMS for the three candidate paths ranges from
0.00461 to 0.00832 across endpoints. This averages 11 role coordinates using
frozen discovery scales and excludes only relative ball height; ego height is
still included. Small predicted nuisance changes cannot establish unchanged
physics, particularly for coordinates the evaluator predicts poorly. Physical
coupling can also make a real off-target change appropriate rather than an error.

## Dose clipping and repeated responses

None of the 20 baseline probe-height estimates required clipping; they span
336.975–739.366. Negative target doses frequently reached the discovery lower
support boundary. All 10 matches have at least one clipped seed/dose condition.
Counts below are across 20 match/seed pairs, shared by all five paths.

| Requested dose | Mean effective dose | Effective range | Clipped pairs |
| ---: | ---: | ---: | ---: |
| −600 | −385.192 | [−600.000, −250.596] | 19/20 |
| −300 | −293.013 | [−300.000, −250.596] | 6/20 |
| 0 | 0 | [0, 0] | 0/20 |
| +300 | +300 | [+300, +300] | 0/20 |
| +600 | +600 | [+600, +600] | 0/20 |

The extreme effective-dose span is 850.596–1200 units. Six pairs across four
matches have four distinct effective doses because the two negative assignments
coincide; the remaining 14 pairs have five. There are no inconsistent tied-dose
responses. One wrong-x pair at 0.2 seconds has an undefined rank correlation;
the other seed keeps that match eligible, so all 10 matches remain in every
reported rank average. The missing pair is retained in the counts.

Clipping prevents interpreting the requested-dose contrast as a common symmetric
±600 intervention. It does not remove the observed ordering failure: every pair
still has four or five distinct effective doses. Effective-dose plots retain
original assignment groups, whose ranges differ across pairs; they are not fits
at one shared realized dose.

## What changed, and what remains unproven

The generation audit verifies identical actions and pre-edit sampler inputs,
unchanged context latents, and exact decoded context pixels across conditions.
All 400 nonzero edit conditions have nonzero saved latent differences at all
four generated times. There is therefore a downstream internal effect of the
edits; absence of reliable measured steering is not absence of any model change.
Latent differences do not identify a physical variable or prove coherent video.

The [posthoc edit-delivery check](../../results/generated_evaluation_v2/selection_delivery/summary.json)
also shows that all 80 nonzero linear-probe conditions delivered 97.52%–99.81%
of their effective target change to the internal ball-z probe, with median
99.17%. Relative descriptor mismatch is 1.79%–5.38%; requested-versus-realized
descriptor direction cosine is at least 0.9988. These metadata do not support a
missing hook or an edit erased by quantization as the explanation for the weak
downstream ordering. This does not exclude downstream precision sensitivity or
other numerical effects. The same internal probe constructs and measures this edit,
so this is an engineering delivery check, not independent physical validation.
All 400 per-condition rows and every grouped delivery statistic were independently
recomputed from the already audited metadata; maximum formula discrepancy was
below 3.4×10⁻¹⁶. No new model execution or hypothesis selection was involved.

The following limits remain material:

- Reference absolute ball-height MAEs are 263.713 units on recorded video and
  264.858 on codec reconstructions. These development errors are **not paired
  effect thresholds**: shared errors can cancel, while edit-dependent errors can
  create bias. A small deterministic paired change is not automatically
  meaningless or accurate. Generated-domain calibration is still missing.
- Every 16-frame evaluator window contains the first generated frames 16–17.
  A lasting estimate can reflect these shared early pixels, even if later
  physical changes fade. Exact context-pixel controls remove one confound but do
  not remove this dependence. The evaluated horizon is only 0.4 seconds.
- Views, times, doses and paired seeds are repeated measurements. The independent
  unit is the match. Intervals omit evaluator bias, broader model randomness and
  multiple-comparison correction. No generated-ground-truth R² can be claimed;
  weak source-domain R² for a nuisance coordinate limits conclusions about its
  preservation.
- Geometry fits did not establish a nonlinear physical manifold. The earlier
  sparse conditional-fidelity result retained most full-donor gain even with the
  descriptor replaced by its discovery mean. Its retained native complement
  prevents treating dictionary reconstruction as feature sufficiency. See the
  [mean-control result](sparse_causal_fidelity.md).
- These are selection results. They must not select a new path, dose, site,
  evaluator or reporting endpoint and then relabel the current confirmation
  cohort as untouched confirmation for that new choice. All registered
  confirmation conditions remain unchanged.

## Evidence and review scope

The generation and measurement audits passed before this interpretation was
written. This review independently checked their hash bindings and recomputed
the reported equal-match means from saved per-pair results. It tabulates existing
audited intervals and metrics; it performs no new fitting or model execution.
Clipping, realized control-norm ratios and nonzero latent-change counts are
descriptive checks of the audited generation metadata.

- [Selection measurement report](../../results/generated_evaluation_v2/selection.json):
  SHA256 `55b483c3a240629ea53cf226e91e36597d16713229ef6748fc59e0735ab4c5b0`.
- [Independent measurement audit](../../results/generated_evaluation_v2/selection_audit.json):
  SHA256 `0f11d0e92a119b9db059148426401b405f0021b566442fb74d3112778c8fd48f`.
- [Generation manifest](../../results/rollout_steering_v2/selection.json):
  SHA256 `02ebf78b9c3c54cf38a84a4fa58c79b2ff56201af55d4e5d55f8f96eb5285d84`.
- [Independent generation audit](../../results/rollout_steering_v2/selection_audit.json):
  SHA256 `503437ff00d0038b1e17cc67fa4fa58c262d2737d8bb07a4bfd819e9eb12e28f`.
- [Complete exported figures and tables](../../results/generated_evaluation_v2/selection_figures/export.json).

The supported conclusion is that this fixed selection experiment did not
demonstrate reliable ordered steering under the learned estimator. It does not
establish that the model lacks physical representations or that every possible
intervention would fail.
