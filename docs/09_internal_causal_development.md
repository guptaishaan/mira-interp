# Internal causal development

This experiment tests whether changing a residual activation changes a fixed
readout of the original model's output. It does not measure a rendered physical
trajectory. The primary variable is fixed in advance as **ball height in world
coordinates**. No velocity result can replace a negative primary result.

## Prerequisites and commands

`scripts/causal_development.py` requires completed, independently audited
development probes. Its default independent report is
`results/development_probes_v3/probe_audit.json`, with `status: passed`,
`development_selection_sha256` and `model_archive_sha256` binding the actual saved
selection JSON and coefficient NPZ. `--probe-audit` can specify another path.
All 336 corrected captures must already pass their independent audit. The script
never fits a probe or consumes old/new confirmation matches.

Run each phase only after the previous one succeeds:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/causal_development.py --phase register
# Set CUDA_VISIBLE_DEVICES to the single GPU explicitly allocated for this job.
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/causal_development.py --phase pilot
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/causal_development.py --phase discover
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/causal_development.py --phase evaluate
```

`register` is CPU-only. It writes an immutable protocol, source pair list and
hashes of code, source manifests, source arrays, coefficient archive, prior
selection and independent audits before causal execution. Code/input changes
invalidate that registration. Completed reports are immutable; interrupted
stages can reuse complete per-pair artifacts only after validating their hashes.
The process-local NumPy setting avoids the previously measured hugepage
compaction stalls; host settings stay unchanged.

The reserved pilot uses the same reserved clip on both sides. It verifies all 17
finite gradients and zero donor difference, input/middle/final site patching,
exact repeat/self/restoration equality and the spatial pullback calculation.
At the three pilot sites it also compares the analytic derivative along the
fixed ball-z probe direction with central finite differences at steps1e-4,
1e-3 and1e-2 times the tile's L2 norm. All steps and linearization errors are
reported. BF16 forward quantization can make finite differences disagree with
the surrogate backward derivative; this diagnostic does not filter research
matches or tune a step to obtain a preferred result.
It is an engineering test, not evidence of a physical representation. No research
pair is used to choose thresholds or conditions during this pilot.

## Fixed output objective

The output readout is the saved
`codec_spatial_X/absolute30/position` ridge. Its 4608 inputs are a complete
9×16×32 normalized codec tile, and its 15 outputs are ball/player XYZ positions.
Its coefficients and discovery normalization are loaded without refitting.

Upstream `LatentWorldModel.prepare_training_inputs` defines flow as
`v = z_clean - z_noise` and input as
`z_tau = tau*z_clean + (1-tau)*z_noise`. Therefore the one-step clean-endpoint
estimate is `z_hat = z_tau + (1-tau)*model_prediction`. The model prediction alone
is not a codec state. The fixed objective is

```
J = -((codec_probe(z_hat[view 0, latent7]).ball_z - donor_source_ball_z)
      / discovery_ball_z_std)**2
```

All objectives use FP64 frozen readout coefficients and FP32 endpoint values.
Model parameters remain checkpoint FP32 with BF16 autocast. The output probe was
trained on observed clean codec tokens; applying it to predicted/edited endpoint
tokens changes its input distribution. The report records the clean-input probe
estimate, endpoint estimate and endpoint error to the recipient's observed codec
state. These diagnostics do not independently validate the proxy on edits.

## Discovery ranking and natural pairs

The exact original 31 discovery and 11 selection matches keep their roles. Each
match contributes one deterministic natural pair: lexicographically first clip
as recipient and last as donor among its eight eligible clips. This rule uses
metadata, not labels or favorable intervention results. Player identities must
match within the pair. Both seeds 2026090701 and 2026090702 are used for each pair.
The donor and recipient receive identical noise tensors for a given seed.

The donor can differ in position, velocity, actions and other scene content. It
is not a simulator-controlled one-variable contrast. Every recipient condition
keeps the recipient's video, all actions, clean past, tau and noise fixed and
checks their tensor hashes afterward.

Capture block0 input plus all 16 block outputs. At each site, retain only the
view 0/latent7 9×16 residual tile for attribution. Rank sites by the mean of
`gradient_recipient(J) * (donor - recipient)`, summed over that tile, across the
31 discovery matches and two seeds. Higher scores predict lower donor-target
error; negative scores are retained. Save all 17 scores before fixing the top 3
sites, with canonical site order breaking ties. Evaluate those same sites on all
11 development selection matches without selecting a new site there.

The alpha choices already used development selection data, so even the
subsequent site-ranking separation does not turn these matches into fresh
confirmation. Two seeds are repeated measurements within a match.

## Exact tests and controls

For each chosen site, load that site's current spatial absolute-position probe.
Pull its ball-z derivative back through discovery standardization, the fixed
2048×128 projection and the3×4 spatial-bin means. Each bin contains12 tokens,
so each receives1/12 of the projected bin weight. This produces a raw residual
direction confined to view 0/latent7. The experiment runs:

1. Exact cached recipient replacement, which must reproduce the original output
   bitwise.
2. Exact donor tile replacement.
3. Transfer only the donor-minus-recipient component along the ball-z direction.
4. Remove the centered ball-z component toward its discovery target mean.
5. Restore the exact cached recipient after defining that removal. Equality to
   baseline is an implementation control, not independent sufficiency evidence.
6. An isotropic random direction with the component-transfer norm and support.
7. The same random unit direction scaled separately to full-donor and ablation
   norms, giving each intervention its own paired norm control.
8. The ball-x probe direction with the same signed dose/norm. Report the x/z
   direction cosine; these controls need not be orthogonal.
9. The same ball-z component delta applied to view 1's original tile. This is a
   wrong-view control, not a wrong-entity control: the ball is globally shared.

Full donor replacement can have a different norm from component transfer.
Three random controls separately match component/full/ablation changes, while
x/view controls match component transfer. Realized edit norms are recorded.
No normalization by an unstable
clean-minus-corrupted recovery denominator is used.

Reports include raw standardized-error gain `J_edit-J_baseline`, all 15 position
proxy shifts and final-view estimates, original model flow/endpoint changes,
and endpoint reconstruction error. Confidence intervals resample entire
selection matches after averaging their two seeds. They are descriptive and
conditional on development choices, without multiplicity or population claims.
Paired intervention-minus-matched-control error gains use identical match
resamples and are reported even when negative.

## Saved evidence and limitations

Each discovery pair retains all 17 signed attributions and norms. Selection NPZs
retain selected-site recipient/donor/gradient tiles, probe pullbacks, each exact
replacement tile, full baseline/donor/edited flow predictions, final endpoint
tiles and clean codec/noise inputs. This supports later comparisons between
original residual edits and SAE/block-feature reconstructions at the same site.
JSON reports bind each tensor archive by SHA256. Tensor archives are written
under `/data2/ishaangp/mira-interp/causal_development_v1`; source simulator target
arrays are private and must be excluded from public artifact exports.

The current forward supplies noisy observed target pixels and clean observed
past at tau 0.5 over 16 source frames. There is no full denoising rollout, streaming
cache edit, decoded video, independent physical evaluator, monotonic steering or
causal geometry claim yet. Attribution is a first-order screen; exact patches
can disagree. Natural donor transfer can change collateral properties, and a
probe objective can improve through an off-distribution artifact. All negative
and collateral effects remain part of the result.

An exploratory vertical-velocity extension can be registered after the primary
experiment and an explicit assessment of the codec velocity readout. It is not
silently enabled by primary success, nor substituted for primary failure.

## Reserved pilot execution

The 22-test CPU suite passed before registration. The subsequent reserved GPU6
pilot passed all 17 finite-gradient checks, same-input repeat and exact
self/restoration equality at input, block7 output and block 15 output. It took
30.87 seconds after loading and peaked at 18.54 GiB allocated. Ablation and its
matched random control changed the original model output at each tested site.
The donor equals the recipient in this engineering fixture, so donor-transfer
attributions are exactly zero and are not scientific findings.

At the largest fixed finite-difference step (1% of residual-tile L2 norm),
relative directional-derivative errors were 93.6% at input, 38.7% at block7 output
and 5.17% at block 15 output. Smaller BF16 steps were sometimes sign-inconsistent.
The gradient is connected and finite, but this check does not establish an
accurate local linear approximation everywhere. Preserve all steps, rely on exact
interventions for measured causal effects, and interpret rankings as approximate.

Evidence: `results/causal_development_v1/code_audit.json`, `pilot.json`, the
per-pair pilot JSON and its hash-bound private tensor archive. The pilot report
SHA256 is `1591a0718eedae08a6212c62908b6797fb1c57f8ec9ee56ab2b95178b8b4e3d7`;
the causal registration SHA256 is
`1676ed34ac265e53968151f34b7e67bb9566d0bd6d866876ee41a2d2c25142b6`.
Discovery ranking and selection interventions require separate completed
reports; the pilot alone does not complete those stages.

## Discovery execution and frozen sites

All 31 discovery matches and both paired seeds completed, producing 62 reports
with all 17 signed site scores. The frozen sites for exact testing are:

| Site | Mean attribution | Positive / negative seed pairs |
| --- | ---: | ---: |
| Block 15 output | 0.496878 | 48 / 14 |
| Block 14 output | 0.331224 | 50 / 12 |
| Block 13 output | 0.264613 | 54 / 8 |

The counts describe repeated seed measurements, not 62 independent matches.
Attribution means range from −0.138489 at block 0 input to 0.496878 at block 15
output. These are predicted first-order improvements in the fixed donor-target
objective, not measured exact-intervention effects. The figure
`results/causal_development_v1/discovery_attribution.png` shows all site means and
all 31 per-match seed averages; no unfavorable observation was removed.

`scripts/audit_causal_discovery.py` independently verified every pair/source hash,
own-view final-frame target/time/player mapping, the saved tau 0.5 interpolants,
the clean-endpoint output proxy, all attribution means and the frozen top three.
Maximum output-proxy numerical discrepancy was 2.05e-12; maximum objective
discrepancy was 1.78e-15. It did not rerun backward gradients or recover unpersisted
discovery gradient tiles. Its scope is recorded explicitly in the audit.

The first discovery process received SIGTERM after 32 complete pairs; no parent
or agent intentionally terminated it, and the cause remains unknown. All 32
tensor hashes were verified, the interruption/log retained, and the unchanged
registered protocol resumed under a durable process wrapper. The remaining 30
pairs finished with exit 0. No incomplete pair or alternative protocol entered
the final selection.

Frozen ranking SHA256:
`493d9a885cadc13fae89e623ca06f85863aeec9309525de8c7c8739578a23efb`.
Independent audit SHA256:
`6c51d7a0384495e81ba04e7f5408f3f25f11a24b58433a95e0c9689085e7e4cd`.
Exact tests on the separate development selection matches subsequently completed
as reported below. Neither stage establishes physical control.

## Exact selection results

All 11 selection matches and two paired seeds completed at the three frozen
sites, with 10 conditions per site: **660 condition outputs**. The independent
saved-tensor audit passed every condition, source join, endpoint/proxy metric,
paired match summary and bootstrap interval. All self-replacements and
restorations reproduced the original output bitwise, and every edit left earlier
flow frames bitwise unchanged. Recomputed selected-site attribution dot products
matched exactly; maximum output-proxy discrepancy was 1.60e-12. Independent
ablation readouts reached the discovery mean within 0.000130 raw units.

The following gains reduce standardized squared error toward the donor's source
ball height, as measured by the frozen **output codec proxy**:

| Frozen site | Full donor gain [descriptive 95% interval] | Matches with positive full-donor gain | Height-component gain [interval] |
| --- | --- | --- | --- |
| Block 15 output | 0.68283 [0.03998, 1.64304] | 6/11 | 0.003548 [0.000765, 0.008026] |
| Block 14 output | 0.68127 [0.03943, 1.63912] | 6/11 | 0.002232 [0.000774, 0.004009] |
| Block 13 output | 0.67998 [0.03854, 1.63655] | 6/11 | 0.002835 [0.000677, 0.005531] |

The positive mean full-donor effect is concentrated in a few matches; the largest
match contributes a gain around 4.6. It is not reliable improvement on every
trajectory. Full donor replacement also moves other position readouts: non-target
proxy-shift RMS is about 0.179 discovery standard deviations, and its endpoint
reconstruction error to the recipient's observed codec state increases. The
donor differs in many physical properties and scene details, so this is not an
isolated-variable transfer.

The height component has a small mean effect above its matched random control:
block 15 **0.003476 [0.000657, 0.007955]**, block 14
**0.002431 [0.000662, 0.004850]**, block 13
**0.003202 [0.000688, 0.006528]**. Its non-target proxy-shift RMS is correspondingly
small, 0.000386–0.000668 standard deviations. These comparisons support a narrow
internal readout effect; they do not show that the probe direction carries the
large full-donor effect or controls generated physical height.

Variable specificity is incomplete. The ball-x direction also improves the
ball-height objective; the correct-minus-ball-x difference at block 14 is
0.000337 with interval [−0.000108, 0.000817]. The other-view edit has exactly zero
effect on the view 0 objective after the final block, where the remaining output
head acts locally by token. That structural locality is an implementation
control, not evidence of learned entity specificity. Earlier selected blocks
show small other-view effects. The original view 1 baseline tile was not saved,
so its location/norm audit partly relies on registered hook tests and runtime
measurements; this limitation is explicit in the independent audit.

Removing the centered height component sometimes improves the donor-target
objective. Because the objective targets the donor rather than the recipient,
this does **not** establish necessity of the component for representing the
recipient's true state. Restoration is exact by construction and validates the
implementation; it is not an independent sufficiency result.

Attribution predicts the relative sizes of full-donor effects well within these
11 matches, while remaining imperfect in magnitude and sign:

| Site | Pearson | Spearman | Positive/negative sign agreement | Attribution approximation RMSE |
| --- | ---: | ---: | ---: | ---: |
| Block 15 output | 0.99949 | 0.97273 | 10/11 | 0.17604 |
| Block 14 output | 0.99113 | 0.97273 | 9/11 | 0.29528 |
| Block 13 output | 0.97168 | 0.94545 | 9/11 | 0.47744 |

Each point averages the two seeds within a match. Large observations strongly
influence Pearson correlation; the scatter, Spearman and sign agreement are
reported alongside it. These are descriptive checks at the three shortlisted
sites, not evidence that attribution accurately ranks every untested site.

The six PNG/PDF figure pairs include full-scale effects and controls,
`selection_component_effects_zoom` and `selection_component_controls_zoom`
(10⁻³ standardized units), `selection_attribution_vs_exact`, and
`selection_collateral_proxies`. Every plotted comparison retains all 11 matches.
The full-scale figures remain available so the tiny component effects are not
mistaken for full-donor-sized changes.

Execution took 439.19 seconds after loading and completed with exit 0. Evidence is
in `selection_results.json`, `selection_audit.json`, per-pair JSON reports and
their private, hash-bound tensor archives under the causal output directory.
The independent audit SHA256 is
`69f36fec2aaf396bf0c0276e014c8c9f857757ba417952924e302dee654b54a9`.
Block 15 remains the originally top-ranked discovery site for subsequent feature
development; no new site was selected using these exact-test results.

This completes the registered **development-only internal causal test**.
It does not complete independent generated-video measurement, controlled
single-variable simulator pairs, geometry-preserving feature recovery or
multi-frame physical steering. The next feature experiment must measure how
descriptor reconstruction changes these original-model effects, rather than
treat reconstruction loss alone as causal fidelity.
