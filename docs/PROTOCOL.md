# Execution protocol

Version 1, prepared before access to any physics-labeled research examples.
This is a study plan, not a claim that the experiments have passed.

## 0. Establish readiness

Pin upstream source, model, codec, data revision, software, and random seeds. Verify
download hashes, strict pretrained state loading, finite outputs, layer coverage,
deterministic replay, hook identity, and intervention locality. Record actual GPU peak
memory and runtime. Use only the two authorized GPUs; recheck occupancy before launch.
Keep a 25 GiB storage reserve. A small random network tests software only.

**Exit:** reproducible model execution plus accessible synchronized video/actions/state.
Model readiness can pass independently, but this does not pass the dataset gate.

## 1a. Build and audit the datasets

The independent unit is a complete match. Assign discovery, selection, and confirmation
by deterministic match-ID hashing before labels or activations are inspected. Keep
all perspectives and overlapping clips of a match in one partition. Also keep a small
engineering pilot set out of every research partition. Respect official test isolation
when constructing the final manifest; do not silently mix model-training exposure with
interpretability-discovery exposure. The reproduction's model-training match manifest
is not available, so unseen-to-the-world-model claims require separate evidence.

Verify timestamps, recording offsets, canonical player IDs, team IDs, coordinate units,
positions, velocities, camera viewpoint, missing entities, and event/replay filtering.
Evaluate action equality on all four original action streams at source sampling rate;
OR-pooled actions at a reduced frame rate can hide differences. Report pair counts,
state differences, matching distances, and coverage before selecting a pilot budget.

**Critical adjustment:** Rocket Science contains recorded trajectories, not an interface
for resetting and rendering the simulator. Similar observed states with equal future
actions are observational matches. They do not establish that only one physical
variable changed. True controlled sweeps need a reproducible simulator state-reset and
renderer, including non-target state and camera checks. If those are unavailable,
observational decoding may proceed under that name; controlled geometry and steering
confirmation cannot be declared complete.

**Exit:** integrity report, hashed split manifests with zero match overlap, sufficient
independent matches and complete labels. A synthetic fixture cannot satisfy this gate.

## 1b. Decode physical state at every layer

Start with ball and canonical-player XYZ positions and XYZ velocities, retaining world
coordinates. Treat viewpoint-relative targets as a separately registered analysis.
Capture the input of residual block 0 and every block output at explicitly recorded
denoising timesteps; preserve frame, view/tile, register, and spatial-token identity.
Record whether inputs are teacher-forced, clean context, noisy targets, or generated.
Future target pixels must not leak into a claim of predictive future-state decoding.

Fit regularized linear probes with feature normalization learned on discovery only.
Choose regularization, readout, timestep, and sites on selection; freeze the choices
before confirmation. Report the full layer profiles, physical-unit error, per-coordinate
R², constant/kinematic baselines, and match-level label-shuffle controls. Aggregate and
bootstrap over matches, not frames. Report all planned targets and multiplicity handling.
Use pilot variance to set sample size and practical-effect thresholds before the main
confirmation run. Poor decoding is an informative negative result, not a reason to fish
in confirmation data. Document every protocol amendment and dataset version.

**Exit:** complete all-layer report, leakage audit, frozen selection artifact, untouched
confirmation evaluation. Decodability alone is not causal use.

## 1c. Validate the trajectory measurement

Train or fine-tune a video-to-state evaluator (VideoMAE ViT-B is the initial candidate)
on independent real clips, using train-only normalization and matched train/validation/
test match splits. Report per-player and ball error, velocity error, occlusion failures,
calibration, and identity swaps. Do not use the intervention-target probe as the sole
measure of the intervention's success. Compare against simple visual/kinematic
baselines before investing in a larger evaluator.

An evaluator trained on real videos may be unreliable on generated videos. Add a
generated-video reliability audit and cross-view consistency checks. If its uncertainty
is comparable to the intended effect, physical causal validation is blocked. Pixel
divergence and a changed internal probe prediction are secondary diagnostics only.

**Exit:** frozen, independently validated physical evaluator with error small enough
to resolve the registered intervention size; otherwise no physical-effect claim.

## 1d. Identify and validate causal sites

Hold all four future actions, the noise tensors, sampler, starting context, and rollout
length fixed within an intervention comparison. Attribute an explicitly signed physical
objective using `(clean - corrupted) dot gradient_at_corrupted`; if the objective is a
loss, beneficial changes have negative sign. Log both the signed effect and convention.
Attribution is a first-order ranking heuristic and must be compared with exact effects.

Use discovery/selection to shortlist sites across layers, denoising times, frames and
spatial regions. Test exact donor replacement, removal relative to a documented
reference, and restoration. Include no-op, norm-matched random, wrong-player,
wrong-variable and wrong-time edits, using paired seeds. Preserve KV-cache semantics;
a post-block edit does not automatically change that block's already computed cache.
State whether an intervention is a one-step impulse or reapplied over denoising/frames.

Confirm only frozen candidates on independent matches and seeds. Report physical
effect, restoration fraction, non-target effects and confidence intervals per match.
Register the primary family and control multiplicity. Failed controls stop this stage.

**Exit:** targeted physical effects, specificity and restoration pass frozen thresholds
on confirmation. Negative results are reported as such; dependent claims stop.

## 2a. Characterize geometry after causal validation

For each controlled source context, sweep velocity magnitude/components or heading
while fixing non-target initial conditions and subsequent actions. Heading is circular;
compare periodic models using a documented zero direction and wrap convention.
Compare affine and prespecified nonlinear predictors with the same training values,
holding out interior values and then complete contexts/matches. Separately test any
claimed extrapolation. Measure prediction and curvature in the original activation
space. PCA plots must give retained variance and cannot prove a manifold.

Distinguish a curved one-variable response from extrinsic curvature of a joint
representation. A flat plane with curved coordinate paths is a required negative
control. Freeze model order, smoothing and layer choices before confirmation.

**Exit:** raw-space held-value/held-match evidence for or against predictable nonlinear
geometry, with controls and uncertainty. An affine result remains a valid result.

## 2b. Recover sparse features

Only now train Top-K SAEs and block-sparse featurizers. Match encoder/decoder parameter
budgets and active scalar dimensions, reporting active blocks separately. Compare
reconstruction/sparsity tradeoff curves; exact simultaneous matching of parameters,
activity and reconstruction may not be feasible. Never describe unmatched models as
matched. Keep training examples, optimizer budgets and seeds comparable.

A temporal extension penalizes changes in block support for explicitly tracked entity
tokens while allowing coordinates within an active block to change. Include a standard
BSF, temporal-shuffle, no-regularization, and no-tracking control to detect mere temporal
smoothness. Do not assume spatial tokens correspond to persistent entities. Measure
reconstruction, activity, held-out geometry preservation and reproduction of the exact
causal effects from step 1d; inspect dead features and seed stability.

**Exit:** measured comparison at documented budgets, with geometry and causal fidelity.
Failure to recover features is reported before attempting feature-based steering.

## 2c. Steer trajectories

Learn reusable linear and geometry-aware edits from discovery/selection only. An oracle
target-activation replacement is a diagnostic, not a deployable steering rule. Compare
equal-norm and equal-requested-physical-dose edits separately. Sweep positive/negative
doses with unchanged all-player actions and paired random seeds. Report monotonicity,
physical calibration, non-target changes, and persistence across frames on held matches
and values. Include impulse versus repeated-edit comparisons and the established
negative controls. Generated physics are measured with the validated evaluator.

Changing ball velocity must change later ball position and can change collision
outcomes. Those are expected causal consequences, not failures of specificity. The
fixed quantities are actions, non-target initial state and unrelated outcomes on a
registered causal horizon. Define that horizon before selecting successful examples.

**Exit:** independently confirmed physical control over the registered range/horizon.
The study can support conditional evidence for causal coordinates; it does not formally
prove general physical correctness or control of every possible trajectory.

## Sources

- [MIRA code](https://github.com/mira-wm/mira) and [Rocket Science](https://huggingface.co/datasets/kyutai/rocket-science)
- [Linear world-model representations](https://arxiv.org/abs/2309.00941)
- [Attribution patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- [Interpreting activation patching](https://arxiv.org/abs/2404.15255)
- [Goodfire geometry discussion](https://www.goodfire.com/research/can-saes-capture-neural-geometry)
- [Counting-task geometry](https://transformer-circuits.pub/2025/linebreaks/index.html)
- [Block-sparse featurizers](https://arxiv.org/abs/2606.25234)

PSI V15 supplied the local conventions for source-disjoint partitions, complete layer
profiles, original-space geometry, protocol hashes and separating readiness from
scientific completion. No PSI results or capture artifacts are reused here.
