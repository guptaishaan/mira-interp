# Geometry, sparse features, and the causal experiment gates

Reviewed primary sources and current repository code on 2026-09-07 UTC. This is a
methodological review and a proposed development plan, not a new experiment.
The completed v2 data, captures, selection, and confirmation results are unchanged.

**The current pooled linear-probe result is a valid, limited baseline. Its weak
velocity scores do not establish that MIRA lacks motion representations. The
protocol unnecessarily blocks exploratory geometry, sparse dictionaries, and
model-level activation interventions behind requirements needed for stronger
physical counterfactual claims.** Fix the measurement and scope of each claim
before interpreting another score or requiring a simulator.

## What the linked methods actually do

| Primary source | Method relevant to this project |
| --- | --- |
| [Goodfire: Can SAEs Capture Neural Geometry?](https://www.goodfire.com/research/can-saes-capture-neural-geometry) | A concept can be distributed across multiple SAE directions. The article distinguishes compact capture, shattering, and dilution, and groups features using dependencies in their firing patterns. Inspecting isolated high-activation examples is insufficient to reconstruct the whole geometry. |
| [Bhalla et al., full paper, v1](https://arxiv.org/html/2604.28119v1) | The discovery pipeline binarizes feature activity, estimates an Ising coupling model using pseudolikelihood, clusters the absolute couplings with Louvain, and checks candidate groups' PCA spectra. Negative dependencies matter because features tiling different parts of one concept may exclude each other. Decoder cosine or positive coactivation alone is therefore an incomplete grouping rule. |
| [Anthropic: When Models Manipulate Manifolds](https://transformer-circuits.pub/2025/linebreaks/index.html) | Sparse features helped identify distinct counting variables after initial scalar probes proved misleading. The authors average activations conditional on character count, obtain a six-dimensional PCA subspace, test subspace ablation against random subspaces, and intervene by subtracting the original count mean and adding a target-count mean. They test model behavior, rather than treating PCA as the causal result. Their 150-way logistic probes also expose structured responses that scalar regression misses. |
| [Fel et al., BSF paper, v1](https://arxiv.org/html/2606.25234v1) | Signed coordinates are grouped into blocks. Vanilla BSF retains the largest block norms; Grassmannian BSF ties the encoder to orthonormal decoder blocks; the paper's Group Lasso variant uses group soft shrinkage. The assumed concept has a small linear span, a stronger condition than small intrinsic dimension. The paper compares minimum description length, including support, code, residual, and dictionary costs. Its diffusion demonstration uses observed block coordinates and Kohonen-map waypoints; it does not establish calibrated physical coordinates for MIRA. |

The Anthropic page exceeds the browser extractor's size limit. Its complete
primary HTML was retrieved directly and the text of the methods, interventions,
and appendices was read; this review is not based only on a search snippet.

### Pin the implementation, not only the paper title

The [official BSF repository](https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9)
was inspected at commit `219f121ea82d2b19200d1dac918396e6058d7eb9`.
There are material paper/code differences:

- [`group_lasso.py`](https://github.com/goodfire-ai/block-sparse-featurizer/blob/219f121ea82d2b19200d1dac918396e6058d7eb9/bsf/group_lasso.py)
  uses a hard block JumpReLU gate, learned thresholds, and a target-L0 dual update;
  it does not implement the paper's soft-shrinkage formulation.
- [`grassmannian.py`](https://github.com/goodfire-ai/block-sparse-featurizer/blob/219f121ea82d2b19200d1dac918396e6058d7eb9/bsf/grassmannian.py)
  uses one unconstrained gain per block and QR during decoder evaluation. The
  paper describes one shared positive gain and periodic reprojection.
- [`vanilla.py`](https://github.com/goodfire-ai/block-sparse-featurizer/blob/219f121ea82d2b19200d1dac918396e6058d7eb9/bsf/vanilla.py)
  with block size one is signed absolute-TopK. This is not the same baseline as
  conventional nonnegative ReLU Top-K.

Start with pinned Vanilla BSF and explicitly named scalar baselines. If adopting
Grassmannian BSF, record the code variant. Do not label an implementation a
paper replication while silently combining these variants.

## What the current representation measurement loses

These findings follow from local code inspection, not from another confirmation fit.

| Current choice | Consequence | Concrete development correction |
| --- | --- | --- |
| [`pool_views`](../../scripts/capture_observations.py) averages the full 9×16 spatial grid independently for each view/time. | Every spatial component summing to zero is annihilated. Entity location, motion across patches, and opposing local changes can disappear. The retained mean cannot reconstruct them afterward. | Capture native spatial tokens with `(match, clip, view, latent_time, row, col, layer)` identity. Compare mean pooling with a fixed spatial-grid readout under matched feature/probe budgets. |
| Each view's mean predicts the same four canonical-player slots; view index is stored but not supplied as a probe condition. | A camera view is not an entity token. The local player changes with the view, and a patch may contain the ball, another car, scenery, or several objects. Correct source IDs do not solve this representational correspondence problem. | Compare shared and view-conditioned readouts; derive a documented viewer/local-player mapping. Report self, teammate, and opponents separately where identity permits. Keep world coordinates as one target family. |
| Only world XYZ position/velocity are decoded with one affine probe per site. | A model may use relative displacement, speed, direction, time-to-contact, or a nonlinear code. Failure on one coordinate system is not failure on all of these. | Prespecify a small target family: world velocity, relative velocity to the local player, horizontal speed, and direction represented by cosine/sine. Compare with simple nonlinear or binned probes on development matches. |
| A single selection objective averages all 30 standardized coordinates. | Its winning site/alpha is a compromise. It need not be the best velocity representation or a natural entity decomposition. | Retain the original winner and report as historical. A new development study may choose per-family candidates, with its own selection budget and later independent test. |
| One noisy observed-target setting, τ=0.5, is captured. Labels use each two-frame latent's final source frame. | This measures observed-target denoising, not future prediction. Temporal mixing and codec compression can affect instantaneous velocity readout. | Compare a small prespecified set of denoising times; separately implement context-only future prediction. Audit temporal receptive fields, action intervals, and label alignment before naming either result. |
| Codec baseline is a 32D spatial mean, RGB is 192D, and residual features are 2048D. | A weak baseline can reflect readout loss or capacity differences. | Retain codec grids and compare equal-dimensional spatial readouts, fixed random projections, and equal regularization-selection budgets. |

Do not copy the DINO-specific positional-mean subtraction from the BSF examples
without a MIRA ablation: absolute spatial position is itself relevant here.
Keep token position as metadata and fit any centering on discovery only. For
geometry, per-channel whitening changes distances and angles; report raw-space
results and state the metric whenever normalized coordinates are used.

The current four views are repeated observations of one match, not four independent
samples. Their state rows retain separate timestamps. Exact zero-latency 3D
alignment remains unproven; heading and velocity analysis is especially sensitive
to this limitation. Do not average misaligned view trajectories into a single curve.

## A practical velocity/heading geometry pilot

The following are new proposed tests, not claims that the linked papers established
the same geometry in MIRA.

1. **Fix the semantic target.** Horizontal velocity heading is
   `(vx, vy) / hypot(vx, vy)`, with an explicit minimum-speed mask. It is undefined
   when effectively stationary and is distinct from vehicle orientation.
   Report coverage and angular error with wraparound. Define a speed threshold
   from discovery label uncertainty; the helper's default 1 uu/s is an engineering
   default, not a validated measurement threshold. Separate vertical velocity.
2. **Test recoverability before naming the shape.** Compare affine decoding,
   coarse speed/direction classification, and a small regularized nonlinear
   decoder using exactly the same match split and sample budget. Direction bins
   alone do not test interpolation to unseen angles; use held-out angular arcs
   and periodic regression for that question. Use lagged labels only in a
   separately named alignment diagnostic, never optimize lag on confirmation.
3. **Study conditional activation geometry.** Fit forward maps from physical
   values to activations as well as inverse probes. Compare affine maps with
   low-order periodic bases for heading and regularized splines for speed. Hold
   out interior physical values and whole matches; separate extrapolation.
   Account for camera/view, position, speed when testing heading, and heading
   when testing speed. Record support gaps and nuisance imbalance. These remain
   conditional observational relationships without isolated input manipulation.
4. **Keep the geometric test in its original space.** Use discovery-only PCA to
   visualize conditional means and held-out points with explained variance.
   Evaluate held-out activation prediction, distance distortion, local rank,
   tangent consistency, and sensitivity to context. A circle in a two-dimensional
   velocity plane is a required control: its heading sweep is curved even though
   the joint velocity space is flat. A low-dimensional projection or nonlinear
   decoder improvement alone does not establish topology or physical causality.
5. **Add a behavioral intervention.** On development contexts, edit a candidate
   subspace or block with paired model inputs/noise and measure a specified
   model-output change. This can test model dependence before a physical evaluator
   exists. Quantitative claims about 3D generated motion wait for a validated
   measurement, not for a particular evaluator architecture by name.

Suggested engineering budget: deterministically choose eight original discovery
matches and two existing clips per match, using six matches for fitting and two
for pilot validation. Capturing all 17 sites without spatial pooling yields
73,728 token observations and about 4.78 GiB of float16 residual arrays before
metadata. Check disk and actual runtime first. This is a development pilot with
eight independent matches, not a statistically adequate confirmation study.
No selection/confirmation data need be recaptured or re-fitted to start it.

## Fair sparse-feature comparisons

Treat a sparse dictionary as a way to discover candidates, not as evidence of
causality or guaranteed entity isolation. Use native token activations, and keep
each token's location for activation maps. Training tokens from the same match
remain correlated; validation and bootstrap units must remain matches.

For input dimension `d`, total latent scalar width `M = G*b`, and `k` active
blocks, record both block activity `k` and maximum active scalar count `k*b`.
An untied Vanilla BSF has `2*d*M + M` encoder/decoder parameters before any
shared preprocessing. A similarly biased untied scalar baseline can match this
exactly at width `M`. Tied Grassmannian models have a different parameter count;
same width is not same capacity. Count actual trainable parameters in the report.

A simple initial, explicitly proposed comparison uses `d=2048`, `M=4096`,
block sizes `{1,2,4,8}`, and active scalar budgets `{16,32}`. Thus `G=M/b` and
`k=budget/b`. Include both conventional nonnegative Top-K and signed scalar
absolute-TopK. Comparing signed BSF only with nonnegative SAE would confound
block structure with sign freedom. This grid is a feasibility choice, not a
claim that MIRA concepts have dimension two or four.

Use the same examples, train-only centering/global scale, optimizer steps,
initialization-seed set, and selection allowance. Report held-out reconstruction,
dead-feature rates, actual activity, parameter count, time, and downstream
denoising degradation. Compare tradeoff curves and common reconstruction-error
ranges; exact simultaneous equality of capacity, activity, and reconstruction
cannot generally be forced. Minimum-description-length estimates may complement
this, but must specify quantization, support coding, sample count, and dictionary
amortization. They are not interchangeable with reconstruction R².

For feature recovery, report geometry retained by the decoded reconstruction,
held-out target accessibility in selected blocks **and in the complement**, and
seed stability up to block permutation/internal basis rotation. A decoder norm or
block norm is not the direction coordinate. Low stable rank is not a proof of
intrinsic manifold dimension. A selected block may still contain nuisance factors.

Post-hoc grouping of SAE features is a worthwhile comparison to BSF. Estimate
conditional firing dependencies on a manageable discovery-only feature subset,
include inhibition, and validate group geometry on other matches. If a simpler
coactivation heuristic replaces the full published estimator, name that adaptation.

Do not begin the temporal extension by forcing identical support at the same
image-grid location across frames. The object moves while the grid does not.
Use audited tracks or explicitly localized entity masks; state if localization is
oracle-assisted. Penalize support changes only while the entity is visible and
identity is reliable, allowing within-block coordinates to change. Compare with
ordinary BSF, shuffled temporal links, incorrect tracks, and equal-strength generic
smoothing. Otherwise static scenery or forced persistence can appear successful.

## Correct the causal gates without weakening the claims

The current [protocol](../PROTOCOL.md) places geometry after causal validation and
sparse training only after geometry. The [causal prerequisite report](../06_causal_prerequisites.md)
does allow a narrower observational intervention design, but later-stage status
language effectively treats the strict pair audit as a universal blocker.
**That ordering is too restrictive for discovery.** A prospective amendment should
distinguish the following tests; the frozen v2 protocol itself must stay unchanged.

| Test | Required before running | Claim supported by a successful result |
| --- | --- | --- |
| Conditional geometry / dictionary discovery | Valid token/label joins, match-separated development, explicit nuisance/support analysis | Observational structure or sparse reconstruction |
| Corruption/restoration with native output loss | Reproducible inputs, exact hooks, paired noise, effective corruption, controls | Causal influence of the intervention on specified model behavior |
| Natural-donor activation interchange | Verified donor/recipient correspondence, identical recipient actions/noise, explicit donor differences | Model dependence on transferred information, with stated specificity limits |
| Independent measurement of generated trajectories | An evaluator adequate for the selected quantity on generated outputs, or validated direct/manual measurement | A measured generated-output physical effect |
| Isolated physical-state counterfactual claim | Evidence isolating the state change and relevant nuisance factors; simulator reset/rendering is one strong sufficient design | Specific dependence on the manipulated physical state under that design |
| Generalizable physical steering | Frozen reusable edits, unseen matches/values, monotonic doses, nuisance checks, rollout persistence | Control within the evaluated scope |

[Attribution patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
uses a gradient approximation to the output effect of replacing an activation.
It supplies a ranking heuristic, especially fragile for large residual changes;
exact replacements should check it. [The activation-patching interpretation
paper](https://arxiv.org/html/2404.15255v1) emphasizes that different corruptions
trace different properties and that noising and restoration answer different
questions. Neither method requires a physical simulator as a prerequisite to
an intervention on the neural network. Their results remain specific to the
chosen contrast, metric, and replacement distribution.

Concretely, the previous zero-pair result covers only 868 comparisons among eight
clips per discovery match. It is neither a search over all recorded frames nor a
test of whether model interventions are executable. Requiring 29 state coordinates
and every source action to match exactly is exceptionally stringent for naturally
recorded continuous trajectories, and unnecessary for a first model-level patch.

For a natural-donor pilot, choose matched contexts using discovery-only tolerances
for measured nuisance variables and contrast one declared target variable. Run the
recipient baseline and every patched recipient under **the same supplied four-player
future actions** and noise tensors. The donor's originally recorded future actions
do not have to equal those supplied to these recipient runs. If donor activations
depend on future actions, compute the donor forward using the same supplied future
actions too. Retain differing histories and state variables in the audit: this still
does not create an isolated simulator counterfactual or a true target rollout.

An even simpler development experiment corrupts a specified observed-video region
or a defined action input while holding the rest of the model inputs fixed, then
tests exact activation restoration against the uncorrupted output using the native
denoising loss. This tests the model's processing of that corruption. A mask can
remove several semantic properties; an action edit is an action experiment, not a
velocity intervention. State those limits instead of withholding the experiment.

Use no-op, random norm-matched, wrong-time, and identity-valid donor controls;
test both directions and restoration. A physical probe at the edited site is not
an independent physical-effect metric. Native loss, decoded pixels, an independent
output readout, and eventual physical measurements answer different questions.

The proposal can therefore progress now through improved readouts, observational
geometry, sparse discovery, and exact model-output patching. Calibrated causal
physical-state identification and persistent steering remain stronger goals to
earn with additional evidence. New choices made after seeing v2 confirmation
require a new prospective analysis version; the already inspected 11 confirmation
matches cannot be presented as untouched confirmation of those choices.
