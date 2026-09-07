# Review: probing and causal identification

Reviewed 2026-09-07 against the three requested primary references and the actual
MIRA, capture, probe and intervention implementations. This review does not change
the frozen v2 analysis or run a new model fit. The 11 v2 confirmation matches have
now informed our next questions: they are development data for future choices.
New confirmatory claims require new untouched matches and a new registration.

## Main conclusion

The v2 ridge calculation is correct, but the experiment is too narrow to test the
proposal's central physical-state and control claims. Its weakest design choices
are spatial averaging, a shared viewpoint-unconditioned canonical-player target,
and probing a single noisy reconstruction setting. They can suppress accessible
information even when the model represents it. Conversely, successful decoding
with observed target pixels cannot demonstrate prediction or causal use.

There is no established coding defect that explains the weak velocity results.
The priority is to test these specific readout/target hypotheses on development
data, while preserving the original negative and positive results. Increasing
probe flexibility until a number improves is not an adequate repair.

## What the cited methods actually establish

**Nanda, Lee and Wattenberg, 2309.00941v2.** Sections 2–4 and Appendix E show that
the feature definition matters: relative Mine/Yours labels reveal structure that
absolute Black/White labels obscure. The probe is a linear three-class classifier,
not ridge regression. The authors subsequently test model-output consequences
against an independently computable altered board, rather than relying on probe
accuracy alone. Their Appendix B also warns that a single-layer intervention may
be repaired downstream. These are transferable methodological lessons, not an
accuracy target for video or a prescribed regularization grid.
[Paper, including appendices](https://arxiv.org/html/2309.00941v2).

I also inspected the primary implementation. `board_probe.py` trains separate
layer probes with AdamW and preserves a separate output for each board square;
`intervene.py` uses normalized probe directions and simulator legal-move sets.
Its checked implementation applies a layer-5 probe at several attention-output
sites. This makes matching the precise hook and intervention schedule important
when reproducing the claimed experiment.
[Probe source](https://github.com/ajyl/mech_int_othelloGPT/blob/58aa116a18076e040c8b828b81a31e8799bced4c/mech_int/board_probe.py),
[intervention source](https://github.com/ajyl/mech_int_othelloGPT/blob/58aa116a18076e040c8b828b81a31e8799bced4c/mech_int/intervene.py).
The commit was verified from the authors' repository during this review.

**Nanda, attribution patching.** The method differentiates the selected metric
with respect to the corrupted activation and contracts that gradient with the
clean-minus-corrupted difference. It is an exploratory first-order estimate.
The article specifically reports poor approximation for large residual-stream
patches and discusses normalization, saturation, nonlinear composition and
interactions. Exact patches must test shortlisted sites and a sample of sites
that attribution ranks low; otherwise false negatives can disappear from the
study. Read sections “What Is Attribution Patching?”, “Does This Work In
Practice?”, and “Does This Make Any Sense?”.
[Primary article](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching).

**Heimersheim and Nanda, 2404.15255.** Sections 2–4 distinguish clean-to-corrupted
restoration from corrupted-to-clean disruption: they answer different conditional
sufficiency/necessity questions. Corruption choices determine the property being
tested; redundancy and recovery can obscure component effects. A recovered
difference score can reflect damage to the corrupted answer rather than restored
correct behavior, so individual outcomes and multiple metrics matter. These
claims are tied to the tested input distribution, not a universal circuit.
[Paper](https://arxiv.org/html/2404.15255v1),
[requested abstract link](https://arxiv.org/abs/2404.15255).

## Findings against our implementation

### 1. Spatial averaging is a demonstrated information bottleneck

[`pool_views`](../../scripts/capture_observations.py) averages each view's 9×16
spatial grid into one 2048-dimensional vector. A permutation of those 144 spatial
cells leaves this descriptor unchanged. A location code carried by where a
feature occurs is therefore unavailable to the readout unless upstream processing
has already converted it into a change in channel averages. This is a mathematical
property of the measurement, not a speculative problem with the trained model.

The weakness is particularly sharp at block-0 input. In upstream
[`diffusion_transformer.py`](../../external/mira/src/mira/world_model/diffusion_transformer.py),
lines 53–58 and 135–152, that residual is

`h0(s) = W_current z_tau(s) + W_past z_previous(s) + b`.

Both projections are affine, with positional RoPE and action conditioning used
later inside the blocks. Therefore the spatial mean of `h0` depends on only the
32 current-latent means and 32 previous-latent means, plus a constant, for this
patch-size-one checkpoint. Nominal width 2048 does not make this a general
2048-dimensional spatial readout. This rank bound is an exact-arithmetic
statement, not a measured numerical rank: finite-precision projection can break
it. Spatial permutation invariance of the pooling operation still holds.
The codec's contextualized channels can themselves encode spatial information;
averaging does not prove that all location information is absent. It proves that
the readout cannot distinguish latent fields having the same channel means.

The frozen result—MSE 1.057696 at input versus 0.846902 at block-5 output—does show
better accessibility to this readout after transformer processing. It does not
show that physical information was absent from the input or newly created at
block 5. This baseline comparison is much less mechanistically decisive than it
initially appears.

**Repair:** compare global means with fixed coarse spatial bins and first spatial
moments. Keep projection seeds fixed before labels are used; report readout
dimension and effective rank. Apply identical spatial summaries and tuning to
codec and suitable frozen visual-encoder baselines. Preserve the old descriptor
as an ablation. Entity-centered crops require an independently available detector;
using ground-truth 3D labels to choose crops would introduce privileged input.

### 2. The target has unnecessary viewpoint and identity demands

[`physics_targets`](../../src/mira_interp/data.py) correctly joins persistent
player IDs and produces canonical player0–3 world XYZ targets. The row metadata
records the local view, but [`select_models`](../../src/mira_interp/probes.py)
does not condition its coefficients on `view_index`. The same map must recover
all four canonical players from a feature vector taken from each different camera.
Correct identity joins do not establish that this is the coordinate system the
model uses internally.

This creates a plausible role-binding difficulty: the locally controlled car
changes canonical output slot with viewpoint, while visual roles such as local
car, teammate and opponent are more consistent. The model may encode enough
perspective information to solve this, so it is not a proven impossibility or a
label-swap bug. World XYZ also asks a linear readout to undo varying camera
orientation. The Othello feature-definition lesson motivates testing alternatives;
it does not guarantee they will work here.

**Repair:** first compare canonical targets against local-player state and
ball-minus-local-player position/velocity. Evaluate relative coordinates as
relative coordinates; do not silently reconstruct world predictions using true
player pose. Compare shared coefficients with explicitly view-conditioned
coefficients, implemented as `one_hot(view) ⊗ features`. Merely appending a view
one-hot changes intercepts and does not allow view-specific feature mappings.
An ego-heading rotation is a further hypothesis requiring orientation convention
tests. Track opponent identities consistently rather than sorting by distance
and introducing discontinuous identity swaps.

### 3. Ridge scaling is correct; inadequate baseline tuning is still plausible

The implemented objective is
`sum_i w_i ||T_i - Z_i B||² + alpha ||B||²`, with `sum(w)=1` and discovery-only
standardization. Its coefficients solve this objective: the independent
[`selection audit`](../../results/probes_v2/selection_audit.json) found maximum
normal-equation residual `5.59e-15` across all 38 fitted coefficient sets. There is
no missing sample-count factor relative to the registered objective.

All discovery matches have 256 rows, so an equivalent **unweighted sum-SSE**
implementation would use `lambda = 7936 * alpha`. Thus our alpha=1 is not the same
as a library's alpha=1 under an unnormalized sum objective. Feature standardization
also changes the effective penalty in raw activation coordinates. These are
convention issues, not evidence that every fitted probe was overregularized.

The saved selection curves argue against that simplistic diagnosis: every
residual-output site chose the interior value alpha=1, and alpha=0.1 was worse.
At the frozen winner the selection MSEs were 0.94377 at 0.1, 0.89834 at 1, and
0.95885 at 10. Codec and RGB instead chose the smallest tested alpha=0.1, so their
low-regularization range was not resolved. A stronger baseline might narrow the
apparent residual advantage.

**Repair:** in a new development protocol, use an expanded fixed grid such as
`1e-6, 1e-5, …, 1e3`, retaining the normalized loss convention. Report discovery
and development losses, effective degrees of freedom, coefficient norms and
boundary selections. Compare per-variable-family tuning with the current
30-target average: one alpha/site is a compromise that can obscure velocity
even when position is accessible. Do not change the published v2 choices.

### 4. One reconstruction setting does not test a trajectory representation

The capture uses `z_tau = 0.5*z_observed + 0.5*noise`, with clean previous observed
latents, one seed per clip, eight latent pairs, and a BOS reset at each clip start.
This matches a valid upstream reconstruction input, but is not an autoregressive
rollout. The first pair has no observed past; later pairs have different available
histories, yet a common readout mixes them. A 16-frame window covers only 0.8
seconds at the source rate, with 0.75 seconds between first and last frame.

The source convention is `z_tau = tau*z_clean + (1-tau)*noise`: **tau=0 is pure
noise; tau=1 is clean**. Any sweep must use this convention consistently. Velocity
can depend on motion over time and coordinate frame; a static spatial mean at one
latent index is a restrictive place to look. The correct own-view pair-final
label join was independently verified, but exact video/state latency remains
partly dependent on the publisher export contract.

**Repair:** after readout/target checks, compare tau=0, 0.5 and 1 with fixed paired
noise seeds, separate first-pair and later-pair summaries, and short causal
histories. For a predictive experiment, use the actual sampling path, replace all
unknown future latents with noise/placeholders, and require placeholder-invariance
and causal masking checks. Repeated seeds measure noise sensitivity; they do not
increase the number of independent matches.

### 5. The baselines do not isolate a world-model contribution

The codec baseline averages 32 channels over space; RGB uses a downsampled 8×8
image from the pair-final frame. They differ from the residual in dimension,
spatial treatment, previous-frame access and temporal processing. The action
stream also enters transformer blocks but not the raw visual baselines.
Consequently, the residual advantage does not distinguish learned world dynamics
from a better visual descriptor, action correlations, temporal context or pooling.

**Repair:** use the same spatial and temporal summary for codec, frozen DINO
features, and residuals; disclose capacity differences or add fixed-dimensional
projections. Include action-only, history-only and simple persistence/constant-
velocity baselines where the input information is actually available. A
random-initialized transformer with the same frozen pretrained codec is a useful
architecture baseline, but it must be reported as such. Avoid treating weak codec
means as the strongest visual baseline available.

### 6. The missing causal experiment is a missing stage, not a failed patch

The trained scientific run did not perform attribution patching, exact physical
replacement, ablation/restoration, or generated-trajectory evaluation. Hook and
linear-gradient smoke tests establish engineering behavior only. The saved pooled
research activations are detached, and capture runs under `torch.inference_mode()`;
they cannot supply the required gradients. A separate gradient-enabled forward
and a specified downstream metric are necessary. The helper's contraction formula
matches the attribution article, but that alone establishes no physical effect.

The availability audit found no exact all-player future-action matches among 868
within-match pairs from the sampled discovery clips. That is not a proof that the
full dataset contains no useful contrasts, and natural pairs were never a
substitute for simulator-controlled state resets. Conversely, arbitrary video
replacement can test the model's response to an input difference without becoming
a single-variable physical intervention.

**Repair:** first establish a behavior the model actually exhibits and a stable
output metric. For a narrow exploratory test, choose and document a reproducible
contrast, save full relevant tokens, rank by attribution, then test exact patches
in both directions. Report the contrast's nuisance differences. Such a result may
be a useful causal effect inside the network while remaining short of physical
state control. Separately build reset/render/action-fixed counterfactuals and
validate an independent physical evaluator on generated video before stronger
claims. Do not require perfect 30-variable probing before any narrow causal study,
but do not call probe-output movement a generated physical trajectory change.

## Development-only experiment matrix

Run these sequentially, preserving failed configurations. No new GPU experiment
was launched as part of this review.

| Priority | Question and minimal comparison | Main check | Next-step criterion |
| --- | --- | --- | --- |
| 0 | Audit role remapping, coordinate conventions, time joins and exact pooling information loss on deterministic fixtures | Permuted IDs/view order map correctly; spatially moved features preserve global means but change spatial descriptors; no ground-truth-dependent feature extraction | All semantic tests pass before a new capture |
| 1 | Existing pooled features: shared canonical vs local-player/relative targets; shared vs view-conditioned coefficients | Match-separated development folds; same sample cohort and tuning budget; all target families reported | Choose a scientifically defined target/readout, not merely the best observed scalar |
| 2 | Existing pooled features: expanded alpha grid; family-specific vs shared alpha | Inspect only development folds; identify boundary minima; compare mean and shuffled-label controls | Freeze a resolved range or record that added flexibility does not help |
| 3 | Recapture all 17 sites with mean vs coarse spatial bins/moments; matching codec/DINO summaries | Fixed projections and equal information access; descriptor dimensions/ranks; paired development errors | Retain a spatial readout only if it provides reproducible incremental access |
| 4 | Chosen readout: tau0/0.5/1, multiple fixed seeds, context length and first-pair exclusion | Tau convention; no future-frame access in predictive arm; velocity and nuisance controls | Specify which representation at which sampling time supports the tested variable |
| 5 | Narrow behavior contrast: attribution vs exact patches in both directions | Rank agreement and missed important sites; random/wrong-variable/wrong-player controls; restoration plus output-quality metrics | Establish a reproducible internal causal effect with honest contrast scope |
| 6 | Independently rendered single-variable changes with fixed future actions | Independent generated-video evaluator; off-target states and rollout persistence | Register physical-control confirmation on new untouched matches/values |

For a modest first recapture, a 2×2 spatial summary with a fixed label-independent
channel projection can retain the original 2048 final features. Full 2×2×2048
readouts can be a separately budgeted comparison, not silently treated as matched
capacity. Finite spatial resolution is still a restriction; record it explicitly.

All 53 previously analyzed matches may now support development, with grouped
inner/outer development folds used to reduce feedback overfitting. Calling those
folds a new independent confirmation would be incorrect. Acquire new matches,
freeze their identities before new model/label inspection, and make no unseen-to-
the-world-model claim without a training-cohort manifest. Existing quality and
timing failures remain preserved; no threshold should be relaxed to improve a
probe score.

## What remains valid

The publisher's [MIRA Mini technical report, §17.1](https://alakazam.gg/mira-mini/report.pdf)
provides useful diagnostic context: its single-player teacher/student experiment
compares ridge with a three-layer, 1024-hidden-unit MLP. It reports position nMSE
0.22–0.49 but velocity nMSE around0.9–1.1, near the mean predictor, despite the
nonlinear readout. This is not our multiplayer checkpoint, state definition,
cohort, capture setting or evaluation, so it neither replicates our result nor
establishes a velocity ceiling here. It does show that weak velocity is not
unique to our initial linear analysis. Our inference is to repair measurable
readout/numerical restrictions without promising that those repairs must reveal
strong velocity. That report also distinguishes real-sequence probe error from
generated-rollout error; our clean-codec-trained output proxy needs the same
distribution-shift qualification before supporting physical claims.

The frozen v2 result is partial annotation decodability: ball position is much
more accessible than horizontal velocity under one restricted readout. It is not
a null result for all physical representations. Its match separation, full-layer
reporting, independent coefficient audit and explicit failure lineage remain
useful. The original inference that this preliminary result was as far as the
proposal could practically go was too strong: targeted development experiments,
better baselines and narrowly scoped model-internal causal tests are available
before a complete simulator/evaluator pipeline exists.
