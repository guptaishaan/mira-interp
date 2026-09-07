# Prepared generated-rollout steering protocol

**Status: code and CPU tests prepared. No rollout steering GPU run has been
completed.** The first reserved pilot failed before baseline generation at the
keyboard embedding because source actions were uint8. Its code, registration,
log and exit are preserved in `results/rollout_attempts/01_action_dtype`. The
corrected inference adapter makes an int32 copy, as the original input loader
does, and checks that every numeric action value is identical. It is re-registered
before repeating the reserved pilot; no research outcome informed the correction.
Registration requires the passed extended-input audit, frozen probe
and geometry maps, and completed independently audited sparse causal fidelity.
The fidelity prerequisite concerns valid execution; its effects need not be
positive. The executing agent reviews the reserved pilot before committing the
full runtime and storage budget, within the user's existing authorization.

The input cohort contains one reserved pilot, 10 selection matches and 22
confirmation matches. Each match contributes one independently audited
24-frame clip. The first 16 frames match the prior fractional-FP32 preparation
exactly. Two planned matches failed the extended demolition-quality criterion
and were excluded without replacement. Every original future video pixel in
frames 16–23 is replaced by zero **before** calling the published
`model.inference`. Source simulator targets are not loaded by the generator.
The pilot compares an alternate 255 fill to detect any future-video dependence.

The model uses its published 10-step `linear_quadratic` schedule, noise level 0,
and 16-frame inference context. A fresh call to `model.inference` initializes a
fresh local streaming KV cache for each condition. The supplied 24-frame action
batch is unchanged. The upstream API slices action windows `[1:17]`, `[3:19]`,
`[5:21]`, `[7:23]`; each final generated latent therefore receives the aligned
action pairs `[15:17]`, `[17:19]`, `[19:21]`, `[21:23]`. The last source action
at index 23 is unused by this published alignment. No independent manual action
shift is applied.

The proposed τ=0.5 trigger was corrected **before any rollout experiment**:
the default schedule has no such step. Its FP32 grid is approximately
`[0,.02,.04,.06,.08,.1,.205193,.347789,.527789,.745193,1]`.
The intervention uses index 8, the nearest scheduled forward, without modifying
the schedule. The published sampler casts tau to the codec-latent dtype;
under BF16, the model actually receives **τ=0.52734375**. Both nominal and
realized grids are registered, and each actual forward records tau, dtype,
time-token count, cache presence and action/latent input hashes. This differs
from earlier teacher-forced FP32 tau0.5 experiments and is a distribution change.

One residual edit is applied to block 15 output, view 0, during that step of the
**first generated latent only**. Its support is the single 9×16×2048 view tile.
The remaining denoising steps and next three generated latents are unedited.
The τ=1 context-cache initialization and generated-cache updates are explicitly
excluded. A strict trace expects one context call, four sets of 10 denoising
calls, and four cache updates: 45 calls total, with one intervention-site hit.
The final block's tuple/cache auxiliary and every other view are preserved.

The baseline ball height comes from the frozen spatial ball-z probe applied to
the baseline first-generated-latent residual at the registered step. Clamp
this estimate and each requested target to the fixed discovery support
`[86.37890625,1810.9219970703125]` simulator units. Every report retains the raw
estimate, clipped base, clipped target, requested dose, effective dose and
clipping flags. Doses are −600,−300,0,+300,+600 units, with one shared zero baseline.

The five candidate paths are:

1. `probe_linear`: the minimum-norm descriptor offset producing the effective
   dose under the fixed linear probe, including its discovery normalization.
2. `affine_forward`: frozen forward map `f(target)-f(base)`.
3. `quadratic_forward`: the corresponding frozen quadratic map difference.
4. `random_norm`: a paired fixed random direction with the quadratic lift norm.
5. `wrong_variable_ball_x`: a ball-x probe direction with that same norm.

Each descriptor offset is lifted through the fixed channel projection into
the native residual, preserving its descriptor nullspace up to rounding.
Control directions are fixed for a clip/seed and use the effective dose sign.
Nominally clipped duplicate conditions remain reported; only the common zero
is deduplicated. The result is 21 conditions per pair. Requested and realized
descriptor/raw edit norms and all 15 position-probe changes are recorded,
including any BF16 rounding effects. Forward-map fits did not establish a
nonlinear physical manifold; these are candidate paths, not validated manifold
coordinates.

The reserved pilot runs five complete inferences: unhooked baseline, hooked
baseline, alternate future placeholders, zero edit, and an active quadratic
edit. Its engineering dose is+300 unless the baseline clips at the upper support
boundary, in which case it is−300. This fixed rule ensures a nonzero supported
edit and changes no research dose or path. It requires bitwise raw decoded-video/latent equality for the first
four conditions and exact paired pre-edit residuals. It measures runtime and
compressed artifact size before the full budget review. The planned main
cohorts require 420 selection rollouts and 924 confirmation rollouts, using the
two original paired seeds. Selection completion is independently audited before
the separate confirmation phase; no rollout outcome selects a new path or dose.
These confirmation matches were used for observational decoding earlier;
their generated steering outcomes have not been used to choose this protocol.

The two main workers use disjoint matches in registered order, indices `0::2`
and `1::2`, on the two authorized GPUs. Each match retains both paired seeds and
all21 conditions. After both workers finish, `--aggregate` verifies the complete
condition grid and every report/artifact hash before writing the phase manifest.
The independent completion audit remains required before confirmation. Workers
stop before consuming the25GiB disk reserve. This execution partition changes no
experimental choice, input, random seed, or model arithmetic.

```bash
# Run these two workers concurrently with CUDA_VISIBLE_DEVICES=6 and7 respectively.
.venv/bin/python scripts/rollout_steering.py --phase selection --worker-index 0
.venv/bin/python scripts/rollout_steering.py --phase selection --worker-index 1
# Then combine, independently audit, and use the same partition for confirmation.
.venv/bin/python scripts/rollout_steering.py --phase selection --aggregate
```

Private generated NPZ files contain `frames` FP32 `[4,24,3,288,512]` in 0..1,
`latents` FP32 `[4,12,9,16,32]`, and native/edited descriptor and residual data.
Decoded values are clipped to 0..1 for the evaluator contract; original raw
decoder hashes, extrema and clipping counts are recorded. Public exports must
contain only `frames[:,16:]` and `latents[:,8:]`, never the observed context.

Each manifest record supplies `record_id`, `match_id`, `clip_id`, `split`,
`seed`, `intervention_type`, `dose`, `generated_artifact_path`,
`generated_sha256`, and `baseline_record_id`. The shared baseline points to
itself. All records also report whether the first 16 decoded frames remain
bitwise identical to baseline, their maximum pixel difference, fixed-context
latent equality, and later generated-latent changes. Context-pixel changes are
reported because the independent evaluator's rolling 16-frame windows include
context. The evaluator runs separately with its own frozen arithmetic settings
and reads windows ending at frames 17,19,21,23.

Successful generation is not physical steering success. Monotonic ball-height
changes, unrelated-state changes, persistence across the four generated latent
times, and the evaluator's own generated-domain validity remain separate
measurements. No such result is asserted by this prepared implementation.
