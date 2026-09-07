# What went wrong, what changed, and what the evidence supports

The clearest successful repair was the target convention. Frozen viewing-player
and ball-relative position probes reduced error by 80.9% against their mean
baseline on 23 fresh matches. This is reliable annotation decoding with observed
target frames present. It is not yet reliable control of generated physics.

| Suspected problem | What the audit found | Repair and outcome |
|---|---|---|
| Video/state alignment | Some matches failed unchanged event-clock calibration rules. | Explicit source-clock origins, exact frame/state/action joins, and quality exclusions. Failed cohorts remain preserved; no replacement after exclusion. |
| Model precision and preprocessing | Whole-checkpoint BF16 conversion changed codec/prediction values; integer resized pixels lost fractional information. | Keep FP32 checkpoint parameters/buffers, BF16 autocast and fractional resized RGB. Reserved numerical tests quantify the differences. This was not an isolated causal test of the final score improvement. |
| Player identity and target convention | Canonically ordered absolute player states were harder to read than viewing-player and ball-relative states. | Freeze role-relative targets and development choices before fresh capture. Position decoding improved substantially; horizontal velocity remains weak. |
| Mean pooling | Its information bottleneck was real, but the tested spatial projection did not improve held-out decoding. | Capture all 17 sites and compare both readouts with codec/RGB baselines. Pooling alone is not a demonstrated explanation for weak results. |
| Treating readability as causality | A linear probe can decode information that the model does not use through that direction. | Attribution screening followed by 660 exact interventions and controls. Full donor gains were positive on only 6/11 matches; probe-direction gains were tiny. |
| Over-crediting sparse reconstruction | Whole-tile donor effects survived even when the compressed descriptor was replaced entirely by its discovery mean. | Add recipient/donor mean-descriptor controls. All 1,364 fidelity conditions were audited. Dictionaries improve conditional reconstruction modestly, but the untouched native complement carries nearly all average donor effect. |
| Assuming a nonlinear physical manifold | Quadratic height/speed/vertical-velocity gains were small with intervals crossing zero; extra heading harmonics worsened prediction. | Preserve held-value/held-match comparisons and all negative results. Candidate quadratic paths are not validated manifold coordinates. |
| Assuming temporal feature recovery | The shuffled-time block control matched or exceeded correct temporal pairing. Descriptors cover a whole view, not a localized tracked ball. | Compare matched capacity/activity budgets and report achieved reconstruction/activity. No temporal-specific or tracked-entity BSF benefit is established. |
| An inadequate video evaluator | Frozen VideoMAE features were weak for several state components. | Train heads, then adapt the last two blocks under deterministic gradient checks. Real-video NMSE improved 0.797→0.698; codec reconstruction NMSE is 0.706. Generated-video accuracy remains unverified. |
| Patching the wrong sampling time | The published 10-step schedule has no tau 0.5 step; BF16 also rounds the actual schedule value. | Preserve the schedule and record the actual step8 tau 0.52734375. Pilot traces verify 45 calls and exactly one edit, excluding cache writes. |
| Action tensor type at inference | The first reserved generation attempt passed byte-valued keyboard indices into an integer-only embedding. | Make an int32 copy and assert identical values. Preserve the failed attempt, then rerun and independently audit the complete pilot. |
| Slow execution despite available GPUs | Default array compression took longer than model inference per rollout. | Benchmark lossless encodings, use DEFLATE1, and rerun the pilot. Every saved model array and evaluator prediction matches the original pilot bitwise. |

The full generated experiment now tests all five predeclared paths and both
paired seeds: 420 selection and 924 confirmation rollouts. It measures four future
times after one edit, actions held fixed, against norm-matched random and
wrong-variable controls. Each generation and measurement phase requires an
independent audit before the next phase. No generated result chooses a new path,
dose or evaluator.

Reliable physical-control claims still require more than these engineering
checks: consistent target movement, specificity, persistence, and a measurement
method validated on edited generated video. The experiment can legitimately
return a negative result. A positive hypothesis is not an execution gate.

The additional primary-source reading identifies prospective controls rather
than another demonstrated implementation bug. Norm-matched isotropic random
edits do not account for activation covariance, and a readable state direction
need not be a direction that produces that state. A focused follow-up would test
covariance-matched controls and separate descriptor effects from the retained
native complement under actual generation conditions. Those are new hypotheses;
the current registered outcomes cannot select them and also confirm them. See
the [first](review/anthropic_extended_review_a.md) and
[second](review/anthropic_extended_review_b.md) extended reviews.

Detailed evidence: [literature review](08_review_and_repair.md),
[fresh decoding](11_fresh_confirmation_results.md),
[exact causality](09_internal_causal_development.md),
[geometry](review/geometry_development_results.md),
[sparse comparisons](review/sparse_development_audit.md),
[conditional fidelity](review/sparse_causal_fidelity.md),
[video evaluator](review/video_evaluator.md), and
[current execution](../results/current_execution.json).
