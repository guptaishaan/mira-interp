# Frozen evaluator for generated rollouts

The generated-video helper freezes the independently audited VideoMAE step500
checkpoint. Reserved generation and measurement pilots have since completed and
passed independent audits, including exact prediction parity after a lossless
storage change. See [the v2 pilot audit](../results/generated_evaluation_v2/pilot_audit.json).
There is no training or parameter-selection interface. Outputs are learned state
estimates, not ground-truth simulator state. The full study's current stage is
recorded separately in [the execution snapshot](../results/current_execution.json).

## Inputs and alignment

The sampler provides `frames` as float32 `[4,24,3,288,512]` RGB in 0..1: 16 context
frames plus eight generated frames, at 20 FPS. The evaluator preserves all four views
and processes these rolling windows:

| Final frame, zero indexed | Source frame indices in the 16-frame window | Time after context |
| --- | --- | ---: |
| 17 | 2–17 | 0.10 s |
| 19 | 4–19 | 0.20 s |
| 21 | 6–21 | 0.30 s |
| 23 | 8–23 | 0.40 s |

No later frame enters an earlier measurement window. Each window ends on the
second frame of its last VideoMAE tubelet, matching the endpoint-training
convention. Every view keeps its own identity and timing; this is not a claim
that the four original cameras share an exact absolute timestamp.

`FrozenRolloutEvaluator` strictly loads the original encoder and audited adapted
last two blocks/head, verifies checkpoint/report/registration/code hashes, and
freezes every parameter. It uses the calibrated FP32-parameter/BF16-activation
pipeline, deterministic math attention, and deterministic FP32 spatial averaging.
The input is converted from 0..1 to the established 0..255 preprocessing convention
once. Full-frame 224x224 resizing and published channel normalization are unchanged.

Run in a separately allocated evaluator process with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `NUMPY_MADVISE_HUGEPAGE=0` set before Python.
Explicitly enable `torch.use_deterministic_algorithms(True)`, then construct
`FrozenRolloutEvaluator(repo, device='cuda')`. The helper does not choose a GPU;
CPU is the loader's default, and actual inference requires an explicitly supplied
CUDA device to retain the verified mixed-precision arithmetic. A new reserved
forward/timing check is required before full generated evaluation.

## Predictions and paired interventions

The output contains `role_predictions[4,4,12]`, ordered view, endpoint, coordinate.
The coordinates are ego XYZ position/velocity followed by ball-minus-ego XYZ
position/velocity in world axes. `absolute_ball_predictions[4,4,6]` adds the two
six-coordinate vectors. In particular, absolute ball height is
`role[...,2] + role[...,8]`. This sum retains correlated component errors; adding
component RMSEs as though independent would be incorrect.

Each generated manifest record must contain:

```text
record_id, match_id, clip_id, split, seed, intervention_type, dose,
generated_artifact_path, generated_sha256, baseline_record_id
```

A common-zero baseline has zero dose and points to itself. Every edited record
must explicitly point to a baseline with the same match, clip, partition, and
seed. Duplicate identities/conditions, mismatched baselines, missing views/times,
and nonfinite predictions fail validation. `paired_changes` returns all 12 role
shifts and all six derived absolute-ball shifts at every view/endpoint.

Subsequent dose analysis must retain all registered conditions, include the shared
zero when ordering doses, report monotonicity separately at each generated
endpoint, and aggregate uncertainty by whole match. Four views, four endpoints,
and repeated seeds do not create independent matches. Changes in other predicted
coordinates should be reported; physical coupling means they are not automatically
an intervention error. Baseline drift and seed variability must remain visible.
The sampler must also expose any decoded context-pixel change across conditions,
since the measurement windows contain some context frames.

Each window also retains generated frames16–17, where the first edit enters the
video. Overlapping-window estimates do not independently establish persistence
of a physical effect in later frames. Requested probe/map doses are expressed
in simulator units; they are not verified realized physical displacements.

## Calibration limits

`results/generated_evaluator_reference_calibration.json` recomputes empirical
role and derived-ball errors from the frozen evaluator's recorded development
selection predictions. It includes real/codec MAE, RMSE, and the range of
per-match MAEs. This data also selected the evaluator checkpoint, so these are
optimistic development reference errors, not independent calibration intervals.
No generated-video accuracy or predictive uncertainty interval is established.
The absolute reference errors are not paired-change detection thresholds:
correlated errors can cancel, while edit-dependent bias can remain.

The recorded source future is not ground truth for an edited generated video.
Disagreement with that source mixes model rollout error, intervention effects,
and evaluator error. A monotonic evaluator response can be useful causal evidence
about model pixels while still failing to establish an accurate physical-state
change. Generated-domain accuracy, camera/entity identity failures, and visibility
or occlusion require separate validation. The fixed evaluator will not be tuned
on these generated rollout outcomes.

## Extended-input integrity

`scripts/audit_rollout_inputs.py` independently passed all 33 prepared inputs:
one reserved pilot, ten selection matches, and 22 later evaluation matches. Their
original 16 **float** pixels, labels, actions, presentation timestamps, and source
indices match the corresponding preexisting float preparations exactly. Every
input, component, source registration, and implementation binding was checked.
The only two source exclusions are explicitly whitelisted demolition cases;
there was no identity, action-schema, or other integrity exclusion hidden by the
preparation script's broad ValueError wrapper.

The executed preparation, registration, and original report were preserved.
`results/rollout_inputs_v1_audit.json` binds them with status
`passed_rollout_input_audit` and SHA256
`73d72cd7bcbed79a1a6976e9a4c962688c899c17b23db39a307403a9e515927d`.
Preparation contains future reference frames; the generator must explicitly
consume only the first 16 pixels while using its registered future action stream.
These later matches already have separate observational readout results, so they
must not be described as wholly uninspected matches.

Four synthetic helper tests cover endpoint windows, exact absolute-ball coordinate
sums, input pixel units/shape, and explicit paired-seed/baseline validation. These software tests are not
substitutes for the separately completed generated-video evaluator pilots.
