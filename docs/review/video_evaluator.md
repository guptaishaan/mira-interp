# Independent VideoMAE state evaluator

The proposal's video-state evaluator is implemented as a frozen VideoMAE encoder
with a regularized linear state regressor. This first baseline uses independent
video-model features: MIRA residual activations, actions, and simulator labels are
never encoder inputs. Source labels are joined after pixel encoding. It is a
measurement-development experiment, not yet a validated sensor for generated
trajectories. No full VideoMAE fine-tuning has been performed.

## Source and loading

The official [VideoMAE model](https://huggingface.co/MCG-NJU/videomae-base) is
Kinetics-400 self-supervised pretraining, intended for downstream adaptation; see
also the [authors' code](https://github.com/MCG-NJU/VideoMAE). The pinned revision is
`dc740ceda42fce44faed2ea03c6d447db72f6af9`; the 376,873,760-byte safetensors file
has SHA256 `bc053ca2840a038b1068269a4eec06ca569689e9a1ed9376a5b2b8a111be5290`.
Model files remain under `/data2/ishaangp/mira-interp/models/videomae-base` and are
not redistributed in Git. The source model card specifies CC-BY-NC-4.0.

The existing environment already contains Transformers 5.14.1, Torch 2.9.0+cu126,
and the required torchvision/safetensors libraries. No dependencies were changed.
The checkpoint was written against Transformers 4.22: its separate query/value
biases must be mapped to current Linear-module biases, with the key bias fixed at
zero. This follows the [official older forward implementation](https://github.com/huggingface/transformers/blob/v4.22.0/src/transformers/models/videomae/modeling_videomae.py).
All 250 checkpoint tensors are consumed; the 16 attention modules introduce only
16 known zero key biases. All 266 current state entries then load strictly. Every
converted Q/K/V projection matches the original formula exactly on a deterministic
input. The initial unadapted strict-load failure is preserved under
`results/videomae_attempts/01_legacy_bias_names.json`.

The encoder has 86,236,416 frozen parameters. It stays FP32 and runs without mixed
precision. A reserved pilot passed exact repeatability, finite feature/label
checks, source timing checks, and codec reconstruction. The loading, download,
and pilot reports preserve hashes and implementation details.

## Video, descriptors, and targets

Every input is one original camera view: 16 frames at 20 FPS, 288x512, with the
float-preserving MIRA resize already applied. We resize the whole image to 224x224
using FP32 bilinear interpolation with antialiasing, divide source 0..255 values by
255 once, and apply the published ImageNet mean/std. This deliberately changes
aspect ratio. The official short-side resize plus center crop would remove 43.75%
of the horizontal camera view; retaining camera edges is useful for this spatial
state task. That preprocessing choice is explicit and fixed before fitting.

VideoMAE produces 8 temporal tubelets and a 14x14 spatial token grid with 768 channels
and **no CLS token**. We retain the last tubelet, pool into 3x4 spatial bins, and
keep all 768 channels: 9,216 features per view. The adaptive bins use the framework's
floor/ceiling boundaries and overlap when 14 is not divisible by the bin count.
This preserves coarse spatial structure; it does not localize tracked entities.
Self-attention can use all 16 supplied frames, and no later frame is supplied.

The endpoint is source frame 15, the last frame of the last tubelet (frames 14 and 15).
Each camera retains its own original timestamp and frame index. Targets are that
view's player XYZ position/velocity and ball-minus-that-player XYZ
position/velocity, all in **world axes**, 12 coordinates total. Relative velocity
is ball velocity minus player velocity, not camera-frame velocity. Original
absolute ball coordinates are retained for audit but not fitted by this baseline.
No timing broadcast across views is used. An independent audit recomputes each
endpoint target directly from the original 30-coordinate state array, verifies
view identity/time/frame equality, checks every capture hash, and rejects changed
source membership, roles, or confirmation rows.

The publisher's per-view frame/state-row alignment is still an upstream
assumption; these checks do not independently establish zero renderer latency.
They inherit the previously documented goal-clock calibration and state integrity
gates. Endpoint target pixels are present, so this is contemporary video decoding,
not future physical forecasting.

## Development fit and domain calibration

Only the original 31 discovery and 11 selection matches are used, with eight clips
and four camera views per match: 992 fitting and 352 selection observations.
Normalizer and ridge coefficients fit discovery only. Equal total match weights
handle the shared match as the independent unit. Ridge alpha is chosen on real
selection clips over the fixed grid 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 by mean discovery-
normalized squared error across the 12 coordinates. A whole-match deranged-label
regressor and discovery-mean baseline use the same observations. Raw MAE/RMSE,
per-target R², and per-match errors accompany the aggregate. Selection scores are
development estimates because they also select alpha; they are not confirmation.

For every selection clip, the strictly loaded original MIRA codec reconstructs
all 16 frames, and the same frozen VideoMAE/regressor processes those reconstructed
pixels. Codec parameters remain FP32 under BF16 autocast, as established by the
separate MIRA numerical audit. This path encodes and decodes the codec's own
unnormalized latent codes directly; no world-model latent standardization is
applied. Decoder values outside [-1,1] are clipped before RGB conversion and their
fraction is reported. Codec reconstructions never fit the regressor or select
alpha. We report their state errors against the original annotations and the
prediction change from the real video.

This calibration measures a useful domain shift. A reconstruction may itself
change visible physical state, and reconstructed source clips are not generated
rollouts. Before measuring an intervention, the evaluator needs task-specific
calibration on the intended generated domain, identity/occlusion checks, and
errors small relative to the intended physical dose. A better codec-domain score
alone cannot establish those properties. The analysis therefore leaves
`independent_generated_video_measurement_ready` false.

## Reproduce

```bash
python scripts/download_videomae.py
CUDA_VISIBLE_DEVICES=7 NUMPY_MADVISE_HUGEPAGE=0 python scripts/capture_video_evaluator.py --pilot
CUDA_VISIBLE_DEVICES=7 NUMPY_MADVISE_HUGEPAGE=0 python scripts/capture_video_evaluator.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMPY_MADVISE_HUGEPAGE=0 \
  python scripts/analyze_video_evaluator.py
```

The GPU above was explicitly allocated for this run; recheck ownership before
reproduction. Capture requires the same-code passed pilot and can resume only
hash-matching outputs. Analysis requires all 336 captures, audits them before
fitting, and refuses to overwrite existing analysis results. The original
qualified/frozen files are read only. Captures and fitted models are under
`/data2/ishaangp/mira-interp/video_evaluator_v1`; Git stores code, audits, aggregate
metrics, and documentation. Set `NUMPY_MADVISE_HUGEPAGE=0` before Python starts:
the host's huge-page pressure previously stalled unrelated NumPy allocations.

Engineering tests cover legacy-bias equivalence, full-image/unit preprocessing,
last-tubelet spatial pooling, and exact per-view endpoint/identity alignment.
These are synthetic software checks, separate from the reserved real pilot.

## Completed frozen-encoder baseline

The full capture passed all 336 clips in 439.6 seconds. The independent audit passed
1,344 endpoint observations and all 88 selection codec reconstructions. Ridge chose
alpha 10, inside the registered grid.

| Evaluation video / control | Selection normalized MSE |
| --- | ---: |
| Real source video, frozen VideoMAE + ridge | 0.796862 |
| MIRA codec reconstruction, same fitted regressor | 0.799465 |
| Discovery-mean prediction | 1.073727 |
| Whole-match shuffled training labels | 1.082860 |

Ball-relative XYZ position R² is 0.441/0.430/0.375 on real video, but its raw MAE is
817/1,367/282 Unreal units. Horizontal relative velocity R² is −0.033/−0.045;
this baseline is insufficient for precise velocity intervention measurement.
Codec transfer changes aggregate normalized MSE by 0.002602. Its source-image RGB
MSE averages 103.48 (0..255 units) across 88 clips, with no decoder clipping. Close
aggregate transfer does not erase the baseline's large physical-unit errors.

`results/videomae_analysis.json` contains all 12 target metrics, selection curves,
per-match real/reconstruction errors, model/prediction hashes, and scope limits.
`results/videomae_capture_audit.json` records the independent audit.
`results/videomae_exports/target_metrics.csv` has 48 rows, with MAE/RMSE/R² for
four variants and target-wise reconstruction prediction drift; `summary.csv`
contains the four aggregate scores. Export hashes are in `manifest.json`.
Figures `figures/videomae_state_r2.{png,pdf}` and
`figures/videomae_state_mae.{png,pdf}` display these exact reported metrics.
Both figure layouts were visually inspected. No bootstrap confidence intervals
are attached to these development comparisons.

The next authorized development repair compares a small trained MLP on these
frozen features, then optionally updates the final two VideoMAE blocks under a
bounded training plan. The baseline report above remains unchanged.

## Completed nonlinear-head development repair

A small MLP was trained on the unchanged frozen VideoMAE features: 9,216 inputs,
256 hidden units, GELU, dropout 0.1, and 12 standardized target outputs. All
normalizers are the existing discovery-only normalizers. AdamW uses learning
rate 0.001 and weight decay 0.01, batch 64, gradient clipping 1.0, and 1,000 updates for
each fixed seed 0/1/2. Every match has 32 rows, so uniform row sampling implements
equal match weight. No GPU was used for this cached-feature comparison.

The fixed checkpoint steps 100/250/500/1,000 are compared on real selection clips.
Checkpoint and seed choice use minimum aggregate normalized MSE. Reconstruction
scores are computed afterward and never select the model. These additional
selection choices reinforce the developmental scope of the result.

| Seed | Selected update | Real-video NMSE | Codec-reconstruction NMSE |
| --- | ---: | ---: | ---: |
| 0 | 1,000 | 0.736073 | 0.741364 |
| 1 | 1,000 | 0.728384 | 0.730212 |
| 2 | 500 | 0.719007 | 0.722256 |

All three fitted seeds score below the frozen-ridge 0.796862 baseline on this
selection cohort. The selected seed 2 improves ball-relative position R² to
0.538/0.560/0.418, while relative velocity R² remains 0.026/−0.052/−0.007. A more
flexible head helps some quantities; it does not solve velocity measurement.
There is no matched nonlinear shuffled-label model in this repair, and no fresh
confirmation has been used. `results/videomae_mlp.json` retains every checkpoint
score, each selected model/prediction hash, and real/codec target metrics.
The frozen-ridge report and figures remain unchanged.

## Partial fine-tuning

The bounded adaptation starts from the selected MLP and the original pinned
VideoMAE checkpoint. Only blocks 10 and 11 (zero indexed), the final layer norm,
and the MLP head can train. Encoder parameters remain FP32; BF16 autocast handles
GPU matrix operations. Because the first ten blocks are frozen and deterministic,
their outputs can be cached once, preserving the original computation while
reducing repeated video decoding and encoder work. Cache storage is FP32, with
native BF16 residual dtype restored before the trainable tail. Changing a
residual's arithmetic dtype would otherwise change the experiment.

The reserved pilot compares original FP32 descriptors against the earlier exact
capture, then compares full versus cached mixed-precision descriptors and every
trainable parameter gradient. It verifies unchanged frozen weights and updated
trainable weights after two discarded test updates. Full prefix capture and
training require this same-code pilot to pass. No research training updates are
retained from the pilot.

The registered training budget is 500 AdamW updates, microbatch 4 accumulated four
times (effective batch 16), backbone learning rate 0.00001, head learning rate 0.0001,
weight decay 0.01, and gradient clipping 1.0. Only discovery rows provide gradients.
Real selection data chooses among updates 100/250/500; codec reconstructions are
reserved for the selected checkpoint's domain-transfer measurement. The initial
mixed-precision model score is also recorded, separating numerical transition
from learning. No fresh evaluator results enter these decisions.

`scripts/finetune_video_evaluator.py` implements the ordered `pilot`, `capture`,
and `train` stages. `scripts/run_video_finetune.sh` records each process exit for
durable detached execution. The caller must explicitly set its allocated
`CUDA_VISIBLE_DEVICES`; the wrapper does not acquire an allocation. Pilot and
stage reports have distinct paths and never overwrite existing experiments.

The deterministic fine-tuning pilot now passed in 15.7 seconds with 6.37 GB peak
GPU allocation. Full versus cached mixed-precision descriptors and gradients
matched exactly. Original FP32 descriptors differed by at most 2.62e-6, and the
fixed-matrix spatial average differed from the old adaptive average by at most
2.86e-6, within the preexisting 1e-5 engineering tolerance. These tiny differences
are reported, not described as bitwise equivalence to the frozen baseline.

Two failed engineering attempts are retained under `results/videomae_attempts`.
The initial strict gradient comparison failed despite identical forward outputs;
repeating the full backward produced comparable differences. Enabling deterministic
algorithms then identified the overlapping adaptive-average-pool CUDA backward as
nondeterministic. Training now uses an explicit fixed averaging matrix with FP32
arithmetic and deterministic math attention. The unchanged frozen capture still
uses its original pooling implementation. Converted zero key biases are fixed,
matching the original checkpoint's parameterization. The final pilot verifies
both frozen-weight integrity and the selected trainable parameter update.

The two discarded pilot updates were numerical checks. Its first Adam update at
the originally proposed head learning rate0.001 increased the four-view loss from
0.497 to 0.933. **Before research optimization**, the parent approved one conservative
amendment: lower the warm-started head learning rate to 0.0001, retaining backbone
rate 0.00001, 500 updates, effective batch16, and the same three evaluation steps.
There is no subsequent learning-rate search. The verified prefix/pilot files are
preserved. The new training-only entrypoint is
`scripts/train_video_evaluator_partial.py`; its `--register` action freezes
`configs/videomae_partial_finetune_v1.json`, binding all code and evidence hashes.
Use `scripts/run_video_finetune_v1.sh` inside tmux for this registered training.
The older combined script's unrun `train` mode records the superseded 0.001 plan;
it is not the registered research optimization command.

## Completed adaptation and independent audit

All 336 prefix clips passed, including 1,344 real endpoint views and 352 selection
reconstructions. Capture took 584.7 seconds. The prospective training registration
SHA256 is `3ba4b9434040d9c9af1fcac3a8d05c4f6242ea0117c65fe3c118bd09d456db1d`.
The registered 500-update optimization completed in 130.9 seconds and chose update 500.
Every checkpoint is retained in the report: real normalized MSE was 0.703851 at 100,
0.704072 at 250, and0.698319 at 500. No hyperparameter was changed during training.

For the before/after comparison below, the unchanged ridge and selected MLP were
also evaluated on the **same input pipeline** as the adapted model. Deterministic
math attention produces slightly different codec pixels than the earlier fused
pipeline: source-image MSE changes averaged −0.00138, with maximum absolute change
0.0683 across 88 clips. The separate matched-input calibration avoids treating
those reconstructions as bitwise identical. It changes no fitted parameter and
performs no additional selection. Its unadapted MLP real score reproduces the
recorded initial adaptation score exactly.

| Predictor, using the same input pipeline | Real-video NMSE | Codec-reconstruction NMSE |
| --- | ---: | ---: |
| Frozen encoder + ridge | 0.796860 | 0.799375 |
| Frozen encoder + selected MLP | 0.719027 | 0.722231 |
| Adapted last two blocks + MLP | 0.698319 | 0.706246 |

Ball-relative XYZ position R² reaches 0.565/0.576/0.428, with MAE 766/1,167/269
Unreal units on real selection video. Relative velocity R² remains
0.032/−0.030/0.017, with MAE 632/876/377 Unreal units per second. These results show
modest measurement improvement, with substantial errors and an unresolved relative
velocity problem. The evaluator still cannot certify precise physical steering.

`results/videomae_finetune_v1_audit.json` has status
`passed_development_video_evaluator_audit`: registration predates optimization;
all 500 updates and the exact 100/250/500 checkpoint rule are verified; saved model
and prediction hashes match; discovery normalizers and original zero key biases
are unchanged; and frozen-prefix hashes agree before and after training. An
independent rejoin to source endpoint labels and NumPy metric calculation agrees
within 1.6e-12. This audit does not use either original or fresh confirmation rows.

The final fitted tail/head and predictions are in
`/data2/ishaangp/mira-interp/video_evaluator_v1/partial_finetune/fit_v1`.
`results/videomae_finetune_v1.json` is the complete adaptation report, and
`results/videomae_matched_input_calibration.json` records the unchanged-predictor
calibration. All original frozen baseline/MLP reports remain intact.

`results/videomae_repair_exports/target_metrics.csv` contains 72 model/domain/target
rows and `summary.csv` the three aggregate comparisons. The exact reported scores
are shown in `figures/videomae_repair_summary.{png,pdf}` and
`figures/videomae_repair_targets.{png,pdf}`, with export hashes in
`results/videomae_repair_exports/manifest.json`. All comparisons are development
annotation decoding with target pixels present. No generated-rollout intervention
accuracy, entity-tracking reliability, or physical causal claim is established.
