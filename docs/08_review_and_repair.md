# Literature-led review and repair

The user requested a substantive revision after the initial observational result.
The v2 experiment, its failures, frozen choices and confirmation remain intact.
Its derived-feature archive is published; GitHub's asset size and SHA256 match
the locally verified 1,138,648,642-byte archive (`results/observation_publication.json`).

This document preserves the repair rationale. The completed outcomes are in
[initial progress](progress/README.md) and [diagnosis and repairs](13_diagnosis_and_repairs.md):
the role-relative target convention helped substantially; the tested spatial
readout did not outperform pooling.

## What went wrong, and what is still a hypothesis

1. **The readout discarded spatial layout.** Averaging144 tokens per view makes
   the first2048-dimensional residual readout an affine function of at most64
   current/past latent means, plus noise and BOS. Distinct object layouts can have
   the same average. This is a proven information bottleneck; the effect on probe
   scores must be measured with the revised readout.
2. **The target convention may obscure relevant state.** Absolute player slots
   vary with sorted player IDs, while each visual view follows one car. The cited
   Othello paper's Mine/Yours repair motivates ego-car and ball-minus-ego targets.
   These remain world-axis coordinates; camera-heading coordinates require
   additional orientation metadata and are not silently substituted.
3. **Numerical parity was incomplete.** The released checkpoint contains FP32
   tensors and uses BF16 autocast. The pilot cast all parameters/buffers to BF16,
   mixed latents/noise in BF16, and rounded resized video to uint8. New capture
   retains FP32 parameters/buffers, mixes with FP32 tau and retains fractional
   resized pixels. A reserved numerical comparison measures the effect.
4. **The baseline was weak.** A32-dimensional codec mean discarded the same
   object layout. New development includes full9x16x32 codec tokens and a1536-D
   RGB grid. Their tuning budget includes lower ridge penalties. Ridge arithmetic
   in v2 was correct; its residual minima were interior, so overregularization
   has not been shown to explain the original result.
5. **The causal stopping gate was too strong.** Exact interventions inside the
   model can fix the recipient's actions and noise even when donor recordings
   have different actions. Such experiments establish conditional internal causal
   effects. They do not isolate a single simulator variable or automatically
   establish generated physical control. Those stronger claims need additional
   measurements and controls.

Remaining hypotheses include short temporal context, single-tau measurement,
visibility, camera geometry, model training quality and cross-view state latency.
They are not diagnosed causes. The publisher uses80-frame training windows;
the current controlled readout comparison retains16 frames to avoid conflating
every change. It scores latent steps2–7 and then tests within-view temporal differences.

The [MIRA Mini technical report](https://alakazam.gg/mira-mini/report.pdf), section17.1,
also reports weak velocity decoding using ridge and a three-layer MLP on its
single-player teacher/student comparison. That experiment differs from our
multiplayer study, so its scores cannot be directly substituted. It supports
testing model/readout limitations instead of assuming a simple code failure.

## Sources and coverage

- [Probes, attribution and exact patching](review/probes_and_causality.md): the
  three linked primary papers/articles, including the original probe code.
- [Geometry and sparse features](review/geometry_and_features.md): both linked
  geometry articles, the BSF paper and pinned official implementation differences.
- [Anthropic and model audit](review/anthropic_and_model_audit.md):56 dated
  Transformer Circuits links inventoried; the initial review inspected16 sources.
  The later [first20](review/anthropic_extended_review_a.md) and
  [last20](review/anthropic_extended_review_b.md) extensions cover the remaining
  pages at their declared scope. The [coverage check](../results/review/anthropic_coverage_complete.json)
  verifies all56 URLs. This includes editorial and landing pages as such, and
  does not claim exhaustive reading of appendices, references, interactives,
  videos, or all Anthropic publications.
- [Rocket Science](https://huggingface.co/datasets/kyutai/rocket-science), the
  [MIRA source](https://github.com/mira-wm/mira),
  [MIRA Mini release](https://huggingface.co/alakazamworld/mira-mini-4p) and
  [publisher report page](https://alakazam.gg/mira-mini) were checked. This is an
  independent reproduction, and the publisher describes limited training budget
  and still-maturing early-window action controllability.

## Ordered repair experiment

The development design is recorded in `configs/development_v3.json` before new
research capture/fitting. Only the original31 discovery and11 selection matches
are used. The earlier11 confirmation matches are excluded from this experiment.

1. Re-decode the same336 development windows from source video with fractional
   FP32 resize. Require every source hash and PTS to match, and require rounding
   the new pixels to reproduce the previous pixels exactly. Keep labels/actions
   and their original alignment unchanged. The reserved pilot passed these checks.
2. Verify the corrected GPU capture on the reserved pilot, including exact no-op
   and repeat output, before the complete development capture. The first pilot
   rejected FP32 interpolation overshoot255.000015; preserve that failure and
   allow only a1e-3 numerical tolerance without clipping the input.
3. Capture all17 residual sites as both means and3x4 spatial bins with a fixed,
   data-independent128-channel orthonormal projection (1536 dimensions). Preserve
   stronger codec and RGB descriptors. Audit all files and exact source-label joins.
4. Compare absolute and role-aware targets, all-layer mean versus spatial probes,
   the stronger baselines and whole-match label controls. Discovery-only fits,
   equal-match weights and complete selection curves remain required. A separate
   development pass tests current-plus-previous-difference features at stage-one
   winners. These selection metrics are development evidence, not confirmation.
5. Proceed to narrowly labeled attribution and exact interventions with fixed
   recipient actions/noise, followed by geometry and sparse-feature development
   whose claims match the measured causal effects. A proxy readout cannot certify
   generated-video physical control by itself.
6. Freeze revised choices before evaluating new confirmation. The24 candidate
   matches in `data/fresh_confirmation_split_manifest.json` were chosen by seeded
   metadata-only hashing from the official dev split, disjoint from all old test
   matches. Apply the same source-quality gates, report exclusions and use no
   replacements. No model-performance values were consulted in this reservation.

## Runtime and artifacts

Only authorized physical GPUs6/7 are used. `NUMPY_MADVISE_HUGEPAGE=0` avoids the
host memory-compaction stall observed during v2 without changing mathematical
operations. Inputs/captures remain under `/data2/ishaangp/mira-interp`; code,
protocols, reports and plots are published in Git, with large derived artifacts
in releases. Source videos, raw labels and pretrained weights are not republished.

The revised study is **in progress**. This document records fixes and tests; it
does not promise a positive scientific outcome or mark the proposal complete.
