# Pretrained inference readiness

This step tested the independently trained **Alakazam MIRA Mini 4P 1B** checkpoint.
The official MIRA checkpoint was not located; see
[the access audit](model_access_audit.md). This is a readiness test with one
public example, not the proposal's physical-state identification experiment.

**Passed on 2026-09-06:** strict trained loading, all 16 residual layers plus the
block-zero input, bitwise identical no-op/repeat controls, and one finite sampled
future latent. The final run took 44.6 seconds and used about 4.6 GiB peak GPU
memory. Zero and all255 future placeholders produced bitwise identical generated
latents and video. All saved artifact shapes and hashes were read back and
verified. This validates executable
access to this checkpoint, not physical accuracy or steering.

## Inputs and checks

The public model bundle contains video `[4,80,3,288,512]` and keyboard actions
`[4,80,9]`. It contains no physical labels or match IDs. The smoke test uses four
frames per view: two observed context frames and two future generated frames
(one future latent, 0.10 seconds). It keeps the native spatial resolution.

Before inference, the script verifies the published SHA-256 hashes for both
checkpoints and the example context. It reads checkpoints with
`torch.load(weights_only=True, mmap=True)`, validates exact model keys/shapes,
and calls the base PyTorch strict loader. This deliberately bypasses MIRA's
multiplayer warm-start loader, whose documented purpose permits selected
parameters to retain random initialization. Every parameter in this test must
come from the pretrained bundle.

The first loading attempt found one architectural mismatch: the current official
MIRA wrapper concatenates four players before projecting, whereas Alakazam's
published runtime projects each player and then averages. The checkpoint's
projection matrix confirms the latter dimensions. The small adapter in
[`src/mira_interp/pretrained.py`](../src/mira_interp/pretrained.py) reproduces
those two runtime changes exactly. The original failure is retained in
[`results/pretrained_attempts/01_official_architecture_incompatible.json`](../results/pretrained_attempts/01_official_architecture_incompatible.json).
The independent publisher's `alakazam-mira-mini==0.1.11` wheel was inspected as
source, SHA-256 `0acea9fc285c44537785e2987a42181695500317371e6cc652cefcd38e982cfe`.
The other differences in its MIRA code are distillation training and permissive
checkpoint loading; they are not needed for this non-distilled strict inference.

The script captures all 16 residual layers in one fresh transformer call at
flow-matching time `tau=0.5` with a fixed noise seed. This capture uses noisy
observed targets and teacher-forced clean past. It then uses the upstream
streaming sampler for an eight-step generation with observed actions. Future
video frames are zero placeholders in the primary sampler input, not observed
targets. A second sample with the same actions and seed changes those future
placeholders to all255; both generated latents and frames must match bitwise.
This directly checks future-placeholder invariance for this bounded case. These are
separate checks: the fixed-time capture is not an attribution experiment and
the short generated video is not a physical accuracy measurement.
An unhooked forward, the capture-only forward, and a repeated unhooked forward
must produce bitwise identical predictions. The input to residual block zero is
saved alongside the 16 block outputs.

## Reproduce

Install the project environment and run the documented model download first.
The DINO dependency is source only; trained DINO parameters come from the
verified codec checkpoint. Pin its checkout before running:

```bash
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
git -C external/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:external/mira/src \
  .venv/bin/python scripts/smoke_pretrained.py \
  --assets /data2/ishaangp/mira-interp/assets \
  --activation-dir /data2/ishaangp/mira-interp/captures/pretrained_smoke \
  --device cuda:0
```

GPU 7 is the physical GPU authorized and checked for this run; choose your own
allocated GPU when reproducing. The process sees only that GPU as logical
`cuda:0`. `--load-only` verifies strict pretrained loading on CPU without running
inference. Hash validation is repeated rather than trusting a file's existence.

## Outputs and interpretation

- [`results/pretrained_smoke.json`](../results/pretrained_smoke.json) records
  current stage, success/failure, exact pins, input hashes, runtime overrides,
  loading checks, capture statistics, and GPU memory.
- `results/pretrained_layer_stats.csv` contains all-layer finite-value and
  magnitude checks after a successful capture.
- `figures/pretrained_smoke.png` compares the last observed context frame and
  the generated final frame for all four views. A short context can yield poor
  trajectories; this figure is not a validated physical prediction result.
- Full residual tensors are saved to the explicit local capture directory,
  with SHA-256 hashes and paths in the result JSON. They are reloadable using
  `torch.load(path, weights_only=True)`.
- `generated_frames.npz` in the same capture directory saves both generated
  frames, uint8, with axes `[player_view,time,channel,height,width]`. Its timestamps
  are 0.05 and 0.10 seconds after the last context frame; `source_frame_indices`
  identifies the corresponding requested time positions in the input window.
  `generated_latents.pt` preserves the model's normalized generated latent for
  each player before decoding, at the original inference dtype. Both are hashed
  in the JSON report.

A completed readiness test requires exact loading, all 16 finite residual
captures, bitwise no-op/repeat and future-placeholder controls, and finite
generated latents/video. Source-disjoint physical probes,
attribution/activation patching, feature learning, and steering remain gated on
the proposal's missing data and validation prerequisites.

## Provenance and licenses

| Component | Pin and source |
| --- | --- |
| MIRA architecture | [`mira-wm/mira`](https://github.com/mira-wm/mira), `3d739ec2d31daf83559d33eb01727cea48fe90f7`, Apache-2.0 |
| Pretrained reproduction | [`alakazamworld/mira-mini-4p`](https://huggingface.co/alakazamworld/mira-mini-4p), `d58ee2f9bca27289554c1e652943dc0e539e8971`, model card specifies CC BY-NC-SA 4.0 |
| DINOv3 architecture source | [`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751), `6876159a11b4df116f30f667f8c9888617df0751`, DINOv3 License Agreement |
| Public player inspected | [`Alakazam-studios/alakazam-mira-mini`](https://github.com/Alakazam-studios/alakazam-mira-mini/tree/617a585852445f3581f6b88aaf27ec4e745aa6b4), `617a585852445f3581f6b88aaf27ec4e745aa6b4`; not needed to execute this script |

The script loads pinned local DINO source rather than an unpinned network hub
dependency. Runtime changes are limited to the local codec path, eager execution,
disabled checkpoint recomputation, bfloat16 compute, and the documented shorter
inference context. Original publisher YAML files remain unchanged and their
hashes are recorded. Rocket League content is copyright Psyonix / Epic Games;
the figure retains attribution to that content and the independent Alakazam
release, under CC BY-NC-SA 4.0. No endorsement is implied.

Machine-readable source and wheel provenance is in
[`results/pretrained_source_provenance.json`](../results/pretrained_source_provenance.json).
The source check also rejects tracked modifications and untracked Python/shared
library source files before loading. The JSON hashes the experiment script,
adapter, and residual instrumentation used for that run.
