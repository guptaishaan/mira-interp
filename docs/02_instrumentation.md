# Step 1 prerequisite: residual capture and intervention wiring

**Status: engineering checks passed on CPU and GPU. Physical representation identification has not run.**

The integration uses the actual upstream `DiffusionTransformer` and `AdaSTBlock`
implementations, at commit `3d739ec2d31daf83559d33eb01727cea48fe90f7`, with a tiny
randomly initialized configuration. It loads no pretrained weights, codec, action
encoder, clips, or physics labels. Its random latent perturbation is an engineering
input difference, not a controlled physical-state change. Its objective is a mean
of predicted latent flow, not a physical-state measurement.

## What is implemented

- [`instrumentation.py`](../src/mira_interp/instrumentation.py) discovers every
  residual block in a bare transformer, single-player model, or multiplayer wrapper.
  It captures the first block's input separately and every block's output.
- A single-use `ResidualTrace` context attaches hooks for one transformer call and
  removes them on successful exit or exception. Missing layers, repeated calls,
  unknown site indices, incompatible shapes, and nonfinite donors/attribution
  inputs fail explicitly.
- Exact replacement, zero ablation, and restoration can affect a whole site or an
  explicit boolean mask. Masks can select whole tokens or individual coordinates.
  Patches apply before the output is captured; donors are detached and input tensors
  are not mutated. Multiple patches at the same site execute in listed order.
- Attribution computes `sum((clean - corrupted) * gradient_at_corrupted)` separately
  for each batch sample. It estimates the change in the specified scalar objective.
  If the objective is an error, a negative score means lower estimated error.
- A random-direction utility matches the L2 norm of each sample's intervention
  **delta**, with an explicit random generator. It does not select experimental
  controls or claim they have been evaluated.

## Where the tensors are

MIRA's transformer receives codec latents as `[B, T, H, W, C_latent]`. A latent
patch size `P` first rearranges each `P x P` spatial patch into channels; a linear
projection then maps it to hidden width `D`. Optional clean-past projection adds to
this latent stream. Each residual site has `[B, T_with_registers, H/P, W/P, D]`.
Spatial cells remain a two-dimensional grid; they are not an entity-token sequence.

Each complete `AdaSTBlock` runs spatial attention, optional temporal attention, and
an MLP, each with residual addition. It returns **`(residual, temporal_kv_cache)`**.
The hook changes only tuple item zero and preserves the other item.

| Manifest entry | Meaning |
| --- | --- |
| `input_site` | Input to block 0, after projection and optional register insertion |
| `layer_index: 0` | Output after the first complete residual block |
| `layer_index: n-1` | Output after the last complete residual block |
| `temporal_attention` | Whether that block has a temporal-attention sublayer |
| `register_slots_in_this_call` | Leading time slots to exclude when mapping to physical frames |
| `latent_frame_indices` | Physical latent-frame indices after those register slots |

When register tokens are enabled and no KV cache is supplied, upstream prepends
them on the **time axis**, repeating their embeddings across the spatial grid.
They appear in every capture but are removed before the output head. For example,
our two-frame smoke with one register has residual shape `[1, 3, 2, 2, 64]`.
Residual time index 1 refers to the first latent frame; residual time index 0 is
the register. With an existing KV cache, upstream does not insert register tokens.

The multiplayer wrapper stacks per-player views vertically on **height**:
`[(B * players), T, H, W, C] -> [B, T, players * H, W, C]`. Thus a player-view mask
selects a height region. It is not necessarily a mask for that player's physical
entity: attention mixes views and each view can show multiple entities.

## Noise and time identity

Each trace requires sample IDs, condition name, random seed, noise identity,
zero-based rollout and diffusion-step indices, all `tau` values and their shape,
latent-frame indices, and whether this is a fresh or cached call. The trace checks
batch/time dimensions against the metadata and configured register count. The
caller must pass the actual call's `tau` and input identities; the smoke constructs
these directly from its tensors and records SHA256 hashes of the inputs/actions.

Upstream flow matching uses `z_tau = tau * z_clean + (1 - tau) * noise`: `tau=0`
is noise and `tau=1` is clean. Model output is **latent flow-matching velocity**,
not a ball or player velocity label. Physical velocity would require a separately
validated readout. Denoising, context-cache building, and cache-update calls must
receive distinct trace identities, even when their frame indices overlap.

## Verification and results

Nine focused unit tests pass. They check layer numbering, block input capture,
tuple/KV preservation, no-op equality, exact mask locality and downstream effects,
zero ablation/restoration, cleanup after exceptions, invalid metadata/masks,
register offsets, finite-value rejection, per-sample random norm matching, and
agreement between attribution and an independently evaluated exact linear effect.

The real-upstream integration has three blocks, hidden width 64, grouped-query
attention, adaptive attention LayerNorm, attention gating, spatial patch size 2,
one register slot, and 321,280 randomly initialized parameters. Temporal attention
is present in blocks 0 and 2, so the optional temporal branch is also exercised.

| Check | CPU | GPU 6, NVIDIA A40 |
| --- | --- | --- |
| All residual outputs captured | 3/3 | 3/3 |
| Capture/no-op max absolute error | 0 | 0 |
| Donor no-op max absolute error | 0 | 0 |
| Restoration max absolute error | 0 | 0 |
| Masked ablation locality | Pass | Pass |
| Gradients retained at every block | Pass | Pass |
| Final linear-head attribution vs exact error | 2.84e-10 | 5.82e-11 |
| All hooks removed | Pass | Pass |

The GPU allocator peak was 20,782,080 bytes. This is the tiny smoke's allocation,
not a memory estimate for a pretrained MIRA model. The GPU process exited after
the test, and GPU 6 returned to idle. Machine-readable results, input hashes,
configuration, timestamps, and site manifests are in
[`instrumentation_smoke_cpu.json`](../results/instrumentation_smoke_cpu.json) and
[`instrumentation_smoke_gpu6.json`](../results/instrumentation_smoke_gpu6.json).

Reproduce from the repository root using the prepared environment:

```bash
PYTHONPATH=src:external/mira/src .venv/bin/python -m pytest tests/test_instrumentation.py -q
PYTHONPATH=src:external/mira/src .venv/bin/python scripts/smoke_mira.py --device cpu --output results/instrumentation_smoke_cpu.json
```

After checking GPU 6 still belongs to this task and has no foreign process:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:external/mira/src .venv/bin/python scripts/smoke_mira.py --device cuda:0 --output results/instrumentation_smoke_gpu6.json
```

## Limits and next gate

Run hooks in eager mode with activation checkpointing disabled and the model in
evaluation mode. Gradient attribution requires an enabled graph; with model
parameters frozen, set the latent input to require gradients. Detached CPU capture
is the default; graph capture retains all residuals on their device and can be
expensive on a full model. A separate trace is required for each forward call.

The smoke verifies returned KV tuples in a fresh forward. **It does not validate
cached generation, multi-frame rollouts, or persistent cache interventions.** A
block-output intervention leaves that same block's already-computed KV tuple
unchanged. Later blocks can receive a changed residual and therefore compute
changed caches. A future rollout experiment must explicitly decide and validate
which cache state is part of the intervention.

The exact attribution comparison uses the final linear output head. It verifies
sign, indexing, gradient retention, and arithmetic; it provides no guarantee that
first-order estimates will agree with finite interventions through nonlinear
upstream blocks. Ablation followed by restoration recovers the original tensor
by construction; this is a wiring control, not causal evidence about physics.

Before physical experiments, the project still needs accessible trained weights
with provenance, synchronized match-separated clips and physical labels, validated
encoding/action alignment, and a reliable physical prediction objective. Then
all-layer capture can begin and candidate rankings can be checked with exact
patches on separate data. Wrong-player and wrong-variable controls, trajectory
measurement, geometry, SAE/BSF recovery, and steering remain gated.

Source interfaces audited:
[diffusion transformer](https://github.com/mira-wm/mira/blob/3d739ec2d31daf83559d33eb01727cea48fe90f7/src/mira/world_model/diffusion_transformer.py),
[residual block](https://github.com/mira-wm/mira/blob/3d739ec2d31daf83559d33eb01727cea48fe90f7/src/mira/world_model/layers/transformer.py),
[multiplayer wrapper](https://github.com/mira-wm/mira/blob/3d739ec2d31daf83559d33eb01727cea48fe90f7/src/mira/world_model/multi_wrapper_world_model.py), and
[flow-matching input preparation](https://github.com/mira-wm/mira/blob/3d739ec2d31daf83559d33eb01727cea48fe90f7/src/mira/world_model/latent_world_model.py).
