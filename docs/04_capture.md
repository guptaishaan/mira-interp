# Actual-observation activation capture

This step implements the registered **observational reconstruction-decoding**
study. It does not establish a causal physical feature or predict a future state.
The frozen protocol is
[`configs/observational_probe_v2.json`](../configs/observational_probe_v2.json),
SHA-256 `e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2`.
The original 61-match data cohort failed its clock-quality gate. Before any
research capture, v2 registered all 53 qualifying original matches, preserving
their roles and every threshold. The original failed audit is retained.

**The v2 reserved actual-data pilot passed on 2026-09-07 UTC.** One 16-frame clip
produced 32 correctly aligned rows at all 17 sites. The run took 43.7 seconds,
including pinned checkpoint verification/loading; capture and full-tensor
reference checks took 3.15 seconds. Peak allocated GPU memory was 6.51 GiB on A40.
No-op and repeated forwards matched bitwise. Independent CPU means of every
full residual matched GPU pooling, with maximum absolute difference 3.82e-6.
The compact NPZ is 2.09 MB; full pilot references occupy 320.9 MB.
See [`results/capture_pilot_v2.json`](../results/capture_pilot_v2.json) and the separate
[exact row-join audit](../results/capture_pilot_v2_row_audit.json). This is an
engineering gate only; the reserved pilot is not part of probe fitting/evaluation.
The original v1 pilot and its outputs remain preserved separately.

**Research capture and independent aggregation passed on 2026-09-07 UTC.**
Two A40 workers captured all 424 registered clips (13,568 repeated rows) in
279.8 seconds wall time, each retaining one loaded model and peaking at 6.51 GiB.
Both completed all 212 assigned clips. The independent CPU audit subsequently
verified every source-label join, saved feature file and provenance chain in
121.3 seconds. See the [aggregate report](../results/capture_aggregate_v2.json).
This completes activation capture, not probe fitting or causal validation.

## Forward pass and target alignment

1. Read one audited actual clip: four camera views, 16 source frames at 20 FPS,
   resolution 288x512, and all four keyboard streams. The model receives no
   physical labels, timestamps, match IDs, or split labels.
2. The strictly loaded trained codec yields eight latents per view, shape
   `[4,8,9,16,32]`. Stack the four view grids vertically for MIRA Mini's joint
   transformer. Embed actions using the trained per-player projection and mean.
3. At `tau=0.5`, combine observed clean latents and fixed per-clip seeded noise.
   Supply teacher-forced clean past. Run one eager transformer forward, recording
   the input to block 0 and the output of every one of the 16 blocks.
4. Average spatial positions **within each view tile**, accumulating in float32.
   Save float16 features as `[32,17,2048]`, with rows ordered by view first and
   then latent time. No view concatenation or cross-view pooling is applied.
5. Only after the model computation, join physical labels from each view's own
   final source frame in the codec pair: frame indices 1, 3, ..., 15. Preserve each
   view's timestamp and source-frame index. Four views can have recording-clock
   offsets; one view's state labels are never broadcast to the other views.

The targets are 30 world-coordinate components: ball and four identity-mapped
players, each with XYZ location and velocity. Because noisy observed targets and
clean observed past enter the model, the task is reconstruction decodability.
Views/frames are repeated measurements; the independent evaluation unit is match.

## Outputs

Each clip gets a compressed NPZ and a JSON provenance record. NPZ arrays include:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `[32,17,2048]` | Per-view pooled input and 16 residual outputs, float16 |
| `y` | `[32,30]` | Own-view pair-final physical labels, original units |
| `codec_X` | `[32,32]` | Clean normalized codec spatial means |
| `RGB_X` | `[32,192]` | Pair-final RGB frame pooled to 8x8, range 0 to 1 |
| `match_ids`, `split`, `clip_ids` | `[32]` | Row identity and frozen role |
| `view_index`, `latent_frame_index` | `[32]` | Exact row mapping |
| `source_frame_index`, `timestamps` | `[32]` | Own-view physical-label timing |
| `target_names`, `sites` | `[30]`, `[17]` | Ordered feature/target names |

The JSON records input/output hashes, protocol and capture/helper-code hashes, seed,
noise hash, pooling quantization error, runtime, and pilot controls. Workers retain
one loaded model. Existing outputs are resumed only after hash, shape, finite-value,
row-order, input-provenance, script, and protocol checks.

## Ordered gates

The runner refuses data without a passed preparation audit tied to the exact
clip and split manifest hashes. It recomputes the registered cohort digest and
checks complete per-match clip counts and frozen player/role assignments.
A single reserved
pilot clip then checks actual 16-frame runtime/memory, bitwise no-op/repeat
forwards, and pooled features against independent per-view means of all 17 full
residual tensors (`atol=rtol=1e-5`). The pilot retains those full tensors and
their hashes for inspection. It is excluded from all scientific fit/evaluation.

Research workers additionally require a passed pilot from the same capture
script and frozen protocol. They process deterministic disjoint modulo assignments
of the sorted clip manifest. The v2 cohort is 31 discovery, 11 selection,
and 11 confirmation matches, eight audited clips per match. Any missing or
insufficient match blocks the complete registered cohort.

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/capture_observations.py \
  --assets /data2/ishaangp/mira-interp/assets \
  --manifest data/qualified_pilot_clip_manifest.json \
  --data-audit results/qualified_pilot_data_audit.json \
  --split-manifest data/qualified_split_manifest.json \
  --protocol configs/observational_probe_v2.json \
  --output-root /data2/ishaangp/mira-interp/captures/observation_pilot_v2 \
  --report results/capture_pilot_v2.json --pilot
```

Use the actual passed data-audit path produced by preparation. Full capture is
authorized only after review of that pilot. For the two research workers,
provide the completed research data audit and `--pilot-report
results/capture_pilot_v2.json`, use `--workers 2` with separate `--worker-index 0`
and `--worker-index 1`,
and separate reports on the two allocated GPUs. The shared output directory is
safe because each worker owns different clip IDs.
After both workers pass, the independent [aggregate audit](05_aggregate.md)
checks every source-label join and provenance chain before any probe analysis.

Retain completed worker reports and validate them with the aggregate audit;
do not rerun a completed worker into the same report path. The current resume
path loads and records model provenance only when at least one clip remains to
capture. An entirely cached rerun would replace the report without that proof
and be rejected by aggregation. Resume an unfinished worker only when it still
has pending clips; use new output/report paths for a new complete run.

The reusable strict loader is
[`src/mira_interp/model_loading.py`](../src/mira_interp/model_loading.py). It
revalidates published checkpoint hashes, exact original YAML hashes, clean pinned
MIRA/DINO source, and every state-dict key/shape. It uses the previously validated
Alakazam action-combiner adapter; no parameter remains at random initialization.
The world-model checkpoint also embeds all 936 codec tensors. The loader checks
that every embedded tensor is exactly equal to the standalone codec before
overwriting its weights, so the retained standalone latent-normalization metadata
belongs to the same encoder. The independent CPU audit is
[`results/codec_pair_audit.json`](../results/codec_pair_audit.json).
