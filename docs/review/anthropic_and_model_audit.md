# Anthropic literature and released-model audit

Checked 2026-09-07 UTC, after the original observational run. No prior capture,
protocol or result was changed for this review.

The review found **three concrete numerical/preprocessing differences from the
released implementation, but no demonstrated explanation of weak probe scores**.
The more substantial scientific limitation is that a spatial mean of visual
tokens is being asked to recover absolute world coordinates. A new diagnostic
should preserve spatial information and establish representation/readout
adequacy before testing sparse features or causal steering.

## Coverage

The [Transformer Circuits index](https://transformer-circuits.pub/) and
[Anthropic research index](https://www.anthropic.com/research) were inspected.
The former supplied 56 dated article/note links. Relevant methods, validation,
and limitations sections of 16 sources were inspected; 40 links were inventoried
only. This is **not an exhaustive reading of all Anthropic publications**,
appendices, cited references, or interactive examples. Exact URLs, coverage and
download hashes are in [the source inventory](../../results/review/anthropic_sources.json).
Large pages that exceeded the web renderer's size limit were downloaded from
the same primary URLs and parsed locally. Raw article caches remain external.

## What the primary literature changes about this project

| Primary source | Relevant evidence | Consequence for MIRA |
| --- | --- | --- |
| [Towards Monosemanticity (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | Dictionary features were assessed with activation examples, reconstruction, sparsity and their contribution to model behavior. Width changes can split features. | Sparse reconstruction quality and attractive examples are insufficient alone. Compare feature activity, reconstruction and downstream behavior across capacities. |
| [Scaling Monosemanticity (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | Feature steering and ablation supplied behavioral evidence; activation magnitude was a weaker ablation proxy than attribution in the examples examined. Missing features, shrinkage and feature splitting remained limitations. | Rank candidate physical sites using an output-relevant objective, then run exact interventions. Do not treat a high-activation SAE feature as a causal physical coordinate. |
| [Features as Classifiers (2024)](https://transformer-circuits.pub/2024/features-as-classifiers/index.html) | Raw activations were a strong baseline. Pooling, domain data and consistent input formatting materially affected results. SAE residuals retained predictive information, and feature visualizations exposed spurious correlations. | First establish a strong raw-token readout. Compare different spatial aggregations; inspect top examples and residual-only prediction before blaming the model or crediting an SAE. |
| [June 2024 update](https://transformer-circuits.pub/2024/june-update/index.html) | The preliminary comparison reproduced improved sparsity/reconstruction tradeoffs for Top-K and gated SAEs and separately examined interpretability. | Matching only dictionary width is inadequate. Compare active features, parameters, reconstruction and task effects; improved MSE does not automatically mean improved interpretation. |
| [July 2024 update](https://transformer-circuits.pub/2024/july-update/index.html) | The notes distinguish missing features, cross-layer superposition, attention and interference-weight difficulties, and discuss linear versus multidimensional representations. | Failure of one linear pooled probe does not establish absence of information. Conversely, a curved PCA plot does not establish a useful causal manifold. |
| [Sparse Crosscoders (2024)](https://transformer-circuits.pub/2024/crosscoders/index.html) | A preliminary shared dictionary reads/writes multiple layers and can track features persisting through the residual stream. | Cross-layer consistency is a useful comparator after raw-state decoding works. This does not itself establish temporal entity identity or validate a temporal block-sparse extension. |
| [Circuit Tracing: methods (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Cross-layer transcoders construct interpretable replacement computations. Graphs include error nodes; original-model perturbations test faithfulness. Fixed attention and normalization constrain what early graphs explain. | Attribution is a hypothesis generator. Measure reconstruction and actual output preservation, retain unexplained residuals, and validate proposed effects in the original video model with attention free to respond. |
| [On the Biology of a Language Model (2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Feature labels and groups are chosen before perturbation measurements. The paper distinguishes effects constrained by patch construction from downstream effects that independently test graph predictions. | Freeze entity/variable/site interpretations before confirmation. A reconstruction or constrained patch check cannot serve as its own independent causal confirmation. |
| [When Models Manipulate Manifolds (2025)](https://transformer-circuits.pub/2025/linebreaks/index.html) | Character-count representations were studied through feature families, supervised multiclass probes and low-dimensional geometry. Subspace ablation used random-subspace controls; mean-replacement interventions changed linebreaking behavior. | Compare scalar regression with value-conditioned or binned readouts on held-out values. Infer geometry from matched physical sweeps, then test behavior with ablation and replacement; PCA alone is descriptive. |
| [A Toy Model of Mechanistic (Un)Faithfulness (2025)](https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html) | A low-MSE replacement can implement a different computation, including datapoint-specific solutions. The note separates mechanistic faithfulness from simplicity of description. | Evaluate an SAE/BSF by preservation of original causal effects on unseen examples, not just reconstruction or smooth plots. Dense geometric structure approximated by many sparse pieces is not automatically an error. |
| [Attention through feature interactions (2025)](https://transformer-circuits.pub/2025/attention-qk/index.html) | QK attributions explain attention scores through feature interactions, addressing what earlier fixed-attention graphs omitted. | Video entities move across spatial and temporal tokens. Preserve token coordinates and inspect routing before assuming global pooled residuals are the right units. |
| [Toy interference weights (2025)](https://transformer-circuits.pub/2025/interference-weights/index.html) | Toy examples distinguish meaningful functional pathways from interactions introduced by overlapping representations. | Cosine similarity and decoder-weight connectivity are leads, not adequate evidence of a physical circuit. |
| [Downstream connections and steering, May 2026](https://transformer-circuits.pub/2026/may-update/index.html) | Features with similar activating examples and output associations can have different intervention effects; downstream connectivity helps distinguish them. | Use matched downstream behavior to choose among apparently similar velocity or position features. Semantic labeling alone does not predict steerability. |
| [Turn-averaged SAEs, June 2026](https://transformer-circuits.pub/2026/june-update/index.html) | Averaging reduces the number of feature activations to inspect and emphasizes turn-level characteristics. This is preliminary work on broad transcript properties. | Averaging can be appropriate for global traits, but this result does not justify discarding location when the target is spatial position or motion. |
| [Global workspace, July 2026](https://transformer-circuits.pub/2026/workspace/index.html) | An output-linked Jacobian readout and interventions characterize a restricted family of representations; the authors discuss what their lens cannot name. | Every readout selects a view of the representation. An output-gradient physical lens is a possible later method, but a failed lens cannot prove that the video model lacks state. |
| [Interference effectiveness/helpfulness, August 2026](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html) | Measuring output effects and loss changes distinguishes weight magnitude from functional importance in a tiny model; aggressive filtering does not yield a complete sparse explanation. | Check actual trajectory-objective changes rather than ranking candidate interventions only by norm or parameter magnitude. Tiny-language-model findings are methodological guidance, not evidence about MIRA. |

These applications to MIRA are research judgments, not results established by
the language-model papers. In particular, no source above promises that an SAE,
larger probe, or nonlinear fit will repair this dataset/model/readout combination.

## Released MIRA Mini implementation audit

References are the [published 0.1.11 wheel](https://files.pythonhosted.org/packages/94/eb/48891a6baaba6511e12f98ea00f64967e9e7a9e83bd16f6468c8028ef17d/alakazam_mira_mini-0.1.11-py3-none-any.whl)
(SHA-256 `0acea9fc285c44537785e2987a42181695500317371e6cc652cefcd38e982cfe`),
[pinned model/config](https://huggingface.co/alakazamworld/mira-mini-4p/tree/d58ee2f9bca27289554c1e652943dc0e539e8971),
and [pinned upstream source](https://github.com/mira-wm/mira/tree/3d739ec2d31daf83559d33eb01727cea48fe90f7).
The [file comparison](../../results/review/model_source_comparison.json) found
51 of 54 extracted MIRA Python files byte-identical. Only multiplayer action
combination and unused distillation/permissive-loading additions differ in the
other three files. Both checkpoint state dictionaries contain only FP32 tensors.

| Component | Audit result |
| --- | --- |
| Architecture and pretrained state | All 1,183 WM and 936 codec tensors loaded strictly. Embedded codec weights equal the standalone codec exactly; normalization metadata is paired correctly. Prior evidence remains in the loading and codec-pair reports. |
| Four-view layout | Released code requires contiguous player-ID-ordered views and tiles height. Preparation/capture follow that ordering and retain own-view targets. No channel/view transposition found. |
| Multiplayer action adapter | Released combination adds player embeddings, projects each player independently and averages. The local adapter implements exactly those operations. |
| Keyboard and timing | Nine key columns match the published YAML. Source/target actions are 20 FPS; codec stride 2 makes action pooling stride 2. The local offset 1 and 14-action slice for 8 latents match the published rule. Unknown mouse sensitivity remains NaN for the learned missing-mouse token. |
| RGB normalization | Local code performs uint8/255, codec [-1,1], encoder return to [0,1], then DINO ImageNet normalization. This matches the released chain; there is no double normalization bug. |
| Codec and temporal alignment | The non-overlapping stride-2 bottleneck consumes each consecutive frame pair. Labels deliberately use its second frame. This is a declared reconstruction target, not future prediction. |
| Past and denoising conditioning | Clean past is BOS followed by preceding clean latents, matching training. Tau=0.5 is inside training support but is only one denoising condition. |
| Model precision | **Actual deviation:** capture converts the entire FP32 checkpoint model to BF16. Published `mira_vm/engine.py` leaves checkpoint/default parameter precision intact and uses BF16 autocast. Non-autocast operations and buffers therefore differ. A CPU calculation found a maximum 0.00811 DINO normalized-channel discrepancy from casting its mean/std buffers alone. This is not a measured downstream effect. |
| Flow interpolation precision | **Actual deviation:** local `0.5*z + 0.5*noise` stays BF16, whereas the released `tau*z + (1-tau)*noise` with FP32 tau produces FP32 interpolants. Numerically equivalent real-valued formulas need not be floating-point equivalent. |
| Resize precision | **Actual deviation:** preparation uses the released antialiased bilinear resize but rounds the resized pixels back to uint8. Published training keeps floating resized RGB. The rounding discrepancy is bounded by 0.5/255 per channel before DINO normalization, apart from floating arithmetic. It may matter for tiny objects; its actual downstream effect is unmeasured. |
| Temporal history | Published training windows have 80 frames; the released inference setting uses 78 context frames. Our 16-frame capture has eight latent steps and shorter causal history. Training uses clean previous latents, so 78 is not a mandatory training API length. Short history and the larger fraction of early/BOS observations require a controlled comparison. |

The local preprocessing function is `src/mira_interp/clips.py:decode_video`;
the numerical deviations are in `scripts/capture_observations.py:capture_features`
and the final model `.to(..., dtype=torch.bfloat16)`. No frozen code was edited.
No normalization/adapter bug was found that establishes the original probes are
invalid; the identified deviations motivate a new, separately recorded pilot.

## Reserved numerical pilot: completed

[The audit](../../results/review/numerics_pilot.json) passed in 74.64 seconds on
GPU 7. It first reproduced the original stored pooled features exactly, then
reloaded the pristine FP32 checkpoint before comparison. Every condition passed
bitwise no-op and repeated-forward controls. All conditions used identical
prepared uint8 frames, actions and BF16 noise; no labels were loaded. Resize
rounding was deliberately unchanged, so this audit does not measure its effect.

| Comparison | Codec relative L2 change | Prediction relative L2 change | All pooled means relative L2 change |
| --- | ---: | ---: | ---: |
| Original full BF16 → FP32 parameters with BF16 autocast | 1.964% | 1.591% | 0.218% |
| Then legacy interpolation → FP32-tau interpolation | 0% | 0% | 0% |

The interpolation tensors themselves changed by 0.179% relative L2 in the
second comparison, but model predictions and all 17 site means were bitwise
equal on this pilot. The first comparison changes the complete parameter/buffer
runtime bundle, including BOS and normalization; it does not isolate matrix
rounding alone. Site-level mean changes ranged from 0.189% to 0.542%. Peak
allocated memory was 10.52 GiB for FP32 parameters with BF16 autocast.

These measured differences justify restoring publisher numerical conventions.
They do not establish that numerical precision caused the poor probes, or how
probe accuracy will change. Readout, target observability and context still need
their own controlled comparisons. The [figure](../../figures/numerics_pilot.png)
shows relative changes, not physical prediction accuracy.

The label-free archive is
`/data2/ishaangp/mira-interp/captures/numerics_pilot.npz` (13,882,221 bytes), SHA-256
`af8078f9da7768c930f42213194a5a31020cfb6a3eb0b9eb26bccdbebdb479ae`.
It contains FP32 codec latents `[4,8,9,16,32]`, tiled interpolants/predictions
`[1,8,36,16,32]`, and pooled means `[32,17,2048]` for each condition.

```bash
NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=7 .venv/bin/python \
  scripts/audit_numerics.py --assets /data2/ishaangp/mira-interp/assets \
  --output /data2/ishaangp/mira-interp/captures/numerics_pilot.npz \
  --report results/review/numerics_pilot.json
.venv/bin/python scripts/plot_numerics.py
```

The audit refuses an existing output; a new run needs new output/report paths.

## Ranked fixes and experiments

1. **Numerical parity pilot, before recapture.** On the reserved match, compare
   full-BF16/legacy interpolation with FP32 parameters plus BF16 autocast, then
   exact FP32-tau interpolation. Hold video, actions and noise fixed. Compare
   codec latents, all 17 pooled sites and denoising predictions; retain no-op
   and repeat controls. Separately assess resize rounding. A small relative
   difference rules out one explanation; a large difference still needs a
   downstream physical evaluation.
2. **Preserve space and use appropriate targets.** Compare spatial grids or a
   fixed projection of unpooled tokens with global means. Include native codec
   tokens and RGB baselines under matched fitting budgets. Start with local
   player/ball-relative state, speed and distance as well as world coordinates;
   preserve canonical identity mappings. These are new hypotheses, so their
   selection cannot reuse the old confirmation set as fresh confirmation.
3. **Check visibility, temporal history and denoising stage.** Use fixed,
   separately registered context lengths and tau values; report all settings.
   Compare later latent steps with early/BOS steps. Audit whether each target
   is visually observable in the available views. Positive low-level controls
   can diagnose a broken readout before attempting every world-coordinate label.
4. **Establish an independent generated-video objective.** A probe trained on
   internal teacher-forced activations is not a validated generated-trajectory
   evaluator. Calibrate an independent video/state regressor on disjoint real
   clips and quantify its failure modes before causal scoring.
5. **Causal tests, then geometry/features.** Freeze candidates on discovery and
   selection data, construct defensible action-matched interventions, measure
   attribution, exact replacement, ablation and restoration with matched seeds
   and controls. Only after effects survive independent confirmation should
   SAE/BSF comparisons and linear versus geometry-aware multi-frame steering be
   interpreted as causal-coordinate experiments.

The practical first response to weak probes is a controlled diagnostic that can
reject specific explanations. It is not to search the exposed confirmation data
until a high score appears, or to label the completed pooled probe as a proof
that physical state is absent.
