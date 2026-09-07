# Additional primary-source review: first 20 inventory-only entries

Checked 2026-09-07 UTC. This extension inspects the first 20 entries marked `index inventory only; not read` in the original [source inventory](../../results/review/anthropic_sources.json), preserving its order. The original inventory and all frozen study files are unchanged. This is a reading of 20 primary pages: 17 technical, conceptual, educational or engineering pages, one research-thread landing page, one editorial, and one video landing page. It is **not** a claim to have read every appendix, linked paper, reference, notebook, interactive figure, or video.

The [machine-readable review](../../results/review/anthropic_extended_a.json) records each URL, resolved URL, retrieval time, HTTP status, HTML/text SHA256, exact inspected text-line ranges, limitations and MIRA-specific inference. Full source caches remain local under `/data2/ishaangp/mira-interp/review_sources/anthropic_extended_a/`; they are not republished. Text extraction can omit custom equations and interactive content. No new experiment, fit, GPU execution, or examination of the running rollout outcomes was performed for this review.

The main additional diagnosis is a **missing stronger random control**, not a demonstrated implementation defect. Equal edit norms do not control for the model's uneven activation variance. The other unresolved issues concern optimization adequacy, feature meaning, representation families, and which computations read or write the edited state. These are hypotheses for separately registered development work. They do not justify changing the current study after its hypotheses were frozen.

## What was read

IDs refer to the exact source records and inspected line ranges in the JSON.

| ID and primary source | Relevant method or limitation; consequence for this project |
| --- | --- |
| A01 [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/) | Activation-conditioned explanation models can infer more than the target explicitly represents. Such explanations would add another readout, not establish geometry or a causal physical variable. |
| A02 [Automated auditing](https://alignment.anthropic.com/2025/automated-auditing/) | Repeated audits of implanted behaviors permit tool comparisons. Artificial testbeds, contamination and fixation remain limitations; independent checks do not imply exhaustive scientific coverage. |
| A03 [Circuits thread](https://distill.pub/2020/circuits/) | The landing page organizes vision features and circuit families. Its article summaries were inspected; the linked investigations were not read as part of this item. |
| A04 [Distill hiatus](https://distill.pub/2021/distill-hiatus/) | Editorial discussion of scientific publication. It contributes no model experiment or technical diagnosis. |
| A05 [Transformer exercises](https://transformer-circuits.pub/2021/exercises/index.html) | Explicit weight-construction exercises explain routing and composition. Exercise statements were read; solutions were not implemented or used as evidence about MIRA. |
| A06 [Transformer-circuits framework](https://transformer-circuits.pub/2021/framework/index.html) | Separates attention routing from value movement. Its small attention-only models and fixed-attention analysis do not explain all interactions in an action-conditioned video transformer. |
| A07 [Garçon](https://transformer-circuits.pub/2021/garcon/index.html) | Persistent models, named hooks and local reductions support efficient experiments. Reducing a tensor remains a scientific choice about retained information. |
| A08 [Transformer videos](https://transformer-circuits.pub/2021/videos/index.html) | Only the textual introduction was read. The videos were not watched; the page identifies their informal, partly superseded status. |
| A09 [Induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Combines architectural perturbations with ablations. Redundant routes and altered MLP statistics complicate causal interpretation; pattern-preserving and full ablations answer different questions. |
| A10 [Variables and interpretable bases](https://transformer-circuits.pub/2022/mech-interp-essay/index.html) | Identifying a variable also requires understanding operations on it. A convenient probe direction is not automatically such a variable. |
| A11 [Softmax linear units](https://transformer-circuits.pub/2022/solu/index.html) | Blinded quick interpretations can miss low-activation polysemanticity. Inspecting positives, negatives and activation ranges is stronger than naming features from their largest activations. |
| A12 [Toy superposition models](https://transformer-circuits.pub/2022/toy_model/index.html) | Known sparse factors reveal geometry and correlation-dependent merging. Real-model geometry need not follow these toy results; physical nuisance correlations remain important. |
| A13 [Interpretability dreams](https://transformer-circuits.pub/2023/interpretability-dreams/index.html) | Correlated precursor features can support a probe before the proposed higher-level feature exists. The broader agenda is explicitly speculative. |
| A14 [July 2023 update](https://transformer-circuits.pub/2023/july-update/index.html) | Further optimization and an explicit cancellation algorithm revise earlier toy-model interpretations. Apparent representational surprises can have simpler explanations. |
| A15 [May 2023 update](https://transformer-circuits.pub/2023/may-update/index.html) | Sparse factorization can produce composites; circular feature families can discretize. Its attention-superposition interpretation was subsequently qualified in A14. |
| A16 [Privileged residual bases](https://transformer-circuits.pub/2023/privileged-basis/index.html) | Rotation experiments distinguish coordinate alignment from several proposed causes. Large coordinates alone are not semantic features; optimizer attribution remains provisional. |
| A17 [Composition and superposition](https://transformer-circuits.pub/2023/superposition-composition/index.html) | A probe can recover a shared property from combination-specific units. Decoding that property does not identify an independently used feature. |
| A18 [Memorization and double descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html) | Finite-data toy models can encode individual examples rather than generalizing factors. This motivates data and optimization checks, not a universal training-step prescription. |
| A19 [April 2024 update](https://transformer-circuits.pub/2024/april-update/index.html) | Structured controls address weaknesses of isotropic random perturbations. Training loss, model influence and interpretability are separate evaluation targets. |
| A20 [August 2024 update](https://transformer-circuits.pub/2024/august-update/index.html) | Contrastive and sorting tasks test explanations on activation examples. They cover limited notions of meaning; small differences between variants should not be overstated. |

## Specific unresolved diagnoses

1. **Anisotropy and control strength.** The registered random path samples an isotropic vector in the 1,536-dimensional descriptor and matches its lifted norm. A19 motivates a separate control sampled from discovery covariance or an appropriate feature subspace, with the same raw edit norm. This would test a confound; it would not assume the existing effects are spurious. Current code: [rollout steering](../../src/mira_interp/rollout_steering.py).

2. **Training adequacy.** All 30 dictionaries were correctly trained for the registered 1,000 updates, one width and two activity budgets. That establishes a reproducible budget comparison, not convergence or equality of achieved reconstruction. A14/A18/A19 motivate a separately frozen development budget study. No paper establishes that these particular models are undertrained. Current specification: [feature protocol](../../configs/feature_development_v1.json).

3. **Feature meaning and independence.** Whole-view descriptors have verified view identity, not object localization or visibility. Future semantic checks should predeclare matched positive/negative cases and inspect activation ranges. The successful annotation decoders and conditional intervention effects remain useful evidence, but do not by themselves name independent ball or velocity features. This inference draws on A01/A10/A11/A13/A17/A20.

4. **Alternative representation families.** A12/A15 motivate value-specific or nuisance-conditioned readouts in addition to a global scalar map. These need new development data rules and held-value evaluation. They do not turn the completed weak nonlinear comparisons into evidence of a manifold, and flexible readouts can introduce their own structure.

5. **Writer, reader and timing.** A06/A09 suggest separating the computations that produce and consume a representation. The running study deliberately edits one block output, one view, one denoising step and the first generated latent. Component, longer-context and other-time contrasts remain outside that test. Current outcomes must not be used to select a new hypothesis and call its evaluation fresh confirmation.

These are research judgments from the cited sources and the current local protocol, not claims those papers make about MIRA. The JSON binds the local files used to distinguish existing controls from untested ones.

## What this review does not undo

The [conditional-fidelity audit](sparse_causal_fidelity_audit.md) already includes discovery-mean descriptor controls and explicitly retains the native activation complement. That is a real restriction on what the sparse experiment can establish; additional literature does not convert conditional fidelity into feature sufficiency. Similarly, the [sparse audit](sparse_development_audit.md) already reports reconstruction/activity tradeoffs and the shuffled-time control. A different training budget or semantic evaluation would be new work, not a correction to an incorrectly executed registered comparison.

No new pretrained-adapter defect was demonstrated by these readings. The current five-path generated study should finish with its hypotheses, controls and measurement gates intact. Its learned video estimates still require caution because accuracy on edited generated videos has not been independently established. That limitation remains even if every execution and numerical audit passes.
