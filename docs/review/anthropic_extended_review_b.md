# Additional primary-source review: entries 21–40

This review covers the final 20 of the 40 entries previously marked unread in
`results/review/anthropic_sources.json`. It supplements the earlier proposal
review; it does not alter that inventory or any registered experiment. All 20
publisher pages were retrieved successfully. The accompanying
[`anthropic_extended_b.json`](../../results/review/anthropic_extended_b.json)
records source hashes, inspected sections, limitations, and proposed follow-ups.

**Coverage means selected methods, controls, and limitations were inspected.**
It does not mean every appendix, equation, linked reference, interactive example,
or implementation was reviewed. Three pages failed the web text parser and were
read from a direct HTTP retrieval of the same primary publisher. Full source
copies remain private; this repository contains short paraphrases and links.
Research notes are preliminary and sometimes revise earlier advice. Statements
summarizing other groups' work inside an update are not independent reviews of
those groups' original papers.

The clearest remaining problem is the gap between **reading a state and changing
the computation that produces it**. These sources suggest ways to investigate
that gap. They do not establish an undiscovered MIRA implementation bug or promise
that a different dictionary will produce reliable physical control.

## What the completed MIRA evidence supports

The independently audited conditional-fidelity experiment is particularly
diagnostic. The native full-donor patch has mean standardized donor-error gain
0.682834. Replacing the donor's entire 1536-coordinate descriptor with its fixed
discovery mean still gives 0.672032. Across all 30 dictionaries, reconstructed
donor gains span 0.678243–0.685742; every condition is positive in only 6 of 11
matches. The mean-control difference from the native effect is −0.010802, with
descriptive whole-match interval [−0.035499, 0.006261]. See the
[audited result and full limitations](sparse_causal_fidelity.md).

The native tile has 294,912 coordinates, and these interventions preserve the
complement outside the 1536-coordinate descriptor. Consequently, recovering most
of the full-patch effect does not demonstrate that the dictionary recovered the
causal state coordinate. The mean control directly demonstrates that much of the
effect can survive without descriptor deviations from the mean. Dictionaries
improve conditional reconstruction; the original donor effect remains mostly
unexplained by that descriptor alone. Natural donors also differ in multiple
physical variables, so the effect is not an isolated change in ball height.

This review uses completed pre-rollout evidence. It does not inspect selection or
confirmation rollout outcomes, change a winning path, or treat a running study
as completed. Any subsequent tuning must use development data and a separately
registered, genuinely unused confirmation cohort. Previously inspected
confirmation matches cannot be reused as fresh confirmation for new choices.

## Practical diagnoses and bounded next experiments

| Priority | Diagnosis and existing evidence | Prospective test and deciding evidence |
| --- | --- | --- |
| 1 | The decoded state direction may represent observed state rather than a direction that writes future state. Teacher-forced probing and actual generation have different inputs, noise, and cache conditions. | On a new development cohort, compare fixed readout directions with directions localized by an output objective during the actual sampler. Hold actions, seed, context, frame, and dose fixed. Require exact interventions, collateral measurements, and a frozen generated-video evaluator before confirmation. |
| 2 | Most full-donor gain survives descriptor replacement by its mean; the retained native complement can dominate conditional fidelity. | Cross recipient versus donor descriptors with recipient versus donor complements, including the fixed mean descriptor. Compare dictionary reconstruction with descriptor-only transfer under the same complement. Report full outputs and local directional-response fidelity alongside reconstruction error; do not infer sufficiency from conditional retention. |
| 3 | Routing, entity binding, or context can determine whether a readable feature influences output. Canonical player slots and own-view roles are different targets. | Use controlled role/camera conditions and narrow versus broad examples to isolate query/key routing from output/value content. Preserve entity identity and actions across paired interventions. A causal path should survive exact component tests and negative examples, not only a high attribution score. |
| 4 | Learned video estimates are imperfect measurements, including rolling windows that contain observed context. Correct recomputation does not validate the measurement construct. | Inspect all predeclared representative videos and per-match outliers. Calibrate on known changes from synchronized or simulated video; test context-only, generated-only, and nuisance changes where feasible. Keep pixel effects, estimator predictions, and simulator-validated physical effects separate. |
| 5 | A sparse dictionary can optimize reconstruction while missing a rare physical feature or fragmenting a geometric computation. | Before another training sweep, inspect achieved activity, dead-feature frequency, reconstruction by state range, and the existing dense/mean controls. Any new initialization or architecture comparison must use matched data and measured budgets, preserve negative results, and test causal response fidelity. |
| 6 | Curves in PCA or fitted forward maps can reflect input correlations, camera context, or a convenient projection rather than a causal manifold. | Use held-out physical values and controlled state sweeps, then test composition and inverse edits with specificity controls. Geometry should predict unseen intervention effects, not only reconstruct correlated observations. |

The first diagnosis follows the distinction between representations of an input
and representations that cause an output in the
[July 2025 update](https://transformer-circuits.pub/2025/july-update/index.html)
and the ASCII/SVG generation experiments in the
[October 2025 update](https://transformer-circuits.pub/2025/october-update/index.html).
The proposed response-fidelity tests follow the distinction between activation
reconstruction and computation in
[Sparse mixtures of linear transforms](https://transformer-circuits.pub/2025/bulk-update/index.html).
These are transfers of methodology from language models, not established MIRA mechanisms.

Routing and context hypotheses are motivated by
[Progress on Attention](https://transformer-circuits.pub/2025/attention-update/index.html),
the [November 2025 case study](https://transformer-circuits.pub/2025/november-update/index.html),
and [HeadVis](https://transformer-circuits.pub/2026/headvis/index.html).
Head polysemanticity alone does not demonstrate superposition: high-rank heads
can implement several functions without exceeding their available dimensions.
Likewise, an attribution graph omitting important error or gating components is
not a complete explanation.

Measurement and geometry cautions follow
[Reflections on Qualitative Research](https://transformer-circuits.pub/2024/qualitative-essay/index.html)
and the stated confounding and intervention limitations in
[Emotion Concepts and their Function](https://transformer-circuits.pub/2026/emotions/index.html).
The relevance is experimental design and role binding; this review makes no
claim about emotions or introspection in a video world model.

## Source-by-source scope

| Original unread position | Primary source | Inspected material and relevance |
| --- | --- | --- |
| 21 | [February 2024](https://transformer-circuits.pub/2024/feb-update/index.html) | Dying features, ghost gradients, and tanh penalties. Better sparsity/loss can accompany harder-to-interpret features. Several optimizer recommendations were subsequently revised. |
| 22 | [January 2024](https://transformer-circuits.pub/2024/jan-update/index.html) | MNIST dictionaries, counterexamples to sparse-feature intuition, and prediction of future activations. Sparse or downstream-useful units need not identify unique underlying concepts. |
| 23 | [March 2024](https://transformer-circuits.pub/2024/march-update/index.html) | Discriminative feature attribution, attention output/value effects, revised optimizer advice, and attribution limitations. Exact interventions remain necessary when gradients miss nonlinear effects. |
| 24 | [Stage-Wise Model Diffing](https://transformer-circuits.pub/2024/model-diffing/index.html) | Model/data factorial controls and limitations. Data changes alone can produce apparently meaningful feature changes; representation change is not automatically computational change. |
| 25 | [Qualitative Research](https://transformer-circuits.pub/2024/qualitative-essay/index.html) | Main essay on summary statistics, visual structure, and rigor. Qualitative counterexamples can expose a defective metric; attractive structure can come from inputs. |
| 26 | [September 2024](https://transformer-circuits.pub/2024/september-update/index.html) | Successor-head triangulation and topic oversampling. Weight inspection, ablation, and attribution need not rank all heads alike; training distribution changes dictionary coverage. |
| 27 | [April 2025](https://transformer-circuits.pub/2025/april-update/index.html) | Jailbreak case, narrow feature visualizations, and dense features. Negative examples and unresolved error terms matter. Dense activity does not by itself imply uninterpretable structure. |
| 28 | [Progress on Attention](https://transformer-circuits.pub/2025/attention-update/index.html) | Attention factorization, mixture approaches, and limitations. Query/key routing and output/value content interact; output dictionaries do not explain how attention patterns form. |
| 29 | [August 2025](https://transformer-circuits.pub/2025/august-update/index.html) | Persona circuit vignette. Context-dependent effects and unusually strong interventions motivate checking missing components, intervention norms, and collateral changes. |
| 30 | [Sparse mixtures of linear transforms](https://transformer-circuits.pub/2025/bulk-update/index.html) | Low-rank gated transforms, reconstruction/Jacobian comparisons, and interference. Coordinate reconstruction and preservation of local computation are distinct tests. |
| 31 | [Crosscoder Model Diffing](https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html) | Shared-feature bias, toy model, and penalty modification. Shared support can win for objective-related reasons; apparent exclusivity depends on the dictionary objective. |
| 32 | [Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html) | Concept-vector construction, strength/layer controls, failures, and limitations. Injection can influence outputs through confounds; downstream behavior alone does not uniquely identify the injected concept. |
| 33 | [January 2025](https://transformer-circuits.pub/2025/january-update/index.html) | Dictionary optimization recipe, global normalization, initialization, and schedules. A later empirical recipe is not a fully ablated universal default. |
| 34 | [July 2025](https://transformer-circuits.pub/2025/july-update/index.html) | Input versus output feature language and domain-validation discussion. Reading an entity/state and producing it are different computational roles. Biology examples are secondary summaries here. |
| 35 | [November 2025](https://transformer-circuits.pub/2025/november-update/index.html) | Multiple-choice case with attention competition and query/key interventions. Information can remain represented while routing prevents its use; a single case does not establish a general circuit. |
| 36 | [October 2025](https://transformer-circuits.pub/2025/october-update/index.html) | Perceptual versus generation features, and data-point initialization. Directions useful for recognizing content may be weaker controls than output-related directions. |
| 37 | [September 2025](https://transformer-circuits.pub/2025/september-update/index.html) | Context-length effects with first/last sentence and unrelated-text controls. Feature overlap can depend on accumulated context rather than a static concept identity. |
| 38 | [Emotion Concepts](https://transformer-circuits.pub/2026/emotions/index.html) | Contrastive vectors, geometry, speaker roles, and limitations. Shared-looking geometry can have distinct role-bound directions; linear probes and synthetic elicitation can miss or confound structure. |
| 39 | [HeadVis](https://transformer-circuits.pub/2026/headvis/index.html) | Selected head examples, physical-value-like sweeps, and discussion. Broad/narrow distributions and competing attention candidates expose failures that individual examples miss. Interactive UI/code not reviewed. |
| 40 | [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) | Architecture, training overview, behavioral evaluations, confabulations, and limitations. Unsupervised text explanations require causal/negative controls and do not replace physical-state ground truth. |

There is no recommendation to retrofit all these methods into the current run.
The most economical follow-up is to distinguish descriptor effects from the
retained complement and to localize output-driving computation under generation
conditions. Optimizer changes, attention dictionaries, or natural-language
explanations should follow a demonstrated need and their own development tests.
