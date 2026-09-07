# MIRA physical-state interpretability

This project tests whether a video world model represents physical state in coordinates
that can causally control its predicted trajectories. Experiments advance only after
their prerequisites pass. A software test is not evidence of a physical representation.

## Current execution

Pretrained execution and intervention wiring have passed on NVIDIA A40 GPUs. Official MIRA code is pinned
to `3d739ec2d31daf83559d33eb01727cea48fe90f7`. Official pretrained MIRA weights were not
found in the checked release channels. The executable model candidate is **Alakazam's
MIRA Mini 4P**, an independently trained 1B reproduction, with separately pinned weights.
Results for this model must not be described as results for the original 5B MIRA.

Rocket Science access is working, and all 31.1 GB of the pinned test split passed
publisher checksum verification. The full 61-match data audit rejected eight
matches at the unchanged clock-calibration gate. A separately registered
[quality-qualified study](docs/05_cohort_amendment.md) retains all 53 passing
matches: 31 discovery, 11 selection, and 11 confirmation, eight clips each.
The original failed audit and all exclusions remain available.

The full two-GPU capture and independent aggregation passed: **424 clips, 13,568
rows, all 17 sites**. Each GPU peaked at 6.51 GiB. Every saved label, timestamp,
player/view identity and source-frame join was checked against the audited input.

**The observational probe study is complete.** The site chosen before confirmation
(block 5 output) achieved **20.1% lower standardized error than the mean baseline**
on 11 confirmation matches. Ball-position R² was **0.506 / 0.758 / 0.871** for X/Y/Z;
horizontal ball velocity and many player velocities remained weak. All **46 tests**
and **91 completion checks** passed. See the [results and limitations](docs/07_observational_results.md).

![Held-out decoding at every residual site](figures/observational_layers.png)

This measures annotation decoding with observed target pixels present. Physical
causal validation, geometry, sparse features, and steering remain unrun: controlled
state-reset/rendered pairs and a validated generated-video physical evaluator are
still required. The full proposal is **not complete**.

The [pretrained readiness report](results/pretrained_smoke.json) records strict loading,
all 16 residual layers plus block-0 input, exact no-op controls, and generated output
invariance to hidden future-frame placeholders. The short run took 44.6 seconds and
peaked at 4.65 GiB allocated GPU memory. This is not a full-study memory estimate.

![Four-view short continuation from a public context](figures/pretrained_smoke.png)

The figure is a two-frame readiness continuation from Alakazam's public example.
It does not measure physical accuracy. Full saved readiness tensors are packaged in
the [readiness release](https://github.com/guptaishaan/mira-interp/releases/tag/readiness-2026-09-06),
with checksums in [the archive manifest](results/readiness_archive.json).

## Work and documentation

- [Completed observational results, CSVs and figures](docs/07_observational_results.md)
- [Protocol and stage gates](docs/PROTOCOL.md)
- [Environment and reproducibility](docs/00_setup.md)
- [Model availability audit](docs/model_access_audit.md)
- [Data preparation](docs/01_data.md)
- [Residual hooks and patching tests](docs/02_instrumentation.md)
- [Pretrained model readiness](docs/03_pretrained.md)
- [Real-observation capture](docs/04_capture.md)
- [All-layer probe analysis](docs/04_probes.md)
- [Cohort amendment and exclusions](docs/05_cohort_amendment.md)
- [Causal prerequisites and pair availability](docs/06_causal_prerequisites.md)
- [Current observational protocol](configs/observational_probe_v2.json)
- [Execution handoff](docs/EXECUTION_HANDOFF.md)

Compact reports live in `results/`; exportable figures live in `figures/`.
All pooled research features and full pilot residuals are in the
[observational results release](https://github.com/guptaishaan/mira-interp/releases/tag/observations-2026-09-06),
with [verified array and archive hashes](results/observation_archive.json).
Large downloaded weights and gated source data stay outside Git. Their provenance,
checksums, and reproducible download commands are recorded instead.

## Attribution

MIRA is by General Intuition, Kyutai, and collaborators:
[code](https://github.com/mira-wm/mira), [paper](https://arxiv.org/abs/2607.05352).
MIRA Mini is [Alakazam's independent reproduction](https://huggingface.co/alakazamworld/mira-mini-4p).
Rocket Science and MIRA Mini weights use CC BY-NC-SA 4.0. Rocket League content is
copyright Psyonix LLC / Epic Games, Inc. See [third-party notices](THIRD_PARTY.md).
