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

Rocket Science access is now working. Data integrity preparation is in progress:
the first pilot exposed a source-clock/event-clock offset, so research capture is
waiting for a corrected alignment audit. A metadata-only 37/12/12 discovery/selection/
confirmation cohort plus one separate pilot is registered before research capture.
Causal validation, geometry, sparse feature training, and steering remain unrun.

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

- [Protocol and stage gates](docs/PROTOCOL.md)
- [Environment and reproducibility](docs/00_setup.md)
- [Model availability audit](docs/model_access_audit.md)
- [Data preparation](docs/01_data.md)
- [Residual hooks and patching tests](docs/02_instrumentation.md)
- [Pretrained model readiness](docs/03_pretrained.md)
- [Registered observational probe protocol](configs/observational_probe_v1.json)

Compact reports live in `results/`; exportable figures live in `figures/`.
Large downloaded weights and gated source data stay outside Git. Their provenance,
checksums, and reproducible download commands are recorded instead.

## Attribution

MIRA is by General Intuition, Kyutai, and collaborators:
[code](https://github.com/mira-wm/mira), [paper](https://arxiv.org/abs/2607.05352).
MIRA Mini is [Alakazam's independent reproduction](https://huggingface.co/alakazamworld/mira-mini-4p).
Rocket Science and MIRA Mini weights use CC BY-NC-SA 4.0. Rocket League content is
copyright Psyonix LLC / Epic Games, Inc. See [third-party notices](THIRD_PARTY.md).
