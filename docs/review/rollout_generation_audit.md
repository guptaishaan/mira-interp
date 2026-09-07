# Independent generated-rollout integrity audit

`scripts/audit_rollout_generation.py` checks the completed saved rollout manifest, separate from the sampler and from the frozen VideoMAE evaluation. It uses CPU only and does not load source video or simulator labels. The input action arrays are read only to verify their hash against every condition. This is an engineering completion gate, not evidence that steering improves physical behavior.

Before a phase can pass, the audit checks the frozen registration, implementation hashes, prerequisites, exact match/seed/path/dose coverage, and each worker's disjoint registered match partition. Selection requires a passed independent pilot audit; confirmation requires the passed selection audit. Each artifact and its per-condition report must match their stored SHA256 and identities. The auditor verifies all six saved NPZ arrays, their shapes/dtypes/finiteness, 0–1 video range, and the latent tensor hash.

For every condition, the auditor reconstructs the expected 45-call sampler trace: one eight-latent context initialization, then four repetitions of ten denoising calls and one clean cache update. It checks the actual FP32 or BF16 schedule, the single first-latent step8 intervention, cache flags, and view/time layout. All action-feature hashes must equal the paired baseline. All calls through the intervention input and the saved pre-edit residual tile must agree exactly. The eight context latents must remain unchanged. Decoded context differences are independently measured and reported because evaluator windows contain these frames.

The saved spatial descriptor is independently recomputed from twelve explicit 3×4 token bins and the pinned orthonormal channel projection. The audit recomputes baseline probe height, clipped/effective doses, all five requested path directions, descriptor and raw-lift norms, realized edit norms, stored rounding error, position-proxy shifts, and per-time latent differences. Numerical comparisons allow small FP32 reduction error; ideal lifts are checked within one BF16 storage spacing when the sampler uses BF16. This accounts for storage rounding without modifying the intervention. A zero effective dose must reproduce the paired baseline exactly.

The reserved pilot must include a nonzero active quadratic edit. The registered support-only rule chooses +300 whenever that has a positive effective dose after clipping, otherwise −300; it does not change the research dose grid. Its unhooked, hooked, alternate-future-placeholder, and zero-edit controls must have identical persisted decoder/context/latent/action hashes, and the hashes must agree with the saved baseline. Those discarded controls are checked through recorded evidence and executed-code bindings, not independently rerun. When pixels were clipped, the original unclipped decoder values are unavailable; their hashes remain producer evidence, while saved clipped arrays and all other metrics are checked directly.

Run after each complete phase:

```bash
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  scripts/audit_rollout_generation.py --phase pilot
```

Use `selection` or `confirmation` for subsequent phases. Successful output is `results/rollout_steering_v1/{phase}_audit.json` with status `passed_rollout_generation_audit` and the exact completed `manifest_sha256`. Existing audit files are not overwritten. CPU tests deliberately corrupt cache timing, action-hash format, target call, latent order, and array-comparison inputs, and verify the independent descriptor/lift right-inverse relation. No generated experiment is implied by those tests.

## Completed reserved pilot

The corrected registered pilot completed five inferences and passed the independent CPU audit in 1.99 seconds. `results/rollout_steering_v1/pilot_audit.json` has SHA256 `c4c4b48d9aea274790126e092880a86308f3ad45a7f4200275efe9e59c1255fc` and binds pilot manifest `99a0dbeaf4a216e6fc9053b5d4adb64f5b38523664790386f6faf50c30459b3b`. Both saved conditions passed all trace, source/action, array, context, and intervention-metric checks. The four control hashes matched. The +300 requested edit had effective dose +300, with realized residual edit L2 6.08856 and requested lift L2 6.01542. The descriptor rounding-error L2 was 0.05751.

The pilot used 12.702 GB peak allocated GPU memory. Its two compressed saved outputs total 72.46 MB; recorded individual control inference times were 5.15–7.34 seconds. These are reserved engineering measurements, not a statistical steering result. The initial failed attempt used uint8 actions directly as embedding indices and is preserved under `results/rollout_attempts/01_action_dtype`; the rerun casts those unchanged binary values to int32 before the upstream action encoder.

The saved trace records BF16 latent/tau dtype, but the original producer does not separately persist the intervened block-output dtype. The auditor's BF16 rounding bound is therefore conservative if the residual used a more precise dtype. Actual saved tile differences, descriptors, and norm/proxy metrics are still recomputed directly. This limitation does not establish or refute a physical intervention effect.
