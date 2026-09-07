# Independent saved-probe audit

The revised development probes passed an independent check of all 39 readouts
and 468 saved main/shuffled models. The audit took 22.694 seconds. The largest
relative residual in the weighted ridge normal equations was **1.80e-13**.
No model was refitted and no confirmation features or labels were opened.

`scripts/audit_development_probes.py` requires the completed capture audit,
frozen protocol, original match roles, and exact capture/model/analysis hashes.
It checks the 37 registered current-frame readouts and the two temporal readouts
selected by the registered rule. Source archives are hashed before loading and
again before reporting success.

The auditor independently reconstructs world-coordinate ego and ball-minus-ego
targets, current/previous rows, equal-match weights, and the seeded whole-match
label shuffle. It recomputes normalization from discovery matches only. For
each saved coefficient matrix `B`, standardized features `Z`, standardized
targets `Y`, weights `w`, and its selected regularizer `alpha`, it checks

```text
R = Z.T @ (w[:, None] * (Z @ B - Y)) + alpha * B
relative residual per target = ||R|| / max(||Z.T @ (w[:, None] * Y)||, 1e-12)
```

All relative residuals must be below `1e-6`. Selection predictions and normalized
losses are recomputed directly from those saved coefficients. Every selected
alpha must minimize its preserved seven-value selection curve, including the
documented tie order. All residual-site winners are independently recomputed.
Unselected-alpha models are not saved, so the audit verifies their preserved
curve values as provenance rather than recomputing those fits.

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -u \
  scripts/audit_development_probes.py
```

The result is
[probe_audit.json](../../results/development_probes_v3/probe_audit.json), SHA256
`3db136b7ebc8cc020728a6415ee5279a9c0c841b1db06d0f5e481601e734f52c`.
This establishes computational correctness of the saved development results;
it does not convert adaptive selection into confirmation or physical causality.
