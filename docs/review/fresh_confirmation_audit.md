# Independent fresh-confirmation audit

The audit passed in **45.051 seconds**: all 184 clips, 4,416 scored rows and 32
fixed reported readouts were verified. Every published main, shuffled and mean
baseline metric, bootstrap interval and paired comparison was reproduced.
Report SHA256:
`8e966474da029de603eadc7a8b5f91002afa81ea44f654be4dec92dd8e1a6c0d`.

`scripts/audit_fresh_confirmation.py` checks the fixed revised probes on the
newly reserved cohort without fitting or choosing hypotheses. It first binds
the completed result to the frozen registration, independently audited
development report and saved models. It reconstructs every registered choice
from the prior development losses before opening fresh arrays.

The source audit covers exactly 23 matches, 184 clips and 4,416 scored rows.
Each match contributes eight clips, four views and latent indices2..7. Original,
prepared and captured hashes must match. Every saved label, timestamp, frame
index and player identity is compared with its original source NPZ. No original
test match, including the earlier quality exclusions, may enter the fresh cohort.
Worker timestamps must follow the choice freeze.

Ego targets are independently reconstructed by selecting the six position and
velocity columns for the viewed canonical player. Ball-minus-ego subtracts
those columns from the six ball columns, in world axes. Temporal readouts use
the current feature and its difference from the preceding latent within the
same clip and view. Predictions use the exact saved coefficients and discovery
normalization; every model fingerprint is checked.

The auditor independently recomputes every published fixed model's MAE, RMSE,
R2 and normalized MSE, including the shuffled model and discovery-mean baseline.
RMSE uses equal-match squared errors. Point R2 uses the equal-match variance of
explicitly centered fresh labels. Normalized MSE uses the saved discovery
target scales, so it has a different denominator from R2. The audit preserves
negative per-axis R2 values.

For example, the frozen role-velocity primary has ball-minus-ego horizontal
x-velocity R2 of **-0.00687**, despite positive aggregate improvement. The
absolute-velocity primary also has negative R2 for ball x, player1 x, and player3
x/y velocity. These axes remain in the report and must remain visible in
figures; aggregate gains do not imply uniformly useful velocity readout.

All confidence intervals are reproduced using the registered 500 whole-match
bootstrap samples, seed20260907. Paired comparisons resample the same matches
for both predictions and average standardized squared-error differences.
Positive gain means lower error than the comparator. Every fixed comparator,
including mean-versus-spatial and the mean baseline, is checked.

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -u \
  scripts/audit_fresh_confirmation.py
```

The result is `results/fresh_confirmation_v3/confirmation_audit.json`. The audit
rehashes source arrays, coefficients, registration, result and frozen code
before reporting success. It does not change the registered result or silently
overwrite an earlier audit. Fresh match generalization here remains
teacher-forced annotation decoding; it does not establish predictive rollouts
or physical control. Bootstrap intervals are descriptive and do not include
retraining uncertainty or a family-wide significance guarantee.
