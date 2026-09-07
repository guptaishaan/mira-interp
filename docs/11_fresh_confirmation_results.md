# Fresh confirmation of the repaired readouts

The frozen readouts reproduced strong viewing-player and ball-relative position
decoding on **23 new matches, 184 clips and 4,416 scored rows**. The independent
audit reproduced all 32 fixed readouts, their controls, 500 whole-match bootstrap
intervals and paired comparisons. No model was refitted or chosen using these
confirmation outcomes.

## Results

| Frozen primary readout | Standardized MSE [95% interval] | Mean baseline | Reduction against mean |
| --- | ---: | ---: | ---: |
| Ego and ball-relative positions, mean block 5 | 0.1925 [0.1767, 0.2095] | 1.0067 | 80.9% |
| Ego and ball-relative velocities, block 6 mean + previous difference | 0.7425 [0.7000, 0.7995] | 1.0377 | 28.4% |
| All 12 role targets, mean block 6 | 0.4665 [0.4401, 0.4986] | 1.0222 | 54.4% |
| All 30 canonical absolute targets, mean block 5 | 0.8752 [0.8190, 0.9463] | 1.0487 | 16.6% |

The errors use discovery-set scales and equal match weights. Percentages compare
each readout with its own mean baseline; changing target definitions prevents a
like-for-like comparison with the older absolute-state study. Intervals are
descriptive and do not account for all comparisons or training-seed uncertainty.

![Frozen role position and velocity readouts](../figures/fresh_role_decoding.png)

| Target | X R² | Y R² | Z R² |
| --- | ---: | ---: | ---: |
| Ego position | 0.660 | 0.900 | 0.974 |
| Ball minus ego position | 0.520 | 0.694 | 0.926 |
| Ego velocity | 0.090 | 0.100 | 0.883 |
| Ball minus ego velocity | -0.007 | 0.014 | 0.543 |

Vertical velocity is readable, but horizontal velocity remains weak. The figure
and complete [target CSV](../results/fresh_confirmation_v3/target_metrics.csv)
retain negative results. A high R² indicates explained variation, not precise
physical measurement. These targets use world XYZ axes; they are not rotated into
camera-heading coordinates.

## What was frozen and checked

The metadata-only reservation chose 24 official-dev matches, disjoint from all
62 old test matches. The unchanged clock-quality rule rejected one match;
the remaining 23 passed all eight clips, with no replacements. Publisher checksums
passed for the 45.34 GB of required source shards. Fractional resized inputs
exactly reproduced the old preparation when rounded, retaining original source
timestamps, labels, actions and player identities.

Commit `9cd2198df5e3c9d3505b3d24e7f1faf894f9b800` published
[`configs/fresh_evaluation_v3.json`](../configs/fresh_evaluation_v3.json) before
fresh capture. It binds the audited development models, every primary/comparator
choice, capture code and evaluation code. Selection considered only the original
31 discovery and 11 selection matches. The fresh GPU worker completed all 184
clips in 293 seconds; the independent confirmation audit took 45 seconds.

The audit independently reconstructed role labels from raw source labels, verified
every capture/source join, reproduced all fixed-model metrics and recomputed the
500 paired whole-match bootstrap draws. See the
[audit description](review/fresh_confirmation_audit.md),
[audit report](../results/fresh_confirmation_v3/confirmation_audit.json) and
[complete results](../results/fresh_confirmation_v3/confirmation.json).

## Interpretation and next step

The viewing-player convention addresses a substantial part of the original
player-readout weakness. The tested compressed spatial readout did not improve on
the mean readout; the proven pooling bottleneck is therefore not an established
cause of the original score. Numerical repairs improve fidelity to released
inference, but their isolated effects on decoding were not measured here.

This confirms observational decoding with target pixels present, not prediction
of unseen future state. Fresh matches were unseen by these interpretability
fits; the pretrained model's complete validation history is unavailable. The
model is Alakazam MIRA Mini 4P, not the original 5B MIRA.

The next evidence comes from registered exact internal interventions, including
ablation, restoration and norm-matched controls. Generated-trajectory steering
still requires independent video measurement and rollout tests.

## Reproduction

After reconstructing the pinned assets and passing development audits, execute
the following in order. Completed locks and results refuse overwrite.

1. `scripts/register_fresh_evaluation.py`
2. `scripts/capture_fresh_confirmation.py` with the registered GPU worker options.
3. `scripts/confirm_development.py`
4. `scripts/audit_fresh_confirmation.py`
5. `scripts/export_revision_results.py --phase confirmation`

Use `.venv/bin/python`, `PYTHONPATH=src:external/mira/src` and
`NUMPY_MADVISE_HUGEPAGE=0`. Full source provenance and qualified cohort manifests
are committed; gated source videos and raw label arrays remain local.
