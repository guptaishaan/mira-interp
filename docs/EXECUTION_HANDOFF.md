# Execution handoff

Execute the proposal in order, audit each prerequisite, document key steps, and
publish to `https://github.com/guptaishaan/mira-interp.git`. The user authenticated
Hugging Face and GitHub and authorized publishing the work. Continue within this
scope without asking again.

## Presentation priority

The user clarified that this should demonstrate a strong, careful start to the
project, not a claim of overall success or a rush through every proposed method.
Keep the root README short and plain, with only two initial-progress figures:
the all-layer position readout comparison and frozen decoding on new matches.
Use `docs/progress/`, `results/progress/`, and `figures/progress/` for that small
subset. Keep detailed experiments, audits, and negative findings separately
accessible through `docs/README.md` and the folder indexes. The prior long README
is preserved in `docs/archive/project_log_before_readme_simplification.md`.

The already registered rollout study may finish under its unchanged protocol.
Its completion is not an overall project-success claim. Do not select favorable
paths or add new experiments just to fill the proposal checklist. Further causal
claims require better evidence, including specificity and measurement validity.

## Active revision

The user requested a literature-led diagnosis and repair, then said to continue.
Use `docs/08_review_and_repair.md` and `configs/development_v3.json`. Preserve all
v2 results. The new study uses only original31 discovery/11 selection matches;
old confirmation is exposed and cannot count as untouched again.24 official-dev
candidate matches are reserved by metadata in `data/fresh_confirmation_split_manifest.json`.
Read live `results/development_prepare.json`, `development_capture_pilot.json`,
worker reports and `fresh_confirmation_download.json` before resuming jobs.
All substantial NumPy processes require `NUMPY_MADVISE_HUGEPAGE=0` on this host.

The revised development capture and all 468 fitted/control models passed
independent audits. Their choices were frozen and pushed in commit
`9cd2198df5e3c9d3505b3d24e7f1faf894f9b800` before fresh capture. Exactly 23 of the
24 candidates passed the unchanged clock rule, with no replacement. Fresh
confirmation and its independent audit are now complete: 184 clips, 4,416 rows,
32 fixed readouts. Role-position error is 80.9% below its mean baseline; horizontal
velocity remains weak. See `docs/11_fresh_confirmation_results.md` and
`results/fresh_confirmation_v3/confirmation_audit.json`. Never refit or reselect
using these exposed fresh results.

The causal discovery audit passed all 31 matches and two seeds, freezing block
15/14/13 outputs. Registered exact selection interventions completed and passed
their independent audit: 660 conditions, 11 matches, two seeds. Full donor effects
were positive on only 6/11 matches; probe-component effects were small. Preserve
the narrower internal-output claim in `docs/09_internal_causal_development.md`.
The first discovery attempt received SIGTERM; 32 verified artifacts were reused
and the remaining 30 completed, with unchanged code/protocol. The interruption
is preserved. The VideoMAE evaluator uses development data only; fresh confirmation
outcomes must not be used to select evaluator models. Frozen ridge, MLP and partial
fine-tuning completed; the independently audited selected step500 model achieved
0.698319 real-video and 0.70625 codec-reconstruction NMSE. Horizontal relative
velocity remains weak and edited generated-video accuracy is unverified.

The registered geometry comparisons completed with an independent audit. None
established a nonlinear advantage; richer heading harmonics were worse. All30
sparse dictionaries completed and passed the independent CPU audit at
`results/feature_development_v1/sparse_audit.json`. ReLU had the lowest selection
reconstruction error; the true temporal variant did not outperform shuffled-time
controls. Preserve this outcome and the whole-view, non-entity-localized scope.

Sparse causal fidelity completed and passed its independent audit:1,364 conditions
on22 seed-pairs. Full donor mean gain was0.682834; replacing its descriptor by the
discovery mean still gave0.672032. Dictionaries modestly improved conditional
reconstruction but cannot be credited with carrying the full transfer effect.
All23 full tensor archives are in the verified `sparse-fidelity-v1-2026-09-07`
release; the native complement limitation is documented explicitly.

The prepared rollout inputs passed an independent audit: one pilot,
10 selection matches and22 prior fresh-confirmation matches, each with16 context
and8 future frames. Future source pixels are hidden from inference. The first
generation pilot caught uint8 keyboard indices before baseline generation; its
failure is preserved, and an int32 copy repairs the upstream embedding contract.
The corrected five-inference pilot and fixed VideoMAE evaluation both passed.

Storage-only v2 uses lossless DEFLATE1, after an exact benchmark and another full
pilot. All six generated arrays and all four prediction-array contents match v1
bitwise. Both pilot audits and independent storage parity passed. The experimental
model, actions, seeds, matches, sites, paths and doses are unchanged.

**The full steering study is running**, launched after commit`9f1e7da` was pushed.
Durable tmux session:`mira_rollout_study_v2`; entrypoint:
`scripts/run_registered_rollouts.py`. Read
`results/rollout_steering_v2/study_status.json` for live status, current stage,
completed-stage hashes and frozen code/prerequisite bindings. **Do not edit any
file listed in its `fixed_bindings` while the supervisor runs.**

The sequence is selection generation420→independent generation audit→fixed
VideoMAE measurement→independent metric audit, then the same sequence for924
confirmation rollouts. Two match workers use GPUs6/7; video scoring uses GPU6
after generation finishes. No outcome selects a new path or dose. Pilot-based
runtime is about two hours, and full private output is projected at61GB.
`resource_review.json` is an immutable pre-launch snapshot; `study_status.json`
is the live authority. A successful terminal supervisor status still says
`pending_publication`; do not call scientific physical steering established.

Generated artifacts:`/data2/ishaangp/mira-interp/rollout_steering_v2`.
Measurement caches:`/data2/ishaangp/mira-interp/generated_evaluation_v2`.
Compact reports:`results/rollout_steering_v2` and`results/generated_evaluation_v2`.
The protocol`configs/generated_evaluation_v1.json` stays frozen for v2 too.
Export all paths/negative results, audit figures, package every generated future
tensor in release parts below2GB, and verify remote hashes. Only generated
frames16–23 and latents8–11 may be public; observed context remains private.
Update README and this handoff, commit/push all completed outputs, and verify
remote HEAD. Do not stop while required stages remain runnable or running.

## Environment

- Workspace: `/ccn2/u/ishaangp/mira-interp`; Python: `.venv/bin/python`.
- Imports: `PYTHONPATH=src:external/mira/src`.
- Only physical GPUs **6 and 7** are authorized. Recheck ownership before use.
  GPUs 0–5 belong to another user's training job.
- Large assets: `/data2/ishaangp/mira-interp`; keep a 25 GiB reserve.
- Source/model/software pins: `configs/sources.json`, `configs/tested_versions.txt`.
- GitHub CLI: `/ccn2/u/ishaangp/bin/gh`. Credentials remain in private home paths.

## Completed data and capture gates

The executable model is Alakazam MIRA Mini 4P, an independent reproduction.
Official MIRA pretrained weights were not located. Strict loading and the codec
pairing check passed; every embedded codec tensor matches the standalone codec.
The public-context readiness output is published in the `readiness-2026-09-06`
GitHub release. This is engineering evidence only.

All 31.1 GB of the pinned Rocket Science test split passed publisher hashes.
The original 61-match data audit **failed**, with seven inconsistent source clocks
and one match with insufficient calibration anchors. Preserve that failure.

A separate v2 protocol was frozen before research capture/fitting. Exactly all
53 fully passing matches retain their original roles: **31 discovery, 11 selection,
11 confirmation**, eight clips each. One separate engineering pilot stays excluded.
The unchanged timing gate requires three goals per view and residuals at most
0.1 seconds. Detailed lineage and limits are in `docs/05_cohort_amendment.md`.

Use `configs/observational_probe_v2.json`, SHA256
`e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2`,
`data/qualified_split_manifest.json`, `data/qualified_clip_manifest.json`, and
`results/qualified_data_audit.json`. Original v1 files must remain unchanged.

The v2 reserved GPU pilot and both full workers passed. Each worker captured 212
clips, using 6.51 GiB peak allocated GPU memory, then exited. Independent aggregation
passed every input/output hash and exact per-view source-label join. Both GPUs
are free as of this handoff snapshot.

The analysis input is
`/data2/ishaangp/mira-interp/captures/observations_v2.npz`, 884,598,487 bytes,
SHA256 `b007c5476951bd064b3b7a30545dbe448e5f9c02eacf598d26cfcc9f72b734f2`.
Its gate is `results/capture_aggregate_v2.json`. It contains 13,568 rows with
`X[13568,17,2048]`, 30 targets, codec and RGB baselines, and complete row identities.
These are teacher-forced annotation-reconstruction features; target pixels are
present. Do not call this predictive future-state decoding.

## Analysis and publication

Selection, independent selection audit and confirmation **completed**. The frozen
block-5 readout scored 0.846902 standardized MSE versus the mean baseline's
1.060234 on 11 confirmation matches. All result exports and 91 completion checks
passed. See `docs/07_observational_results.md`; do not rerun or overwrite these
completed phases. The 1,138,648,642-byte derived-feature archive also passed every
member hash and array-equivalence check. GitHub upload and API size/digest
verification passed; see `results/observation_publication.json`.
The ordered reproduction commands are documented in
`docs/04_probes.md`; use the actual aggregate path above and output directory
`results/probes_v2`.

1. `scripts/analyze_probes.py --phase select`: discovery fitting and selection only.
2. `scripts/audit_probe_selection.py --selection-dir results/probes_v2`: independent
   saved-coefficient, normalization, selection-loss and provenance audit.
3. Review that audit, then `scripts/analyze_probes.py --phase confirm`: immutable
   held-out evaluation, no refitting or new choice of winner.
4. `scripts/export_probe_results.py`: format all existing metrics and figures.
5. `scripts/audit_observational_completion.py`: audit results, tests and claim limits.
6. `scripts/package_observations.py`: package every pooled derived feature and all
   17 full pilot residuals; exclude raw source labels/video. Verify the archive and
   GitHub asset digest, push code/results/figures/docs, then verify remote HEAD.

Never overwrite selection locks, confirmation outputs, frozen protocols, or
completed capture reports. The capture runner's fully cached path lacks fresh
loader metadata; audit completed workers through aggregation instead of rerunning
them. Partial capture resumption is only useful for a worker with unfinished clips.

All 46 software tests passed before fitting. These tests are engineering evidence,
not results of the research hypothesis.

## Later proposal gates

The discovery-only availability audit examined all 868 within-match pairs among
the 248 discovery clips and found **zero** identical all-player future-action
streams over the prescribed 0.75-second horizon. This is a result for the frozen
subset, not a claim about the complete Rocket Science dataset.

Controlled single-variable state-reset/rendered pairs and an independently
validated generated-video physical evaluator remain missing. The literature
review corrects the earlier stopping rule: internal activation patching can run
with recipient actions/noise fixed, and exploratory geometry/sparse features can
be studied with narrow claims. Their physical causal faithfulness must still be
measured. See `docs/08_review_and_repair.md`; the older prerequisite document
records the previous, stronger gate.
