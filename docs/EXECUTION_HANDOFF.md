# Execution handoff

Execute the proposal in order, audit each prerequisite, document key steps, and
publish to `https://github.com/guptaishaan/mira-interp.git`. The user authenticated
Hugging Face and GitHub and authorized publishing the work. Continue within this
scope without asking again.

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

Inspect `results/probes_v2/` and actual process status before resuming. Selection
was launched after aggregate review. The ordered commands are documented in
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
validated generated-video physical evaluator remain missing. Exact activation
patching software is tested, but physical causal identification has not run.
Do not advance to geometry, SAE/BSF recovery, or steering merely because an
observational probe decodes annotations. See `docs/06_causal_prerequisites.md`.
