# Independent sparse causal fidelity audit

This experiment reuses the original selected site, 11 selection matches, and two paired random seeds. Every one of the 30 frozen dictionaries receives both a recipient-reconstruction and a donor-reconstruction intervention. Two additional controls replace the respective descriptors with the same frozen discovery mean. There are 62 conditions per pair, or 1,364 conditions in the full development experiment. The unchanged native activation complement is retained in every reconstruction; this remains conditional fidelity, not feature sufficiency.

The reserved pilot must pass an independent audit before the full run:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/audit_sparse_causal_fidelity.py --phase pilot
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/audit_sparse_causal_fidelity.py --phase full
```

The auditor runs on CPU. It checks the exact registration, training audit, all 30 checkpoint hashes, original causal pairs, own-view final-frame source labels/timestamps/player identities, and all saved FP32 arrays. It compares baseline, donor, interpolant, codec, and native tile arrays exactly against the prior audited experiment. It independently recomputes dictionary encoding and synthesis with NumPy, including both discovery-mean reconstructions.

The spatial intervention is checked by independently expanding each descriptor offset into its 3-by-4 spatial bin and applying the fixed channel projection transpose. There is no division by 12: spatial averaging and this broadcast lift are inverse operations on the selected descriptor coordinates. Replacement tiles are not separately saved in this experiment; the audit recomputes the expected lift and its norm, and checks the declared FP32 roundtrip bound. Full modified model outputs are saved for every condition.

For each output, the auditor reconstructs the flow endpoint, applies the frozen clean-codec position probe, and recomputes the donor-height objective, collateral proxy changes, output distances, and source-codec reconstruction error. The first seven latent-frame outputs must remain bitwise equal to baseline. No new model inference or fitting occurs in the audit.

The full audit separately recomputes each condition's paired-seed, whole-match summaries and all 60 dictionary-versus-mean comparisons. It distinguishes increased donor-objective gain from improved agreement with the native gain: overshooting a reference can improve the first while worsening the second. Bootstrap intervals use the registered 500 whole-match draws and remain descriptive development uncertainty.

The reserved pilot audit passed all 62 conditions in 5.582 seconds. Its report is `results/sparse_causal_fidelity_v1/pilot_audit.json`, SHA256 `4e05a827d4b548091919d6c76bcbb0e6c39324ff188fdad6201e1517d9478dc9`. Full-run results and independent verification are required before interpreting the development study. These output-probe effects do not establish generated physical motion or control.

The full development audit also passed: all 1,364 conditions, 22 paired runs, 62 condition summaries, and 60 dictionary-versus-mean comparisons were independently checked in 52.710 seconds. The report is `results/sparse_causal_fidelity_v1/fidelity_audit.json`, SHA256 `321059bc7bf6ec017a3b1f5c9edcd3a295294872d0d594a6e2de8bb05a4b71f2`. This verifies the reported effects; interpretation and plots remain tied to the internal output-proxy scope.
