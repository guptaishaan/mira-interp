# Independent generated-video measurement audit

The frozen evaluator estimates all 12 role coordinates for four camera views at four generated endpoints. Absolute ball coordinates are the sum of the ego and ball-minus-ego coordinates. These remain learned video estimates; no generated-video 3D accuracy has been established.

The producer's metric code was independently reviewed and tested on complete synthetic dose cohorts before generated outcomes were read. Two defects were repaired before registration: contradictory responses at duplicate effective doses could count as nondecreasing, and dropping undefined rank seeds could change match weights. The final 12 tests also cover constant/negative responses, clipped doses, view/time indices, all collateral coordinates, paired random/wrong-variable controls, and whole-match bootstrap calculations. The source-bound review is `results/generated_evaluation_code_review.json`.

The separate CPU auditor does not call the producer's summary function:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/audit_generated_measurements.py --phase pilot
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/audit_generated_measurements.py --phase selection
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/audit_generated_measurements.py --phase confirmation
```

Each phase verifies the prior generation audit, registered measurement source, frozen evaluator checkpoint and target scales, every generated-record/source/seed identity, individual prediction caches, and the combined saved predictions. The pilot verifies cache consistency and the producer's repeated-inference control; it performs no new model inference. Later phases independently reconstruct all paired dose curves, tie-aware ranks, clipping consistency, slopes, collateral effects, extreme-dose contrasts, and 1,000-draw whole-match bootstrap intervals. Both random and wrong-variable comparisons use the same matches and paired seeds.

The auditor's separate five synthetic tests compare its independent formulas with the reviewed producer on positive, mixed-sign, constant, and clipped cohorts. They passed before any real generated measurements were audited. Passed output has status `passed_generated_measurement_audit` and binds `evaluation_sha256`; this is the required gate for the next measurement phase.

Per-observation source simulator labels are not loaded by this auditor. It validates generated artifact hashes before and after measurement checks, and reads frozen calibration metadata/checkpoint scales. Bootstrap intervals describe match sampling, not uncertainty in the video evaluator or simultaneous guarantees across paths and times. A completed measurement audit verifies computation, not physical steering success.

The generated-measurement pilot audit passed both records. Its report is `results/generated_evaluation_v1/pilot_audit.json`, SHA256 `666a67ddc7c92b33a73c8925a7730b0d70cfb4312a6d27c8dd4d5138dfe0abed`, bound to evaluation SHA256 `565703ca9416d7a288cabf02961f0b8776ac9e5e42fa5c0aa4e7131e0b2e90ac`. Source/cache joins, all four views and endpoint axes, absolute ball conversion, and frozen target scales were exact. The audit took 0.248 seconds after the producer completed its two-record measurement pass. This is an engineering gate, not a physical-steering result.
