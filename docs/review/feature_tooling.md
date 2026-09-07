# Development feature tooling

`scripts/analyze_feature_geometry.py` and `src/mira_interp/feature_geometry.py`
provide a bounded numerical pilot. They have been tested with synthetic engineering
fixtures only. No real-data geometry or dictionary experiment has run through this
tool, and the tooling does not constitute an official Goodfire BSF replication.

The command is deliberately gated on a separate review of completed probe and
causal evidence. This implements the requested execution order; it does not imply
that causal validation is mathematically necessary for observational geometry.
See [the methodological review](geometry_and_features.md) for that distinction.

## Input and gate

Prepare a new development-only NPZ with these arrays:

| Array | Shape / meaning |
| --- | --- |
| `X` | `[N,D]`, registered current mean (`D=2048`) or spatial descriptor (`D=1536`) |
| `velocity` | `[N,3]`, selected entity's world XYZ velocity, source units |
| `position` | `[N,3]`, required for the optional height comparison |
| `match_ids`, `split` | `[N]`, whole-match-disjoint `discovery` / `selection` only |
| `clip_ids`, `entity_ids` | `[N]`, nonempty stable identifiers |
| `view_index` | `[N]`, nonnegative integer camera identity |
| `frame_index` | `[N]`, consecutive latent-time index within each clip, not the original two-step source index |
| `timestamps` | `[N]`, that view's time in seconds |

All numeric arrays must be finite. Duplicate match/clip/entity/view/frame keys
are rejected. At least two independent matches in each role are needed. No further
projection or concatenated current-plus-delta descriptor is accepted by this pilot.
It checks split names
before loading feature/velocity arrays and refuses confirmation data. Adapt the
source artifact explicitly; this tool never rewrites a capture or target archive.

The selected-probe JSON needs a nonempty `descriptor_identity`, its `feature_dim`
(1536 or 2048), and `temporal_readout: "current"`. These are used to
identify the already chosen residual descriptor, not to fit another probe or to
claim retention of that probe's predictions. The following separate reviewer
record is required; placeholders below must be replaced by actual evidence:

```json
{
  "status": "passed_feature_development_prerequisites",
  "scope": "development_only",
  "physical_causality_claim": false,
  "causal_scope": "internal_model_output",
  "descriptor_identity": "same value as selected-probe JSON",
  "feature_dim": 2048,
  "temporal_readout": "current",
  "input_npz_sha256": "actual prepared input hash",
  "selected_probe_sha256": "actual selected-probe file hash",
  "probe_audit": {"path": "probe-audit.json", "sha256": "actual hash"},
  "causal_audit": {"path": "causal-audit.json", "sha256": "actual different hash"},
  "entity_mapping_verified": false
}
```

Evidence paths resolve relative to the gate file. Both distinct reports must be
unchanged and have status `passed`, `passed_development_probe_audit`, or
`passed_model_intervention_audit`. The gate is the separate review decision:
the CLI validates evidence integrity/status, not the scientific adequacy of those
experiments. No command here fabricates or automatically creates a passed gate.
Internal original-model-output or downstream-codec-probe effects are sufficient
for the `internal_model_output` scope. Independent generated-physics measurement
is not required for this exploratory run. A separately measured output may use
`independently_measured_generated_output`; the scope is retained in the report.

## Numerical methods

Geometry fits **physical value → every selected-descriptor coordinate**. Specify an
interior held-value interval before running. Discovery rows outside the closed
interval train the maps; selection rows inside it evaluate them. All other rows
are unused for this comparison. There must be at least two eligible matches in
each role, and fitting weights give each match equal mass.

`height`, `vx`, `vy`, `vz`, and horizontal `speed` compare affine and quadratic ridge maps.
`heading` requires an explicit positive minimum horizontal speed, discards slower
rows, wraps angles to `[-pi,pi)`, and compares first versus first-plus-second
sin/cos harmonics. It never fits a discontinuous raw-angle polynomial. The
ridge alpha is fixed, not selected on evaluation. Physical basis scaling and
ridge normalizers use fitting rows only. The report includes original-space
descriptor MSE, a discovery-mean baseline, its discovery-variance normalization,
per-match paired gains and descriptive 1,000-match-bootstrap intervals,
eligible counts, and excluded held-value counts. These are conditional
observational diagnostics; nuisance matching is an upstream responsibility. A
1536D spatial descriptor's geometry is not the geometry of the full unpooled
residual tensor. No new projection is fitted by the geometry analysis itself.

Sparse models use an untied linear encoder with bias, no decoder bias, and unit
L2 decoder-row norms. The compared models are ReLU Top-K, signed scalar absolute
Top-K, and signed block Top-K with block size eight. All have 512 latent scalar
coordinates and **2,097,664 parameters** at input dimension 2048, or
**1,573,376 parameters** at dimension 1536. Activity budgets
16 and 32 correspond to two and four active blocks for the block model. ReLU can
realize fewer nonzero coordinates; actual activity is reported. The signed scalar
and block adapters have now been compared against official VanillaBSF at pinned
commit `219f121ea82d2b19200d1dac918396e6058d7eb9`: matched weights give exactly
equal codes, outputs, loss gradients and normalized decoder atoms. See
[`bsf_equivalence.json`](../../results/bsf_equivalence.json). This is operator
equivalence, not reproduction of the paper's training outcomes. Grassmannian and
Group Lasso variants are not implemented here.

The default width512 is undercomplete at either descriptor dimension and serves
only as a smoke pilot. `--width` and `--active-budgets` are configurable, bounded
at8192 total and128 active coordinates, in multiples of eight. A prospective
research comparison can use `--width 4096 --active-budgets 32 64` for2048D,
or width3072 for1536D: a2× overcomplete dictionary and four/eight active blocks.
Use separate fresh output directories for each of three fixed seeds and a
registered training budget such as1000 updates. These settings have not been run.
Actual parameter counts and expansion ratios are written in every report.

Downstream reconstruction effects remain unimplemented. A later model-level test
must lift descriptor changes through the registered readout into the original
residual while retaining its nullspace, rather than replace every residual token
with a pooled reconstruction. The readout's orthonormal projection and spatial
bin weights determine that lift; its round-trip identity must be independently
checked before using it to claim behavioral fidelity.

Features are centered using equal-match-weighted discovery means and divided by
one global RMS. Unlike per-channel whitening, this preserves relative distances
and angles in the selected descriptor. Training samples matches uniformly and then
rows uniformly. A fixed seed gives all variants identical reconstruction batches;
temporal pair sampling has a separate RNG. The tool defaults to 200 Adam steps
and refuses more than 2,000, batch sizes over 1,024, or input sets over 50,000
rows. Reports contain parameter count, actual scalar/block activity, reconstruction
errors in normalized and raw coordinates, per-match errors, dead scalar/block
fractions, adjacent support Jaccard, timing, and losses every 100 updates.
This bounded run is not a convergence claim or an exactly matched reconstruction
comparison. Use the reported reconstruction/activity tradeoff.

Temporal regularization is optional and requires `entity_mapping_verified: true`
or an explicitly scoped `view_identity_verified: true` prototype. The latter is
reported as `view_identity_only` and never as tracked-entity evidence.
It compares soft block-support gates only for adjacent frames sharing match,
clip, entity, and view, with the expected time gap (default 0.1 s). Nonconsecutive
or mistimed pairs are excluded. The differentiable sigmoid-gate surrogate is a
soft approximation to support changes, not a penalty on block coordinates.
No verified links means failure. A fixed image patch or a view ID alone must not
be declared an entity track. `--temporal-shuffle-control` adds equal-weight
regularization with shuffled times within the same match, clip, annotation entity
and view, excluding self and the original next frame. For whole-view
descriptors, source entity identity can be verified while its visual location and
visibility remain unlocalized: such an experiment must be named a whole-view
temporal prototype, and does not fulfill the proposed tracked-entity extension.
Tracking and generic-smoothing controls remain necessary for that stronger claim.

## Example after real prerequisites pass

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/analyze_feature_geometry.py \
  --protocol configs/feature_development_v1.json \
  --input /data2/ishaangp/mira-interp/features/geometry_development_v1.npz \
  --selected-probe results/geometry_development_v1/selected_probe.json \
  --gate results/geometry_development_v1/gate.json \
  --output-dir results/feature_development_v1/geometry_speed \
  --stage geometry --variable speed --heldout-interval 800 1200 \
  --ridge-alpha 0.01 --device cpu
```

The command must match the finalized protocol exactly. Geometry and sparse
training are separate stages; `--stage both` is refused. The protocol binds the
input, selected descriptor, reviewed gate, complete analysis/export/probe code,
and BSF equivalence report. Both analysis and export use eight BLAS threads and
rehash registered evidence and code after execution.

After geometry audit and the full-width synthetic training pilot pass, the
separately authorized sparse stage for one registered seed is:

```bash
CUDA_VISIBLE_DEVICES="${MIRA_AUTHORIZED_GPU:?Set the allocated physical GPU index}" \
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/analyze_feature_geometry.py \
  --protocol configs/feature_development_v1.json \
  --input /data2/ishaangp/mira-interp/features/geometry_development_v1.npz \
  --selected-probe results/geometry_development_v1/selected_probe.json \
  --gate results/geometry_development_v1/gate.json \
  --output-dir results/feature_development_v1/sparse_seed0 \
  --stage sparse --width 3072 --active-budgets 32 64 --steps 1000 \
  --seed 0 --temporal-weight 0.1 --temporal-shuffle-control --device cuda
```

The caller must recheck the allocated GPU before running; the tool does not
select or reserve GPUs. The registered seeds are0,1,2, each using a new output
directory. Reports contain hashed model checkpoints, discovery-only normalizers
and configuration. Sparse parameters and the CUDA device setting must match
the protocol; examples do not themselves authorize execution.

Nine focused tests verify held-value exclusion, periodic geometry, match and
identity guards, signed versus nonnegative activity, equal parameter counts,
temporal correspondence, finite temporal gradients, discovery-only fitting, and
evidence integrity, registered descriptor dimensions, and exclusion of temporal
concatenations. These tests establish software behavior, not a research result.
Additional generated-output evaluation, feature selectivity/complement checks,
multiple training seeds, and fresh confirmation remain necessary for stronger claims.
