# Sparse dictionary conditional causal fidelity

**Status: complete and independently audited: 22 paired-seed examples from
11 matches, 30 dictionaries, and 1364 dictionary/control conditions.** All 30 trained dictionaries passed their
independent reconstruction audit before registration. This extends the completed internal causal test in
[docs09](../09_internal_causal_development.md), whose original discovery ranking
fixed block 15 output. It does not establish physical trajectory control.

The evaluator accepts all 30 dictionaries: three training seeds, active-coordinate
budgets 32/64, and ReLU Top-K, signed Top-K, block, temporal-view block, and
shuffled-view temporal block variants. Each dictionary has 1536 input coordinates
and 3072 latent scalar coordinates. No dictionary is selected using its causal
effects. Each checkpoint retains its discovery mean and **one global RMS**;
there is no new fitting or channel whitening.

The raw residual support is view 0, latent 7, a 9×16×2048 tile at block 15 output.
The fixed descriptor averages 12 spatial bins (3×4 bins, each with3×4 tokens),
then projects each bin mean with the original orthonormal 2048×128 matrix Q.
For descriptor change δ, the minimum-norm lift broadcasts `δ_bin @ Q.T` to every
token in that bin. There is **no factor 1/12** in this inverse: averaging the 12
identical offsets already gives the desired mean change. The derivative
pullback used in the previous directional intervention has a different purpose
and does divide by 12. The new tile is `native + lift(reconstruction - descriptor(native))`.
This preserves native information in the descriptor nullspace up to floating
point roundoff; it does not replace the raw tile with a 1536D approximation.
The native tile has 294,912 scalar coordinates. Information outside the 1536D
descriptor can therefore preserve the donor effect even when descriptor
reconstruction is poor. This measures **conditional reconstruction fidelity**,
not dictionary sufficiency or independent recovery of the causal feature.

For each original selection match and its two registered noise seeds, evaluate:

1. Reconstructed recipient descriptor lifted onto the native recipient tile,
   compared with the original recipient output.
2. Reconstructed donor descriptor lifted onto the native donor tile, inserted
   into the recipient computation, compared with the original full-donor patch.

Two additional shared controls replace the recipient and donor descriptors by
the identical frozen discovery mean while retaining their respective native
nullspaces. Any remaining difference between these controlled tiles is entirely
outside the descriptor (up to rounding). This reveals how much donor effect can
survive without descriptor deviations from the mean. Each dictionary is compared
with its paired mean control, including absolute deviation from the original
native-reference gain. These controls do not turn conditional fidelity into
dictionary sufficiency.

These are 22 fixed recipient/donor/seed pairs, 1320 dictionary-condition
forwards and 44 shared-control forwards (1364 total). All recipient pixels,
actions, noise, past latents and tau remain fixed. Natural donors differ in
several physical properties; they are not
single-variable counterfactuals. Dictionary inference uses CPU FP32 weights
and the saved FP64 normalizer. The native descriptors are FP32, whereas training
descriptors were stored FP16; that precision difference is recorded explicitly.

The registration hashes all three reports, all 30 checkpoints, the independent
sparse audit, the original causal registration/results/audit, and execution
code. A separate reserved-pilot phase evaluates every dictionary and both mean
controls. Its outputs are independently audited before full evaluation.
Native recipient and full-donor outputs, native residuals, and an
identity descriptor lift must replay bitwise. Full evaluation requires that
pilot. Completed stages are immutable; interrupted runs resume only hash-bound
complete pairs.

The primary output proxy remains the original frozen clean-codec position
probe applied to the model endpoint `z_tau + (1-tau)*v_pred`. Its objective is
negative squared donor ball-height error, normalized by discovery variance.
The report saves the standardized error gain, its difference from the original
reference effect, all 15 position-proxy changes, and native full-flow/endpoint
distances. Raw error differences replace potentially unstable recovery ratios.
The full model flow tensors and descriptor reconstructions are retained so
these metrics can be independently recomputed. Fidelity archives exclude raw
source labels, actions and video frames.

Both noise seeds are averaged within each of 11 matches before 500 whole-match
bootstrap draws. Intervals are descriptive and conditional on the development
choices. The 11 matches were previously used for probe selection and causal
development; this is not fresh confirmation. Successful execution and low
reconstruction error are not evidence that physical state or generated-video
behavior has been recovered. The codec proxy's distribution shift on edited
model endpoints remains unvalidated, and the experiment uses one teacher-forced
flow evaluation without a generated rollout.

The command interface is:

```bash
NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src:external/mira/src \
  .venv/bin/python scripts/sparse_causal_fidelity.py --phase register \
  --feature-report <seed0>/report.json <seed1>/report.json <seed2>/report.json \
  --feature-audit results/feature_development_v1/sparse_audit.json
```

After the training audit and registration are reviewed, the same arguments with
`--phase pilot` run on one explicitly assigned GPU. After the pilot passes and
is reviewed, `--phase evaluate` runs the fixed 22 pairs. GPU ownership must be
checked immediately before either GPU phase. No command above is a completed
experiment record.

The protocol was registered with SHA256
`65f269ca1352d6e1c3967b463d0926012c0da28bbf216e5a988aaf8923282de1`.
The reserved pilot completed all 62 conditions with exit 0, 29.43 seconds after
loading and final checks, and 10.52 GiB peak allocated GPU memory. Its native
baseline, native donor and zero-lift replay controls were bitwise exact. The
maximum descriptor round-trip absolute error was 1.67×10⁻⁶. Its 18.87 MiB
private archive contains model outputs rather than raw source annotations.
These are engineering checks on the reserved self-donor clip, not a scientific
effect estimate. Evidence is in `results/sparse_causal_fidelity_v1/pilot.json`
and its bound pair report/tensor archive.

The independent pilot audit passed all 62 conditions in 5.58 seconds. Its
maximum independently reconstructed descriptor difference was 4.77×10⁻⁷ and
maximum recomputed objective-gain difference was 4.86×10⁻¹⁶. The audit SHA256 is
`4e05a827d4b548091919d6c76bcbb0e6c39324ff188fdad6201e1517d9478dc9`.
The full study ran on GPU7 in durable tmux session `mira_sparse_fidelity_full`
and completed with exit 0 in 573.38 seconds after loading. Peak allocated GPU
memory remained 10.52 GiB. `evaluate.log` and `evaluate.exit` retain the execution
record. The independent full audit passed all 1364 conditions, all source
time/entity joins, 62 match summaries and 60 paired mean-control comparisons in
52.71 seconds. Maximum independently reconstructed descriptor difference was
8.34×10⁻⁷; maximum objective-gain difference was 7.11×10⁻¹⁵. Native reference
replays were exact, and earlier flow frames remained bitwise unchanged.

The full result SHA256 is
`0c6787a0e1e4ffc1fb16824dbba8af8d65f43d14e4862043bec1b0a612a1a62c`;
the independent audit SHA256 is
`321059bc7bf6ec017a3b1f5c9edcd3a295294872d0d594a6e2de8bb05a4b71f2`.

## What the mean controls show

The dictionaries improve conditional reconstruction, but most of the average
full-donor effect survives even when the entire descriptor is replaced by its
discovery mean. The original full-donor standardized error gain is 0.682834.
The mean-descriptor donor control gives 0.672032; its difference from the native
effect is −0.010802, with descriptive 95% interval [−0.035499,0.006261]. Its two-seed
mean is positive in 6/11 matches, exactly as for the original full donor.

| Donor reconstruction measure | Shared discovery-mean control | Range across all 30 dictionaries |
| --- | ---: | ---: |
| Standardized donor-error gain | 0.672032 | 0.678243–0.685742 |
| Difference from native full-donor gain | −0.010802 | −0.004592–0.002908 |
| Full-flow L2 distance from native donor output | 3.50649 | 2.59503–2.90289 |
| Descriptor reconstruction RMSE | 0.535831 | 0.408295–0.448925 |
| Matches with positive donor-error gain | 6/11 | 6/11 for every dictionary |

Relative to the mean control, dictionary mean donor-objective gains increase
by 0.006211–0.013710 standardized units. Their paired improvement in absolute
deviation from the native gain is 0.005917–0.012570. The match intervals are
wide: many include zero. All per-model intervals and per-match values are
retained, and there is no multiplicity-adjusted claim or outcome-selected
dictionary winner. Improved descriptor reconstruction is therefore accompanied
by a modest improvement in conditional output fidelity; it does not explain
the large original donor effect by itself.

For recipient reconstruction, the shared mean descriptor changes the donor
objective by +0.004890[−0.006915,0.019669] relative to the native recipient.
Dictionary changes range from −0.001940 to +0.008647. They reduce full-flow
distance from 3.70940 for the mean control to 2.75949–3.17981, and descriptor
RMSE from 0.566418 to 0.441285–0.493901. Recipient-objective changes are not a
physical-accuracy test: the objective still targets the natural donor's height.

Both PNG/PDF figure pairs, `donor_reconstruction_fidelity` and
`recipient_reconstruction_fidelity`, show every dictionary, training seed,
matched activity budget, and mean control. The shaded intervals and individual
point intervals are descriptive whole-match bootstraps. No observations or
seed variants are dropped. Nominal capacity/activity budgets are matched;
achieved activity and reconstruction error need not be equal across families.

The conclusion is **conditional fidelity, not feature sufficiency**. A native
294,912-coordinate residual tile retains a large complement outside the
1536-coordinate descriptor. The mean controls directly show that this retained
information can support most of the average original transfer. Neither this
study nor the earlier full-donor effect establishes generated physical-state
control, reliable success across matches, or a recovered nonlinear manifold.
