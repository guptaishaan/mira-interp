# Sparse dictionary causal fidelity: prepared protocol

**Status: registered; reserved GPU pilot and its independent audit passed;
full 22-pair evaluation is running.** All 30 trained dictionaries passed their
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
The reserved pilot completed all 62 conditions with exit0, 29.43 seconds after
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
The full study subsequently started on GPU7 in durable tmux session
`mira_sparse_fidelity_full`; `evaluate.log` and `evaluate.exit` record progress
and process completion. Full scientific comparisons require the final output
and its independent audit, not the presence of a running session.
