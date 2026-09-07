# Causal experiment prerequisites

The frozen discovery clips contain **no pairs with exactly equal future actions
for all four players**. This audit does not establish controlled physical
counterfactuals, and the causal-identification gate remains unmet.

| Audited quantity | Result |
| --- | ---: |
| Qualified discovery matches | 31 |
| Clips per match | 8 |
| Discovery clips | 248 |
| Exhaustive within-match clip pairs | 868 |
| Exact four-player future-action matches | 0 |
| Action-matched pairs changing exactly one initial coordinate | 0 |

The machine-readable result is
[controlled_pair_feasibility.json](../results/controlled_pair_feasibility.json).
Run the availability audit with:

```bash
python scripts/audit_controlled_pairs.py
```

The script reads only discovery NPZ files, rechecks their content hashes against
the independently passed qualified-data gate, and checks all `8 choose 2` pairs
within each match. Player IDs and their order must match. It compares every one
of the nine keys for all four players at source frames 1 through 15, preserving
20 FPS timing. Thus the checked future horizon is 0.75 seconds, with no action
pooling, approximate action matching, or identity remapping.

For an exact action match, the planned state check compares the 30 ball/player
position and velocity coordinates at frame 0 in view 0. It requires exactly one
coordinate to differ and every other coordinate to be exactly equal. View 0
provides one internally aligned observation; it does not turn the four views
into an exactly synchronized simulator snapshot. Because no actions matched,
no pair reached the state-isolation criterion. The report exports aggregate
counts and provenance hashes, not selected pairs or physical label values.

The search is exhaustive only within these 248 frozen discovery clips. It says
nothing about the availability of pairs elsewhere in Rocket Science. Expanding
the search would need a separately documented discovery-only sampling protocol;
selection and confirmation clips must not become a source of new pair-design
choices.

Even a pair passing both numerical checks would remain observational. Thirty
coordinates omit rotation, angular velocity, boost, contact state, camera
history, other environmental variables, and hidden simulator state. Equal
recorded controls and nearly identical states do not demonstrate that only one
physical variable was independently changed.

Before strong physical causal claims, the project still needs:

1. A reproducible way to construct isolated physical-state changes and render
   their contexts, or a narrower observational intervention design with explicit
   confound accounting. The released dataset/model interfaces have not provided
   a verified simulator state-reset and counterfactual renderer.
2. An independent video-to-state measurement system with adequate errors on
   both real and generated clips. An internal linear probe alone cannot establish
   that a generated ball or player physically moved as intended.
3. Exact activation interventions with all four future action streams, noise
   tensors, sampler, and context controlled, followed by the registered random,
   wrong-player, wrong-variable, ablation, and restoration tests.

No attribution result, physical activation-patching result, controlled geometry
sweep, sparse-feature recovery result, or steering result is supplied by this
availability audit. The separately registered observational probes test
decodability; they cannot satisfy these causal prerequisites.
