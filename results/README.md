# Results

For the small set of results shown in the root README, start with [progress/](progress/).

| Work | Reports and tables |
| --- | --- |
| Data/model checks | `pilot_*`, `pretrained_*`, `capture_*`, `data_*` |
| Original decoding | [probes_v2/](probes_v2/) |
| Revised decoding | [development_probes_v3/](development_probes_v3/) |
| New-match confirmation | [fresh_confirmation_v3/](fresh_confirmation_v3/) |
| Literature and numerical diagnosis | [review/](review/) |
| Exact interventions | [causal_development_v1/](causal_development_v1/) |
| Geometry and sparse features | [feature_development_v1/](feature_development_v1/) |
| Conditional causal fidelity | [sparse_causal_fidelity_v1/](sparse_causal_fidelity_v1/) |
| Video evaluator | [videomae_exports/](videomae_exports/), [videomae_repair_exports/](videomae_repair_exports/), and `videomae_*` reports |
| Generated rollouts | [rollout_steering_v2/](rollout_steering_v2/) and [generated_evaluation_v2/](generated_evaluation_v2/) |

JSON files contain results, configuration bindings, and audit records. CSV files
contain plotting/analysis tables. Logs and failed attempts are retained with their
stage so changes can be traced. Historical paths stay stable because registered
experiments and release manifests refer to them.

Large tensor/checkpoint artifacts are attached to the [GitHub releases](https://github.com/guptaishaan/mira-interp/releases).
Local `*_publication.json` reports record the verified remote file hashes.
