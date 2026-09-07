# Experiment notes

Start with [initial progress](progress/README.md). The root README shows two
figures from the decoding work; the remaining experiments and checks are here.

## Data and model

- [Setup](00_setup.md), [data alignment](01_data.md), and [cohort rules](05_cohort_amendment.md)
- [Model access](model_access_audit.md), [pretrained setup](03_pretrained.md), and [instrumentation](02_instrumentation.md)
- [Capture](04_capture.md), [aggregation](05_aggregate.md), and [protocol](PROTOCOL.md)

## Decoding

- [Original observational results](07_observational_results.md)
- [Revised readout comparison](10_development_readout_results.md)
- [New-match confirmation](11_fresh_confirmation_results.md)

## Literature and diagnosis

- [Review and repair](08_review_and_repair.md)
- [What went wrong and which changes helped](13_diagnosis_and_repairs.md)
- [Probes and causality](review/probes_and_causality.md), [geometry and sparse features](review/geometry_and_features.md), and [Anthropic methods](review/anthropic_and_model_audit.md)
- Additional source reviews: [first 20 remaining pages](review/anthropic_extended_review_a.md) and [last 20](review/anthropic_extended_review_b.md)

## Exploratory causal work

- [Exact activation interventions](09_internal_causal_development.md)
- [Geometry comparisons](review/geometry_development_results.md)
- [Sparse-feature comparisons](review/sparse_development_audit.md)
- [Conditional causal fidelity](review/sparse_causal_fidelity.md)
- [Video evaluator](review/video_evaluator.md)
- [Generated-rollout protocol](review/rollout_steering.md) and [measurement](review/generated_measurements.md)
- [Audited generated selection results](review/generated_selection_results.md)

## Reproduction and artifacts

- [Figure and table exports](review/generated_measurement_exports.md)
- [Generated artifact format](13_generated_publication.md)
- [Parallel archive publication](14_parallel_publication.md) and [final execution/publication check](review/registered_rollout_completion_audit.md)
- [GitHub releases](https://github.com/guptaishaan/mira-interp/releases)
- [Execution handoff](EXECUTION_HANDOFF.md)
- [Archived project log](archive/project_log_before_readme_simplification.md)

Code lives in `src/` and `scripts/`, settings in `configs/`, numerical reports
and audits in `results/`, and plots in `figures/`. The small progress subset has
its own `docs/progress/`, `results/progress/`, and `figures/progress/` folders.
Detailed historical paths remain stable so registered hashes and references
continue to identify the original evidence.
