# Quality-qualified observational cohort

The original 61-match data gate failed. The complete audit produced eight valid
clips from each of 53 matches, and rejected eight matches at source-clock
calibration. This original failure is preserved in
[`results/cohort_data_audit.json`](../results/cohort_data_audit.json).

Before any research activation capture or probe fitting, a separate version 2
protocol was frozen at **2026-09-07 00:05:44 UTC**. It includes exactly all 53
fully passing matches, with their original roles. No match is replaced, no role
is reassigned, and no threshold or clip is changed.

| Original role | Original matches | Excluded | Qualified matches | Clips |
| --- | ---: | ---: | ---: | ---: |
| Discovery | 37 | 6 | 31 | 248 |
| Selection | 12 | 1 | 11 | 88 |
| Confirmation | 12 | 1 | 11 | 88 |
| Total | 61 | 8 | 53 | 424 |

The separately reserved engineering pilot remains excluded from every research
role. Meeting the operational minimum of 30/10/10 matches does not establish
statistical power; uncertainty is reported by resampling independent matches.

## Why the matches failed

Seven matches had inconsistent offsets between source score changes and goal
event times. Several showed a score update delayed by about 13–14 seconds; one
showed smaller clock shifts of several tenths of a second. The eighth match had
only two usable goals, below the fixed minimum of three. That last exclusion is
insufficient calibration evidence, not proof that its recording is corrupt.

The unchanged gate requires at least three paired goals in every view and a
maximum absolute residual of 0.1 seconds after the median clock correction.
The [discrete-event diagnostics](../results/source_clock_failures.json) record
the evidence without fitting positions, velocities, activations, or probe scores.

The new protocol estimates decoding performance only for this quality-qualified
population. Excluding 13.1% of the original matches can favor recordings with
more goals and more consistent client clocks. Frozen-state and demolition
filtering further restricts the clips to eligible live play. Results cannot be
extrapolated to all Rocket Science recordings without a separate study.

## What the alignment evidence supports

Each view retains its own frame, action, and state index. Video frame counts,
presentation timestamps, player identities, structural validity, and approximate
cross-view ball agreement are checked. At a predetermined pilot frame, all four
visible boost HUD values agree with their source annotations after display
truncation; team colors and scores also agree. See the
[pixel/state spot check](../results/pilot_pixel_state_spotcheck.json).

These checks support the row correspondence but do not independently prove zero
latency between every video frame and its 3D state. The publisher's one-to-one
export contract remains an assumption. The study therefore measures decodability
of contemporaneously indexed dataset annotations. It does not establish accurate
future-state prediction, exact synchronization across views, or causal control.

## Frozen artifacts and next gate

- [Version 2 protocol](../configs/observational_probe_v2.json)
- [Registration and parent hashes](../results/observational_registration_v2.json)
- [Qualified split manifest](../data/qualified_split_manifest.json)
- [Registration script](../scripts/register_qualified_cohort.py)

The new protocol SHA-256 is
`e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2`.
The original version 1 artifacts remain immutable. Independent qualification
must verify all 424 source clip hashes and exact cohort membership before the
version 2 model pilot. A passing pilot under the same capture code and protocol
is then required before full capture. Selection must be frozen before confirmation.
