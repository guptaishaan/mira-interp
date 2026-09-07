# Data preparation and access

**All 11 official test shards are downloaded and verified. The original 61-match
cohort failed its clock-quality gate; a separately registered 53-match observational
subset passed independent validation of all 424 clips.** Original failures remain
in [cohort_data_audit.json](../results/cohort_data_audit.json). The qualified result
is [qualified_data_audit.json](../results/qualified_data_audit.json), with 31
discovery, 11 selection, and 11 confirmation matches. The unchanged engineering
pilot also passed all eight clips.
The verified download is [dataset_download.json](../results/dataset_download.json),
and the engineering gate is [pilot_data_audit.json](../results/pilot_data_audit.json).
These are data and software results, not evidence of physical representations.

The first pilot uncovered an approximately 8.133-second trim origin missing from
the index's recording offsets. Without that correction, the event mask admitted
frozen replay clips. Source-clock calibration now uses at least three unambiguous
scoreboard increments paired with their corresponding goal anchors, with maximum
residual 0.1 seconds. Nine pilot goals per view passed, with worst residual below
0.039 seconds. Structural exclusions occur before uniformly selecting eight live
windows. The original failures are retained in
[the uncorrected-clock audit](../results/pilot_data_audit_uncorrected_clock.json)
and [the pre-filter audit](../results/pilot_data_audit_pre_structural_filter.json).

Public metadata resolved Rocket Science to revision
`a248fc918baa93389e3242fefba193f1bfb0474a`. The metadata lists indices of 1,801,403
bytes for test, 6,010,559 bytes for dev, and 224,238,737 bytes for train. The
smallest listed test video shard is 2,562,437,120 bytes; downloading an entire
split without a budget would be inappropriate. The script initially requests
only the test index, pins the exact study revision, caps total raw payload at 3 GiB, and
requires at least 25 GiB free after that download. It does not accept terms or
write authentication material. Error text is reduced to exception type and HTTP
status to avoid saving signed URLs or credentials.

## Run and resume

To recheck authorized access without overwriting the frozen research allocation:

```bash
python scripts/prepare_data.py --manifest data/index_only_split_manifest.json
```

The command writes `results/data_access.json`. On success it also writes the
specified index-only split manifest, with the input hash and exact dataset revision.
Its success means **index preparation only**: video shards, physical labels,
live-gameplay checks, and matched trajectories remain unprepared. A blocked
access check or empty partition exits with code 2; malformed input raises an
error. To prepare a previously downloaded index without a network call:

```bash
python scripts/prepare_data.py --index /path/to/index.json --revision DATASET_COMMIT_SHA --manifest data/index_only_split_manifest.json
```

Multiple `--index` arguments may be supplied, but duplicate match IDs across
indices are rejected. The manifest preserves the full original index path and
upstream partition for every match. The Hub path establishes its partition;
for local files, pass one `--index-upstream-split` per `--index` to declare it.
Without that declaration local partition provenance is explicitly unknown, never
inferred from a folder name. Upstream `train/dev/test` are model-training partitions;
our discovery/selection/confirmation assignment is a separate experimental
partition. Even an upstream test match is not evidence of being absent from the
world model's training set unless the checkpoint provenance establishes that.

To download one initial shard after authentication:

```bash
python scripts/prepare_data.py --download-one-shard --data-dir /path/to/raw-data --manifest data/index_only_split_manifest.json
```

The selected directory can be on a suitable `/data2` volume. The script chooses
the smallest shard referenced by the pinned index and verifies its byte count
and SHA-256 against Hub LFS metadata. It inventories the tar read-only, rejects
unsafe paths and special files, and checks that every indexed chunk has all four
videos, action tracks, and physics tracks. It never extracts tar members. A
`downloaded_shard_index.json` beside the shard describes only downloaded matches;
the split manifest still describes the full requested index. Inventory success
does not check pixels, JSONL content, physics alignment, or live-gameplay status.
The reports keep `research_dataset_ready: false` until those gates are implemented
and passed; they cannot authorize a research stage by themselves.

## Preparation contract

1. **Assign entire matches once.** A seeded SHA-256 hash of the match ID gives
   fixed 60/20/20 discovery/selection/confirmation partitions. Input ordering and
   adding later matches cannot change earlier assignments. Every chunk, frame,
   and player perspective inherits its match's assignment. Hashing provides
   approximate proportions, not guaranteed sample counts. Empty partitions fail;
   sufficient independent matches and power are additional experimental gates.
   Supply `--pilot-match-ids pilot_ids.json` with an explicit JSON list of match
   IDs selected before research analysis to reserve the engineering pilot. These
   matches are assigned `pilot` and excluded from all three research counts. The
   same list must be passed to artifact writing. Without it, the manifest marks
   the pilot reservation incomplete; it is not a research-ready split.
   **Actual cohort amendment:** the initial 62-match test index produced only six
   confirmation matches under these approximate thresholds. Before research
   labels or activations were inspected, we reserved one pilot and assigned the
   remaining 61 matches by a frozen SHA-256 ordering to 37 discovery, 12 selection,
   and 12 confirmation matches. The old assignment is retained in
   `data/split_manifest_threshold_superseded.json`; the authoritative allocation is
   `data/split_manifest.json`, cohort hash
   `b9754a876c5d54f22c9a38e60b1361daebd4bed13634ab961722a54e99f357bf`.
   Preparation and capture read this manifest directly and never recompute roles
   with the original approximate-split helper.
   The manifest preserves original `chunk_indices`, because source chunks may
   be noncontiguous. Replacing them with list positions silently fetches the
   wrong videos.
2. **Preserve identities and physical units.** `physics_targets` emits 30
   columns: ball and four explicitly identified players, each with three position
   and three velocity coordinates. Every frame is reordered by persistent
   `player_id`, never by its car-list position. Positions are Unreal units and
   velocities are Unreal units per second. `player0` through `player3` denote
   the explicit within-match ID ordering, not universal identities across matches.
   Any team, role, or camera-relative remapping must be declared before analysis.
3. **Fit normalization on discovery only.** `PhysicsNormalizer.fit` refuses
   selection and confirmation rows. Mean and standard deviation are persisted;
   constant columns get scale one. Raw-unit prediction errors remain the main
   interpretable metric. This normalizer covers labels only; hidden-state
   standardization must independently enforce the same discovery-only rule.
4. **Keep source-rate future actions.** All four players' nine-key tracks must
   be compared exactly across the full future horizon with the same player
   ordering, source sampling rate, and relative future time grid. Different
   absolute source-frame origins are allowed; they need not be equal across
   separate trajectories or matches. Time grids are compared within 1 nanosecond
   and source rates within 1e-9 FPS. The upstream model loader can OR-pool multiple action frames;
   pooled equality can conceal unequal source actions. `audit_future_actions`
   therefore accepts only full four-player source-rate tensors and records
   mismatched key/frame entries and identity order. It cannot establish that
   an input really came from an unpooled source: source provenance and frame
   alignment must also be checked. Cross-match identity/role correspondence
   requires a separate protocol and is intentionally rejected by the basic audit.
5. **Keep observational pairing distinct from control.** Even exact future
   action equality does not hold other physics, camera history, random effects,
   or hidden simulator state fixed. The audit always marks
   `controlled_counterfactual: false`. A claim that only one physical variable
   changed needs simulator-generated matched trajectories and a verified state
   intervention, or must be narrowed to an observational model intervention.

`velocity_heading` represents horizontal velocity heading by cosine and sine,
and masks speeds below a declared threshold (default 1 uu/s). Stationary states
have undefined heading. Velocity heading is not the car's orientation quaternion.

## Prepared artifact format

`write_prepared_clip` accepts a single contiguous chunk window at source rate and
writes an NPZ plus adjacent JSON manifest. It rejects missing players, nonfinite
physics, nonbinary actions, and mismatched or noncontiguous source-frame indices.
The NPZ contains `targets [T,30]`, `source_actions [4,T,9]`,
`source_frame_indices [T]`, `player_ids [4]`, and `target_names [30]`.
The JSON records revision, match, source chunk, split, action vocabulary, units,
array SHA-256, and explicit video-alignment/live-gameplay audit flags.

This is a reusable label/action format, not a decoded-video loader. Before
capture, separately verify the video's frame count and FPS, source action/physics
alignment, perspective/world-state agreement, game-event windows, frozen physics,
demolitions, and camera/perspective mapping using the upstream loader and audit
helpers. Post-goal celebrations and replays may show frozen simulator physics;
they must not silently enter live trajectory analysis. Global source indices
must be derived from upstream metadata, not inferred by summing only the
surviving chunks. Never set an audit flag solely because arrays have equal shape.

## Validation performed

```bash
PYTHONPATH=src python -m pytest -q tests/test_data.py
```

Sixteen tests passed on 2026-09-06. All fixtures are explicitly synthetic, temporary
software-validation inputs. Tests cover stable match partitions, noncontiguous
chunks, pilot exclusions, upstream partition provenance, duplicate-match rejection,
identity-stable physics, discovery-only normalization, nonfinite/overflow labels,
undefined heading, all-player action mismatches, relative timing, differences hidden
by OR pooling, missing players, artifact round-trip/alignment, pre-download budget
and revision guards, and unsafe archive paths. No synthetic
fixture is counted as a Rocket Science observation or model result.
The execution record is [data_selftest.json](../results/data_selftest.json).

## Decoded observational cohort

`scripts/prepare_clips.py` writes per-clip NPZ artifacts outside Git, under
`/data2/ishaangp/mira-interp/prepared`. Each contains `frames [4,16,3,288,512]`
uint8, `actions [4,16,9]`, `targets [4,16,30]`, `timestamps [4,16]`,
`source_frame_indices [16]`, `video_pts [4,16]`, `player_ids`, and `target_names`.
The video and actions remain at the pretrained model's 20 FPS. PyAV decodes video;
resizing exactly follows the upstream antialiased bilinear, `align_corners=False`,
round-and-clamp uint8 path. Each view keeps its own physical labels. The four
perspectives are approximately aligned and must not be treated as observations
at an exact shared simulator instant.

Preparation checks every physics chunk's row count before constructing its score
timeline; rejects duplicate tar member basenames; verifies the source shard's
SHA-256 against the publisher-verified download report; and validates chunk/player/
team metadata. Score total `n` maps to the nth goal anchor, including recordings
that begin with nonzero score. Missing, ambiguous, sparse, or inconsistent anchor
pairings stop that match. The index's original recording offset is retained in
the report, but is not substituted for a verified trim origin. Calibrated timestamps
are estimates with measured residuals, not claims of frame-exact master alignment.

Before selecting windows, deterministic structural masks exclude nonfinite or
missing state, wrong identities, demolitions, and fully frozen physics. "Frozen"
means all five tracked entities move no more than 2 uu over an adjacent source
step. The calibrated goal/kickoff/replay event mask also leaves a 0.5-second margin
around event boundaries. Eight nonoverlapping 16-frame windows are selected at
the centers of equal quantiles of eligible windows. No speed bins, target values,
model outputs, or probe performance influence selection. The mask hash and counts
are saved. Candidate clips additionally must pass the unchanged cross-view ball
check: mean residual at most 100 uu after allowing at most four frames of lag.
A failed selected clip stops the gate; it is not silently replaced.

The report records each source component hash, exact selected frame indices,
video frame counts and presentation timestamps, source-clock fits and hashes,
identity checks, cross-view lag diagnostics, prepared-array hashes, and the
preparation code hashes. The final audit binds the exact clip manifest hash.

```bash
python scripts/prepare_clips.py --manifest data/pilot_clip_manifest.json
python scripts/prepare_clips.py --roles discovery selection confirmation --workers 4 --manifest data/clip_manifest.json --report results/cohort_data_audit.json
PYTHONPATH=src python -m pytest -q tests/test_clips.py
```

Six additional synthetic tests passed, covering trim-origin recovery, nonzero
initial score, ambiguous/sparse/inconsistent calibration failures, deterministic
eligible-window planning, per-view targets, structural filtering, and real PyAV
video decoding/count/timestamp checks. Parallel preparation uses at most four
workers and eight configured CPU threads, with atomic progress after each match.
It starts no GPU computation and trains no model. A complete research gate
requires all 61 assigned matches to provide exactly eight passing clips; no
automatic cohort replacement is allowed.

### Original failure and prospective v2 qualification

The complete original audit inspected all 61 assigned matches. Its preserved
`progress.completed_matches = 60` field is the last periodic snapshot, not the
final count. The final 61 `completed_match_ids`, completion timestamp, and 53
passing plus eight failed matches are authoritative; the frozen report has not
been rewritten to change that historical progress snapshot. Seven failed the
unchanged 0.1-second clock-residual rule, and one had fewer than three usable goal
anchors. Its final status remains **failed**, with 53 passing matches and 424 of
488 expected clips. The exclusions were six discovery, one selection, and one
confirmation match. [source_clock_failures.json](../results/source_clock_failures.json)
preserves all discrete-event diagnostics. Examples include one client's scoreboard
update arriving approximately 13.5 seconds late and another recording's inferred
clock origin shifting approximately 0.45 seconds mid-match. These are substantive
source problems, so the threshold was not increased.

Before any research activation capture or probe fitting, protocol v2 registered
exactly all 53 fully passing matches, retaining their original roles and selected
clips. No replacement, rebalancing, state-value threshold change, or performance
selection occurred. The immutable registration is
[observational_registration_v2.json](../results/observational_registration_v2.json);
its protocol hash is
`e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2`.

`scripts/audit_qualified_cohort.py` independently verified every retained NPZ hash,
array shape/dtype, action values, finite per-view labels, target/player ordering,
nonoverlapping frame indices, per-view timestamps, clock-pair residuals, publisher
source hashes, and preparation-code lineage. It also checked that the subset is
exactly the original passing inventory, that all original failures are preserved,
and that the separately reserved pilot's eight artifacts are unchanged. No raw
array was rewritten. The resulting research manifest is
`data/qualified_clip_manifest.json`, hash
`cf0fb3679151b21d508996125b5694cd7021f748cade5d09f538d3ef488d29b5`.
The v2 pilot manifest is `data/qualified_pilot_clip_manifest.json`.

Results from v2 concern this quality-qualified population. Requiring consistent
clients and at least three goals, then excluding frozen or demolition-affected
windows, can bias coverage. The 31/11/11 match counts meet the registered
operational minima; they are not an a priori power calculation. Match separation
also does not guarantee separation of players who appear in multiple matches.

An independent visual spotcheck of the first predetermined pilot frame found
HUD boost values 81/96/0/4 across the four views, compatible with the same rows'
physics values 81.54/96.67/0/4.09 percent. Team colors and scores also agree.
[pilot_pixel_state_spotcheck.json](../results/pilot_pixel_state_spotcheck.json)
records the evidence and limitation: this small check supports own-view row
correspondence, but does not prove zero visual-state latency or complete 3D camera
calibration. Physics JSONL records lack explicit per-row timestamps, so full
correspondence still partly relies on the publisher's export contract.

## Sources checked

- [Rocket Science dataset](https://huggingface.co/datasets/kyutai/rocket-science)
- Upstream MIRA `src/mira/data/schema.py`: match/chunk identity and player ordering.
- Upstream MIRA `src/mira/data/state.py`: physics fields, coordinates, and local-only clock.
- Upstream MIRA `src/mira/data/actions.py`: nine-key vocabulary and OR downsampling.
- Upstream MIRA `src/mira/data/physics.py`: synchronization, frozen-state and event checks.

The exact upstream code revision is recorded by the repository-level provenance.
