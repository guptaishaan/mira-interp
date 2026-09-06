# Data preparation and access

**Authentication now works, and the initial shard is downloaded and verified.**
The first decoded pilot audit failed: the released video timeline is trimmed,
but the index's recording offsets do not include that trim. In the reserved pilot,
nine scoreboard changes align with goal-event anchors after an approximately
8.133-second correction. The uncorrected event mask admitted frozen replay clips.
We are repairing and retesting this source-clock mapping before any research
capture. This is not yet a completed research dataset or a scientific result.
The access observation is in [data_access.json](../results/data_access.json);
the live preparation gate is [pilot_data_audit.json](../results/pilot_data_audit.json).

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

Once this node has been authenticated with the user's authorized account:

```bash
python scripts/prepare_data.py
```

The command writes `results/data_access.json`. On success it also writes
`data/split_manifest.json`, with the input index hash and exact dataset revision.
Its success means **index preparation only**: video shards, physical labels,
live-gameplay checks, and matched trajectories remain unprepared. A blocked
access check or empty partition exits with code 2; malformed input raises an
error. To prepare a previously downloaded index without a network call:

```bash
python scripts/prepare_data.py --index /path/to/index.json --revision DATASET_COMMIT_SHA
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
python scripts/prepare_data.py --download-one-shard --data-dir /path/to/raw-data
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

## Sources checked

- [Rocket Science dataset](https://huggingface.co/datasets/kyutai/rocket-science)
- Upstream MIRA `src/mira/data/schema.py`: match/chunk identity and player ordering.
- Upstream MIRA `src/mira/data/state.py`: physics fields, coordinates, and local-only clock.
- Upstream MIRA `src/mira/data/actions.py`: nine-key vocabulary and OR downsampling.
- Upstream MIRA `src/mira/data/physics.py`: synchronization, frozen-state and event checks.

The exact upstream code revision is recorded by the repository-level provenance.
