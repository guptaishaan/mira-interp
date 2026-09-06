"""Small, strict preparation helpers for Rocket Science physical-state studies.

No generated examples are treated as data. Source-rate action tracks are retained
because OR-downsampled actions cannot establish equal future controls.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SPLITS = ("discovery", "selection", "confirmation")
ACTION_KEYS = ("W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey")
TARGET_NAMES = tuple(
    f"{entity}.{field}.{axis}"
    for entity in ("ball", "player0", "player1", "player2", "player3")
    for field in ("location", "velocity")
    for axis in "xyz"
)


def match_split(match_id: str, seed: str = "mira-interp-v1") -> str:
    """Stable 60/20/20 match assignment, independent of input order or dataset size."""
    if not isinstance(match_id, str) or not match_id:
        raise ValueError("match_id must be a nonempty string")
    value = int.from_bytes(hashlib.sha256(f"{seed}\0{match_id}".encode()).digest()[:8], "big")
    # Integer thresholds keep the assignment independent of floating-point rounding.
    return SPLITS[0] if value * 10 < 6 * 2**64 else SPLITS[1] if value * 10 < 8 * 2**64 else SPLITS[2]


def make_split_manifest(indices: Sequence[dict], *, revision: str, seed: str = "mira-interp-v1",
                        pilot_match_ids: Sequence[str] = (), index_sources: Sequence[dict] | None = None) -> dict:
    """Freeze whole-match assignments; verify chunk identity and index consistency."""
    if not revision:
        raise ValueError("a pinned source revision is required")
    pilots = set(pilot_match_ids)
    if index_sources is None:
        index_sources = [{"upstream_split": "unknown", "basis": "not provided"} for _ in indices]
    if len(index_sources) != len(indices):
        raise ValueError("one source provenance entry per input index is required")
    matches: dict[str, dict] = {}
    for index, source in zip(indices, index_sources):
        entries = index["entries"]
        if index["total_samples"] != len(entries):
            raise ValueError("index total_samples disagrees with the number of match entries")
        for entry in entries:
            match_id = entry["match_id"]
            if match_id in matches:
                raise ValueError(f"duplicate match_id across input indices: {match_id}")
            players = [p["player_id"] for p in entry["perspectives"]]
            if entry["n_players"] != 4 or len(players) != 4 or len(set(players)) != 4 or any(type(pid) is not int for pid in players):
                raise ValueError("each match must have exactly four distinct players")
            if players != sorted(players):
                raise ValueError("perspectives must follow upstream player_id ordering")
            frames = entry["chunk_frames"]
            chunks = entry.get("chunk_indices")
            if chunks is None:
                chunks = list(range(len(frames)))
            if not frames or any(type(n) is not int or n <= 0 for n in frames):
                raise ValueError("chunk_frames must be nonempty positive integers")
            if len(chunks) != len(frames) or len(set(chunks)) != len(chunks):
                raise ValueError("chunk_indices must uniquely identify every present chunk")
            if any(type(n) is not int or n < 0 for n in chunks):
                raise ValueError("chunk indices must be nonnegative integers")
            for perspective in entry["perspectives"]:
                if perspective["frames"] != sum(frames):
                    raise ValueError("perspective frame total disagrees with chunk_frames")
                if not np.isfinite(perspective["duration"]) or perspective["duration"] <= 0:
                    raise ValueError("perspective duration must be finite and positive")
                if not np.isfinite(perspective.get("recording_offset_sec", 0)):
                    raise ValueError("recording offset must be finite")
            teams = sorted(p["team"] for p in entry["perspectives"])
            if teams != [0, 0, 1, 1]:
                raise ValueError("expected two players on each of the two teams")
            matches[match_id] = {
                "match_id": match_id, "split": "pilot" if match_id in pilots else match_split(match_id, seed),
                "upstream_split": source.get("upstream_split", "unknown"), "source_index": dict(source),
                "shard": entry["shard"], "player_ids": players,
                "chunks": [{"chunk_index": c, "source_frames": n} for c, n in zip(chunks, frames)],
            }
    if not matches:
        raise ValueError("cannot prepare an empty dataset")
    if pilots - set(matches):
        raise ValueError("pilot match IDs must exist in the provided indices")
    rows = [matches[key] for key in sorted(matches)]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    return {"schema_version": 1, "source_revision": revision, "seed": seed,
            "assignment": "sha256 match_id; fixed 60/20/20 thresholds", "sha256": digest,
            "match_counts": counts, "pilot_match_ids": sorted(pilots), "pilot_count": len(pilots),
            "engineering_pilot_reserved": bool(pilots),
            "upstream_split_provenance_known": all(s.get("upstream_split", "unknown") in ("train", "dev", "test") for s in index_sources),
            "all_splits_nonempty": all(counts.values()), "matches": rows}


def tensorize_source_actions(tracks: Sequence[Sequence[str | dict]]) -> np.ndarray:
    """Four aligned source-rate keyboard tracks -> uint8 (4, T, 9), without pooling."""
    if len(tracks) != 4 or not tracks[0] or len({len(t) for t in tracks}) != 1:
        raise ValueError("actions require four nonempty equal-length player tracks")
    output = np.zeros((4, len(tracks[0]), len(ACTION_KEYS)), dtype=np.uint8)
    vocabulary = {key: i for i, key in enumerate(ACTION_KEYS)}
    for p, track in enumerate(tracks):
        for t, raw in enumerate(track):
            row = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
            keys = (row.get("keys") or []) if isinstance(row, dict) else []
            if row not in ("", None) and not isinstance(row, dict):
                raise ValueError("action row must be a JSON object")
            for key in keys:
                if key not in vocabulary:
                    raise ValueError(f"unknown action key: {key}")
                output[p, t, vocabulary[key]] = 1
    return output


def _validate_actions(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[0] != 4 or value.shape[1] == 0 or value.shape[2] != 9:
        raise ValueError("future actions must have shape (4, nonzero source frames, 9)")
    if not np.isin(value, [0, 1]).all():
        raise ValueError("actions must be binary and finite")
    return value.astype(np.uint8)


def _validate_source_indices(indices: Sequence[int], expected_frames: int) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.ndim != 1 or len(indices) != expected_frames or not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any():
        raise ValueError("source indices must be frame-aligned nonnegative integers")
    if len(indices) > 1 and not np.all(np.diff(indices) == 1):
        raise ValueError("source frames must be contiguous and unpooled")
    return indices


def audit_future_actions(reference: np.ndarray, donor: np.ndarray, *,
                         reference_player_ids: Sequence[int], donor_player_ids: Sequence[int],
                         reference_source_frame_indices: Sequence[int], donor_source_frame_indices: Sequence[int],
                         reference_source_fps: float, donor_source_fps: float) -> dict:
    """Exact equality for all players at source rate; never certify counterfactuals.

    The caller slices both arrays to the intervention's future horizon. Identity
    mapping must be identical; cross-match role mapping needs a separate protocol.
    """
    reference, donor = _validate_actions(reference), _validate_actions(donor)
    if any(not np.isfinite(fps) or fps <= 0 for fps in (reference_source_fps, donor_source_fps)):
        raise ValueError("source_fps must be finite and positive")
    reference_indices = _validate_source_indices(reference_source_frame_indices, reference.shape[1])
    donor_indices = _validate_source_indices(donor_source_frame_indices, donor.shape[1])
    for ids in (reference_player_ids, donor_player_ids):
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("four distinct player IDs are required")
    identity_equal = list(reference_player_ids) == list(donor_player_ids)
    same_shape = reference.shape == donor.shape
    relative_time_equal = same_shape and np.allclose(
        (reference_indices - reference_indices[0]) / reference_source_fps,
        (donor_indices - donor_indices[0]) / donor_source_fps, rtol=0, atol=1e-9)
    rate_equal = bool(np.isclose(reference_source_fps, donor_source_fps, rtol=0, atol=1e-9))
    difference_count = int(np.count_nonzero(reference != donor)) if same_shape else None
    equal = identity_equal and same_shape and relative_time_equal and rate_equal and difference_count == 0
    return {"schema_version": 1, "all_four_players": True,
            "reference_source_fps": float(reference_source_fps), "donor_source_fps": float(donor_source_fps),
            "same_relative_time_grid": bool(relative_time_equal), "same_source_rate": rate_equal,
            "time_grid_tolerance_seconds": 1e-9,
            "reference_horizon_frames": reference.shape[1], "donor_horizon_frames": donor.shape[1],
            "same_player_identity_order": identity_equal, "same_shape": same_shape,
            "different_key_frame_entries": difference_count, "future_actions_exactly_equal": bool(equal),
            "controlled_counterfactual": False,
            "interpretation": "observational action equality only; other state and simulator provenance are not established"}


def physics_targets(frames: Sequence[dict], player_ids: Sequence[int]) -> np.ndarray:
    """Return (T, 30) ball/car position and velocity in original Unreal units.

    Explicit persistent player IDs, rather than each frame's list order, define
    player0..player3. Units are positions in uu and velocities in uu per second.
    """
    if len(player_ids) != 4 or len(set(player_ids)) != 4 or not frames:
        raise ValueError("nonempty physics and four distinct player IDs are required")
    rows = []
    for frame in frames:
        cars = {car["player_id"]: car for car in frame["cars"]}
        if len(frame["cars"]) != 4 or set(cars) != set(player_ids):
            raise ValueError("physics player identities changed, duplicated, or are missing")
        entities = [frame["ball"], *(cars[pid] for pid in player_ids)]
        row = [float(entity[field][axis]) for entity in entities for field in ("location", "velocity") for axis in "xyz"]
        if not np.isfinite(row).all():
            raise ValueError("physical labels must be finite")
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def velocity_heading(xy_velocity: np.ndarray, minimum_speed: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return unit [cos(theta), sin(theta)] and mask; standing still has no heading."""
    values = np.asarray(xy_velocity, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2 or not np.isfinite(values).all():
        raise ValueError("finite horizontal velocity pairs are required")
    if not np.isfinite(minimum_speed) or minimum_speed <= 0:
        raise ValueError("minimum_speed must be positive")
    speed = np.hypot(values[..., 0], values[..., 1])
    if not np.isfinite(speed).all():
        raise ValueError("velocity magnitude overflowed")
    valid = speed >= minimum_speed
    heading = np.full_like(values, np.nan)
    np.divide(values, speed[..., None], out=heading, where=valid[..., None])
    return heading, valid


@dataclass
class PhysicsNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, splits: Sequence[str]) -> "PhysicsNormalizer":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2 or not values.shape[1] or len(splits) != len(values):
            raise ValueError("at least two labeled rows and one split per row are required")
        if set(splits) != {"discovery"}:
            raise ValueError("normalizer must be fitted on discovery rows only")
        if not np.isfinite(values).all():
            raise ValueError("normalizer inputs must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            scale, mean = values.std(axis=0), values.mean(axis=0)
        if not np.isfinite(scale).all() or not np.isfinite(mean).all():
            raise ValueError("normalizer statistics overflowed")
        return cls(mean, np.where(scale > 0, scale, 1.0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("normalizer dimensions disagree or labels are nonfinite")
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale + self.mean

    def to_dict(self) -> dict:
        return {"fit_split": "discovery", "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "constant_target_scale": 1.0}


def write_prepared_clip(destination: str | Path, *, match_id: str, chunk_index: int,
                        player_ids: Sequence[int], physics: Sequence[dict],
                        source_actions: np.ndarray, source_frame_indices: Sequence[int],
                        source_fps: float, source_revision: str, synthetic: bool = False,
                        seed: str = "mira-interp-v1", pilot_match_ids: Sequence[str] = ()) -> dict:
    """Persist a verified, unpooled source-rate label artifact and adjacent manifest.

    This is a label/action preparation artifact, not a decoded-video dataset. Its
    manifest locates the source match/chunk; downstream code must align pixels.
    """
    actions = _validate_actions(source_actions)
    targets = physics_targets(physics, player_ids)
    indices = _validate_source_indices(source_frame_indices, len(targets))
    if len(indices) != actions.shape[1]:
        raise ValueError("physics, source actions, and source indices must be frame aligned")
    if not np.isfinite(source_fps) or source_fps <= 0 or type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("source FPS must be positive and chunk index nonnegative")
    if not source_revision:
        raise ValueError("a pinned source revision is required")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix != ".npz":
        raise ValueError("prepared array path must end in .npz")
    np.savez_compressed(destination, targets=targets, source_actions=actions, source_frame_indices=indices,
                        player_ids=np.asarray(player_ids), target_names=np.asarray(TARGET_NAMES))
    manifest = {"schema_version": 1, "artifact_type": "source_rate_labels_and_actions",
                "synthetic": synthetic, "research_evidence": False if synthetic else None,
                "match_id": match_id, "split": "pilot" if match_id in set(pilot_match_ids) else match_split(match_id, seed), "split_seed": seed,
                "chunk_index": chunk_index, "player_ids": list(player_ids), "source_revision": source_revision,
                "source_fps": float(source_fps), "source_frames": len(targets),
                "target_names": list(TARGET_NAMES), "action_keys": list(ACTION_KEYS),
                "array_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "raw_units": {"location": "Unreal units", "velocity": "Unreal units per second"},
                "video_alignment_checked": False, "live_gameplay_checked": False}
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
