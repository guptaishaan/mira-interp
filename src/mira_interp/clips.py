"""Deterministic observational clip preparation; preserve each view's own clock."""
from __future__ import annotations

import io
import hashlib
import json
from typing import Sequence

import numpy as np

from .data import physics_targets, tensorize_source_actions


def live_at(anchors: Sequence[dict], timestamp: float, margin_seconds: float = 0.5) -> bool:
    """Conservative metadata-only live mask: after kickoff ends, before next event."""
    ordered = sorted(anchors, key=lambda event: event["master_sec"])
    previous = None
    for event in ordered:
        moment = float(event["master_sec"])
        if abs(moment - timestamp) < margin_seconds:
            return False
        if moment > timestamp:
            break
        previous = event["event_name"]
    return previous == "KickoffEnded"


def calibrate_timeline(entry: dict, game_tracks: Sequence[Sequence[dict]]) -> dict:
    """Recover trimmed-video time origins from unambiguous discrete score events.

    Targets such as position, velocity, speed, and model activations are never
    used here. Score total n identifies the nth GoalScored anchor even when the
    available recording starts with a nonzero score.
    """
    if len(game_tracks) != 4:
        raise ValueError("timeline calibration requires all four score tracks")
    perspectives = sorted(entry["perspectives"], key=lambda p: p["player_id"])
    results = []
    for view, track in enumerate(game_tracks):
        if len(track) != sum(entry["chunk_frames"]) or not track:
            raise ValueError("score timeline length disagrees with indexed frames")
        anchors = sorted((a for a in perspectives[view]["anchors"] if a["event_name"] == "GoalScored"),
                         key=lambda a: a["master_sec"])
        previous = int(track[0]["score_blue"]) + int(track[0]["score_orange"])
        pairs = []
        for frame, state in enumerate(track[1:], 1):
            current = int(state["score_blue"]) + int(state["score_orange"])
            if current == previous:
                continue
            if current != previous + 1 or current > len(anchors):
                raise ValueError("score changes cannot be paired unambiguously with goal anchors")
            source_seconds = frame / 20
            anchor_seconds = float(anchors[current - 1]["master_sec"])
            pairs.append({"score_total": current, "source_frame": frame, "source_seconds": source_seconds,
                          "anchor_seconds": anchor_seconds, "offset_seconds": anchor_seconds - source_seconds})
            previous = current
        if len(pairs) < 3:
            raise ValueError("fewer than three unambiguous goal anchors; trimmed source origin is unknown")
        origin = float(np.median([pair["offset_seconds"] for pair in pairs]))
        for pair in pairs:
            pair["residual_seconds"] = pair["offset_seconds"] - origin
        worst = max(abs(pair["residual_seconds"]) for pair in pairs)
        if worst > 0.1:
            raise ValueError("score-anchor timestamp residual exceeds frozen 0.1 second tolerance")
        results.append({"view": view, "player_id": perspectives[view]["player_id"],
                        "metadata_recording_offset_seconds": perspectives[view].get("recording_offset_sec", 0),
                        "calibrated_origin_seconds": origin, "max_absolute_residual_seconds": worst,
                        "goal_pairs": pairs,
                        "anchor_sha256": hashlib.sha256(json.dumps(anchors, sort_keys=True).encode()).hexdigest(),
                        "score_track_sha256": hashlib.sha256(json.dumps(track, sort_keys=True).encode()).hexdigest()})
    return {"method": "median GoalScored master time minus corresponding source score-increment time",
            "minimum_goal_pairs": 3, "maximum_residual_seconds": 0.1,
            "uses_position_or_velocity_values": False, "views": results}


def structural_validity_mask(physics: Sequence[dict], player_ids: Sequence[int], teams: Sequence[int],
                             view: int) -> tuple[np.ndarray, dict]:
    """Frame validity from missing/invalid entities, demolitions and frozen state only."""
    values = np.full((len(physics), 30), np.nan)
    valid = np.ones(len(physics), dtype=bool)
    counts = {"invalid_state_frames": 0, "demolition_frames": 0, "frozen_frames": 0}
    for index, frame in enumerate(physics):
        try:
            values[index] = physics_targets([frame], player_ids)[0]
            cars = {car["player_id"]: car for car in frame["cars"]}
            local = [car["player_id"] for car in frame["cars"] if car.get("is_local")]
            if local != [player_ids[view]] or [cars[pid]["team"] for pid in player_ids] != list(teams):
                raise ValueError("identity mismatch")
            if any(cars[pid]["attacker_player_id"] != -1 for pid in player_ids):
                valid[index] = False
                counts["demolition_frames"] += 1
        except (KeyError, TypeError, ValueError):
            valid[index] = False
            counts["invalid_state_frames"] += 1
    if len(physics) > 1:
        positions = values.reshape(len(physics), 5, 6)[..., :3]
        still = np.all(np.linalg.norm(np.diff(positions, axis=0), axis=-1) <= 2., axis=-1)
        frozen = np.concatenate([still, still[-1:]])
        valid &= ~frozen
        counts["frozen_frames"] = int(frozen.sum())
    return valid, counts


def plan_clips(entry: dict, count: int = 8, clip_len: int = 16,
               calibrated_origins: Sequence[float] | None = None,
               source_valid_mask: np.ndarray | None = None) -> list[dict]:
    """Evenly spaced, nonoverlapping source windows selected using index metadata only."""
    chunks = entry.get("chunk_indices") or list(range(len(entry["chunk_frames"])))
    if chunks != list(range(len(chunks))):
        raise ValueError("gapped source chunks need explicit original timestamps; cannot sum surviving chunks")
    if count <= 0 or clip_len <= 0:
        raise ValueError("count and clip_len must be positive")
    perspectives = sorted(entry["perspectives"], key=lambda p: p["player_id"])
    if calibrated_origins is None or len(calibrated_origins) != 4 or not np.isfinite(calibrated_origins).all():
        raise ValueError("four verified calibrated timeline origins are required")
    if source_valid_mask is None:
        source_valid_mask = np.ones(sum(entry["chunk_frames"]), dtype=bool)
    if np.asarray(source_valid_mask).shape != (sum(entry["chunk_frames"]),):
        raise ValueError("structural validity mask must cover every source frame")
    fps_values = [float(p.get("fps", p["frames"] / p["duration"])) for p in perspectives]
    if len(perspectives) != 4 or not np.allclose(fps_values, 20.0, rtol=0, atol=1e-6):
        raise ValueError("released model preparation requires four 20 FPS views")
    candidates = []
    start = 0
    for chunk_index, frames in zip(chunks, entry["chunk_frames"]):
        for local_start in range(0, frames - clip_len + 1, clip_len):
            indices = np.arange(start + local_start, start + local_start + clip_len)
            if not np.all(source_valid_mask[indices]):
                continue
            if all(all(live_at(p["anchors"], float(frame / 20 + calibrated_origins[view]))
                       for frame in indices) for view, p in enumerate(perspectives)):
                candidates.append({"chunk_index": chunk_index, "local_start": local_start,
                                   "source_start_frame": int(indices[0]), "frame_count": clip_len})
        start += frames
    if len(candidates) < count:
        raise ValueError("insufficient metadata-eligible live windows")
    # Center of each equal-size quantile, avoiding duplicate endpoints and late selection.
    positions = np.floor((np.arange(count) + 0.5) * len(candidates) / count).astype(int)
    return [dict(candidates[position], candidate_ordinal=int(position), eligible_candidates=len(candidates))
            for position in positions]


def decode_video(video: bytes, selected: Sequence[int], expected_frames: int,
                 size: tuple[int, int] = (288, 512)) -> tuple[np.ndarray, np.ndarray, dict]:
    """PyAV decode + upstream-equivalent antialiased bilinear CPU resize."""
    import av
    import torch
    import torch.nn.functional as F

    if list(selected) != sorted(set(selected)) or not selected:
        raise ValueError("selected frame indices must be nonempty, unique, and increasing")
    chosen = set(selected)
    arrays, times = [], []
    count = 0
    all_pts = []
    with av.open(io.BytesIO(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        native_size = [stream.height, stream.width]
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("video frame is missing presentation timestamp")
            timestamp = float(frame.pts * frame.time_base)
            all_pts.append(timestamp)
            if index in chosen:
                arrays.append(frame.to_ndarray(format="rgb24"))
                times.append(timestamp)
            count += 1
    if count != expected_frames or len(arrays) != len(selected):
        raise ValueError("decoded video count disagrees with index/selected frames")
    if not np.isclose(fps, 20.0) or not np.allclose(np.diff(all_pts), 1 / 20, rtol=0, atol=1e-7):
        raise ValueError("video presentation timestamps are not contiguous 20 FPS")
    pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    resized = F.interpolate(pixels.float(), size=size, mode="bilinear", align_corners=False,
                            antialias=True).round().clamp_(0, 255).to(torch.uint8).numpy()
    return resized, np.asarray(times), {"decoded_frame_count": count, "native_size": native_size,
                                       "fps": fps, "pts_monotonic_and_uniform": True}


def audit_physics_tracks(physics: Sequence[Sequence[dict]], action_tracks: Sequence[Sequence[str]],
                         player_ids: Sequence[int], teams: Sequence[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Audit identity/units/time evolution without forcing views to share one label."""
    if len(physics) != 4 or len({len(track) for track in physics}) != 1:
        raise ValueError("four equal-length physics tracks are required")
    targets = np.stack([physics_targets(track, player_ids) for track in physics])
    actions = tensorize_source_actions(action_tracks)
    if actions.shape[1] != targets.shape[1]:
        raise ValueError("physics and actions are not frame aligned")
    demolished = 0
    for view, track in enumerate(physics):
        for frame in track:
            local = [car for car in frame["cars"] if car.get("is_local")]
            if len(local) != 1 or local[0]["player_id"] != player_ids[view]:
                raise ValueError("local camera player identity disagrees with index")
            actual_teams = {car["player_id"]: car["team"] for car in frame["cars"]}
            if [actual_teams[pid] for pid in player_ids] != list(teams):
                raise ValueError("physics and index team mapping disagree")
            demolished += sum(car["attacker_player_id"] != -1 for car in frame["cars"])
    positions = targets.reshape(4, targets.shape[1], 5, 6)[..., :3]
    moved = np.linalg.norm(np.diff(positions, axis=1), axis=-1)
    frozen_steps = np.all(moved <= 2.0, axis=-1)
    if np.any(frozen_steps):
        raise ValueError("fully frozen physical-state transition in proposed live clip")
    if demolished:
        raise ValueError("demolition-affected entity in proposed live clip")
    ball = targets[:, :, :3]
    offsets = []
    for view in range(1, 4):
        residuals = []
        for lag in range(-4, 5):
            a = ball[0, max(0, lag):min(ball.shape[1], ball.shape[1] + lag)]
            b = ball[view, max(0, -lag):min(ball.shape[1], ball.shape[1] - lag)]
            residuals.append((float(np.linalg.norm(a - b, axis=-1).mean()), lag))
        residual, lag = min(residuals)
        offsets.append({"view": view, "best_lag_frames": lag, "mean_ball_residual_uu": residual})
    # A large mismatch is an integrity failure. Passing is approximate shared-world
    # evidence only, not exact synchronization or controlled physics.
    if any(item["mean_ball_residual_uu"] > 100 for item in offsets):
        raise ValueError("cross-view ball trajectories disagree beyond 100 uu after up to four frames of lag")
    velocity_error = []
    for view in range(4):
        error = np.linalg.norm(np.diff(ball[view], axis=0) * 20 -
                               (targets[view, 1:, 3:6] + targets[view, :-1, 3:6]) / 2, axis=-1)
        velocity_error.append(float(np.median(error)))
    return targets, actions, {"all_player_identities_checked": True, "all_team_ids_checked": True,
                              "frozen_transitions": 0, "demolished_entity_frames": demolished,
                              "cross_view_ball_alignment": offsets,
                              "median_ball_finite_difference_velocity_error_uu_s": velocity_error,
                              "exact_cross_view_synchronization": False}
