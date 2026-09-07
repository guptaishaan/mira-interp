#!/usr/bin/env python3
"""Discovery-only observational pair availability, never simulator counterfactuals."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    manifest_path = ROOT / "data/qualified_clip_manifest.json"
    gate_path = ROOT / "results/qualified_data_audit.json"
    manifest = json.loads(manifest_path.read_text())
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "passed" or gate["clip_manifest_sha256"] != sha(manifest_path):
        raise ValueError("qualified data gate has not passed for this exact manifest")
    records = [record for record in manifest["records"] if record["role"] == "discovery"]
    groups = defaultdict(list)
    npz_hashes = {}
    for record in records:
        path = Path(record["artifact_path"])
        digest = sha(path)
        if digest != record["sha256"]:
            raise ValueError("discovery NPZ changed after qualification")
        npz_hashes[record["clip_id"]] = digest
        with np.load(path, allow_pickle=False) as arrays:
            ids = tuple(arrays["player_ids"].tolist())
            if list(ids) != record["player_ids"] or len(set(ids)) != 4:
                raise ValueError("player ordering is inconsistent")
            actions, targets = arrays["actions"], arrays["targets"]
            if actions.shape != (4, 16, 9) or not np.isin(actions, [0, 1]).all():
                raise ValueError("source actions must be binary and include all four players")
            if targets.shape != (4, 16, 30) or not np.isfinite(targets).all():
                raise ValueError("invalid per-view state labels")
            if not np.all(np.diff(arrays["source_frame_indices"]) == 1):
                raise ValueError("source frames are not contiguous")
            times = arrays["timestamps"]
            if times.shape != (4, 16) or not np.allclose(np.diff(times, axis=1), .05, rtol=0, atol=1e-9):
                raise ValueError("source action timing differs from the registered20FPS grid")
            future = np.ascontiguousarray(actions[:, 1:16, :], dtype=np.uint8)
            groups[record["match_id"]].append({"ids": ids, "future": future.copy(),
                                               "action_hash": hashlib.sha256(future.tobytes()).hexdigest(),
                                               "initial_state": targets[0, 0].copy()})
    if len(groups) != 31 or len(records) != 248 or any(len(clips) != 8 for clips in groups.values()):
        raise ValueError("expected all31 discovery matches with8clips each")
    total, action_equal, exactly_one, initial_identical = 0, 0, 0, 0
    matches_with_equal_actions = 0
    for clips in groups.values():
        if len({clip["ids"] for clip in clips}) != 1:
            raise ValueError("canonical player identities change within a match")
        this_match_equal = 0
        for reference, donor in itertools.combinations(clips, 2):
            total += 1
            if reference["action_hash"] != donor["action_hash"]:
                continue
            if not np.array_equal(reference["future"], donor["future"]):
                raise ValueError("action hash collision")
            action_equal += 1
            this_match_equal += 1
            differences = int(np.count_nonzero(reference["initial_state"] != donor["initial_state"]))
            exactly_one += differences == 1
            initial_identical += differences == 0
        matches_with_equal_actions += this_match_equal > 0
    if total != 31 * 28:
        raise ValueError("pair enumeration is incomplete")
    report = {"schema_version": 1, "status": "completed_availability_audit",
              "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha(Path(__file__)), "qualified_clip_manifest_sha256": sha(manifest_path),
              "qualified_data_audit_sha256": sha(gate_path), "protocol_sha256": gate["protocol_sha256"],
              "cohort_sha256": gate["cohort_sha256"], "roles_read": ["discovery"],
              "selection_or_confirmation_arrays_read": False, "discovery_matches": len(groups),
              "discovery_clips": len(records), "clips_per_match": 8,
              "discovery_npz_inventory_sha256": hashlib.sha256(json.dumps(npz_hashes, sort_keys=True).encode()).hexdigest(),
              "within_match_pairs_evaluated": total, "exact_all_player_future_action_matches": action_equal,
              "matches_with_any_exact_action_pair": matches_with_equal_actions,
              "exact_action_pairs_with_one_initial_coordinate_changed": exactly_one,
              "exact_action_pairs_with_identical_initial_30_coordinates": initial_identical,
              "action_comparison": "exact equality of all4players x sourceframes1..15 x9keys, at20FPS; no pooling",
              "future_horizon_seconds": .75,
              "state_comparison": "view0 sourceframe0,30 ball/player XYZ position/velocity coordinates; exact float equality",
              "canonical_player_identity_rule": "same match and identical ordered player IDs only",
              "controlled_counterfactual_pairs_established": 0,
              "causal_identification_gate_passed": False,
              "limitations": [
                  "Availability is exhaustive only among the frozen8clips per discovery match, not the complete dataset.",
                  "Thirty coordinates omit rotation, angular velocity, boost, contact state, camera history, and hidden simulator state.",
                  "Observed equal actions and a one-coordinate label difference would not prove a controlled state intervention.",
                  "No simulator reset interface or reproducible counterfactual renderer has been established.",
                  "No paired-seed causal model intervention or independently validated generated-video physical evaluator was run.",
                  "No selected pairs or physical label values are exported by this audit."]}
    output = ROOT / "results/controlled_pair_feasibility.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("within_match_pairs_evaluated", "exact_all_player_future_action_matches",
                    "exact_action_pairs_with_one_initial_coordinate_changed", "causal_identification_gate_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
