#!/usr/bin/env python3
"""Independently validate v2's exact quality-qualified subset without rewriting data.

The original 61-match audit remains failed. This audit can pass only the separately
registered 53-match observational population, preserving every original role/clip.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL_SHA = "e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write(name: str, payload: dict) -> None:
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_clock(calibration: dict, player_ids: list[int]) -> list[float]:
    require(calibration["minimum_goal_pairs"] == 3 and calibration["maximum_residual_seconds"] == .1,
            "clock thresholds changed")
    require(calibration["uses_position_or_velocity_values"] is False, "clock fit used target variables")
    require([row["view"] for row in calibration["views"]] == list(range(4)), "missing clock view")
    origins = []
    for view, item in enumerate(calibration["views"]):
        require(item["player_id"] == player_ids[view], "clock player identity changed")
        pairs = item["goal_pairs"]
        require(len(pairs) >= 3, "fewer than three clock pairs")
        offsets = np.asarray([pair["anchor_seconds"] - pair["source_frame"] / 20 for pair in pairs])
        require(np.isfinite(offsets).all(), "nonfinite clock offsets")
        origin = float(np.median(offsets))
        residual = np.max(np.abs(offsets - origin))
        require(residual <= .1, "clock residual exceeds original gate")
        require(abs(origin - item["calibrated_origin_seconds"]) < 1e-9, "reported clock origin differs from goal pairs")
        require(abs(residual - item["max_absolute_residual_seconds"]) < 1e-9, "reported clock residual differs from pairs")
        require(len({pair["score_total"] for pair in pairs}) == len(pairs), "duplicate goal-pair identity")
        require(all(len(item[field]) == 64 for field in ("anchor_sha256", "score_track_sha256")), "missing clock source hashes")
        origins.append(origin)
    return origins


def validate_record(record: dict, assigned: dict, calibration: dict, targets: list[str], publisher: dict) -> dict:
    match_id = record["match_id"]
    row = assigned[match_id]
    require(record["role"] == row["split"], "clip role differs from its original match")
    require(record["player_ids"] == row["player_ids"], "clip player identities changed")
    require(record["source_shard"] == row["shard"], "clip source shard changed")
    require(record["source_shard_sha256"] == publisher[record["source_shard"]], "clip source publisher hash differs")
    require(record["timeline_calibration"] == calibration, "clip clock calibration differs from match audit")
    origins = validate_clock(calibration, row["player_ids"])
    physical = record["physical_audit"]
    require(physical["all_player_identities_checked"] and physical["all_team_ids_checked"], "identity gate not passed")
    require(physical["frozen_transitions"] == physical["demolished_entity_frames"] == 0, "structural gate not passed")
    require([item["view"] for item in physical["cross_view_ball_alignment"]] == [1, 2, 3], "missing cross-view diagnostic")
    require(all(abs(item["best_lag_frames"]) <= 4 and np.isfinite(item["mean_ball_residual_uu"])
                and item["mean_ball_residual_uu"] <= 100 for item in physical["cross_view_ball_alignment"]),
            "cross-view physical gate not passed")
    require(len(record["video_audit"]) == 4 and all(item["fps"] == 20 and item["pts_monotonic_and_uniform"]
                for item in record["video_audit"]), "four-view video timestamp gate not passed")
    path = Path(record["artifact_path"])
    require(sha(path) == record["sha256"], f"prepared NPZ hash mismatch: {record['clip_id']}")
    plan = record["plan"]
    chunk = row["chunks"][plan["chunk_index"]]
    require(chunk["chunk_index"] == plan["chunk_index"], "source chunk ordering not contiguous")
    start = sum(item["source_frames"] for item in row["chunks"][:plan["chunk_index"]]) + plan["local_start"]
    require(start == plan["source_start_frame"] and plan["frame_count"] == 16, "source-frame plan changed")
    require(plan["local_start"] + 16 <= chunk["source_frames"], "clip crosses source chunk boundary")
    require(all(item["decoded_frame_count"] == chunk["source_frames"] for item in record["video_audit"]),
            "decoded source count differs from chunk metadata")
    with np.load(path, allow_pickle=False) as arrays:
        frames, actions, y = arrays["frames"], arrays["actions"], arrays["targets"]
        require(frames.shape == (4, 16, 3, 288, 512) and frames.dtype == np.uint8, "invalid video shape or dtype")
        require(actions.shape == (4, 16, 9) and np.isin(actions, [0, 1]).all(), "invalid source actions")
        require(y.shape == (4, 16, 30) and np.isfinite(y).all(), "invalid per-view physical labels")
        require(arrays["player_ids"].tolist() == row["player_ids"], "NPZ player identities differ")
        require(arrays["target_names"].tolist() == targets, "NPZ target ordering differs from registration")
        expected_indices = np.arange(start, start + 16)
        require(np.array_equal(arrays["source_frame_indices"], expected_indices), "NPZ source frames differ from frozen plan")
        times = arrays["timestamps"]
        require(times.shape == (4, 16) and np.isfinite(times).all(), "invalid per-view timestamps")
        require(np.allclose(times, expected_indices[None] / 20 + np.asarray(origins)[:, None], rtol=0, atol=1e-9),
                "NPZ per-view clock alignment differs from audited calibration")
        require(arrays["video_pts"].shape == (4, 16) and np.allclose(arrays["video_pts"],
                np.arange(plan["local_start"], plan["local_start"] + 16)[None] / 20, rtol=0, atol=1e-7),
                "NPZ video PTS differ from the selected source frames")
    return {"clip_id": record["clip_id"], "match_id": match_id, "role": record["role"], "status": "passed",
            "npz_sha256": record["sha256"], "all_view_arrays_and_timestamps_verified": True}


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    registration = read("results/observational_registration_v2.json")
    require(sha(ROOT / registration["protocol_path"]) == registration["sha256"] == EXPECTED_PROTOCOL_SHA,
            "v2 protocol differs from independently frozen registration")
    require(sha(ROOT / registration["qualified_split_manifest_path"]) == registration["qualified_split_manifest_sha256"],
            "qualified match manifest changed")
    for name, digest in registration["parent_artifact_sha256"].items():
        require(sha(ROOT / name) == digest, f"original artifact changed: {name}")
    protocol = read(registration["protocol_path"])
    original = read("data/split_manifest.json")
    qualified = read(registration["qualified_split_manifest_path"])
    original_audit = read("results/cohort_data_audit.json")
    original_clips = read("data/clip_manifest.json")
    pilot_audit = read("results/pilot_data_audit.json")
    pilot_clips = read("data/pilot_clip_manifest.json")
    clocks = read("results/source_clock_failures.json")
    original_audit_hash = sha(ROOT / "results/cohort_data_audit.json")
    require(original_audit["status"] == original_clips["status"] == "failed", "original failed status was altered")
    require(original_audit["clip_manifest_sha256"] == sha(ROOT / "data/clip_manifest.json"), "original clip manifest not bound")
    require(original_audit["split_manifest_sha256"] == sha(ROOT / "data/split_manifest.json"), "original split not bound")
    require(pilot_audit["status"] == pilot_clips["status"] == "passed", "engineering pilot has not passed")
    require(pilot_audit["clip_manifest_sha256"] == sha(ROOT / "data/pilot_clip_manifest.json"), "pilot manifest hash differs")
    require(clocks["audit_sha256"] == original_audit_hash and clocks["cohort_gate_relaxed"] is False,
            "clock diagnostic does not bind the failed original audit")
    excluded = {error["match_id"] for error in original_audit["errors"]}
    require(len(excluded) == len(original_audit["errors"]) == 8, "original failure set changed")
    require(all(error["stage"] == "source_clock_calibration" for error in original_audit["errors"]),
            "unregistered exclusion cause")
    require({item["match_id"] for item in clocks["findings"]} == excluded, "clock failure diagnostics incomplete")
    assigned = {row["match_id"]: row for row in original["matches"]}
    retained = {row["match_id"]: row for row in qualified["matches"]}
    require(len(assigned) == len(original["matches"]) == 62, "original match IDs not unique")
    require(len(retained) == len(qualified["matches"]) == 54, "qualified IDs not unique")
    require(set(retained) == set(assigned) - excluded, "qualified cohort is not exactly all original passing matches plus pilot")
    require(all(row == assigned[match_id] for match_id, row in retained.items()), "qualified match role/player/source metadata changed")
    semantic_hash = hashlib.sha256(json.dumps(qualified["matches"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(semantic_hash == qualified["sha256"] == registration["qualified_cohort_sha256"] ==
            protocol["cohort"]["balanced_manifest_sha256"], "qualified semantic cohort hash differs")
    require(Counter(row["split"] for row in retained.values()) ==
            Counter(discovery=31, selection=11, confirmation=11, pilot=1), "qualified role counts differ")
    require(set(original_audit["completed_match_ids"]) == {mid for mid, row in assigned.items() if row["split"] != "pilot"}
            and len(original_audit["completed_match_ids"]) == 61, "original audit did not finish every assigned match")
    for audit in (original_audit, pilot_audit):
        for name, digest in audit["preparation_code_sha256"].items():
            require(sha(ROOT / name) == digest, f"source preparation code changed: {name}")
    download = read("results/dataset_download.json")
    require(download["status"] == "PASS" and download["revision"] == qualified["source_revision"], "source download gate failed")
    require(original_audit["download_report_sha256"] == sha(ROOT / "results/dataset_download.json"), "download lineage changed")
    publisher = {Path(item["path"]).name: item["publisher_sha256"] for item in download["files"]
                 if item.get("verified") and item["sha256"] == item.get("publisher_sha256")}
    outcomes = []
    for scope, base_manifest, base_audit, output_manifest, output_audit in (
        ("research", original_clips, original_audit, "data/qualified_clip_manifest.json", "results/qualified_data_audit.json"),
        ("pilot", pilot_clips, pilot_audit, "data/qualified_pilot_clip_manifest.json", "results/qualified_pilot_data_audit.json"),
    ):
        records = base_manifest["records"]
        expected_ids = {mid for mid, row in retained.items() if (row["split"] == "pilot") == (scope == "pilot")}
        counts = Counter(record["match_id"] for record in records)
        require(set(counts) == expected_ids and set(counts.values()) == {8}, "incomplete qualified match clip coverage")
        require(len({r["clip_id"] for r in records}) == len(records), "duplicate clip IDs")
        require({r["clip_id"] for r in base_audit["clips"] if r["status"] == "passed"} == {r["clip_id"] for r in records},
                "qualified clips differ from original passing clip inventory")
        for match_id in expected_ids:
            used = set()
            for record in (r for r in records if r["match_id"] == match_id):
                indices = set(range(record["plan"]["source_start_frame"], record["plan"]["source_start_frame"] + 16))
                require(not used & indices, "overlapping clips within a match")
                used.update(indices)
        report = {"schema_version": 1, "status": "running", "scope": f"qualified observational {scope} data integrity",
                  "started_at_utc": started, "protocol_sha256": registration["sha256"],
                  "registration_sha256": sha(ROOT / "results/observational_registration_v2.json"),
                  "split_manifest_sha256": registration["qualified_split_manifest_sha256"],
                  "cohort_sha256": semantic_hash, "original_full_audit_sha256": original_audit_hash,
                  "original_full_audit_status": "failed", "original_clip_manifest_sha256": sha(ROOT / "data/clip_manifest.json"),
                  "original_passed_pilot_audit_sha256": sha(ROOT / "results/pilot_data_audit.json"),
                  "source_clock_diagnostics_sha256": sha(ROOT / "results/source_clock_failures.json"),
                  "excluded_match_ids": sorted(excluded), "expected_matches": len(expected_ids), "expected_clips": len(records),
                  "clips_per_match": 8, "roles": ["pilot"] if scope == "pilot" else ["discovery", "selection", "confirmation"],
                  "preparation_code_sha256": base_audit["preparation_code_sha256"],
                  "independent_audit_script_sha256": sha(Path(__file__)),
                  "per_view_zero_visual_state_latency_independently_proven": False,
                  "alignment_basis": "publisher own-view row correspondence, exact source/NPZ hashes, counts/PTS/player checks, calibrated score events, limited pilot HUD spotcheck",
                  "raw_data_rewritten": False, "thresholds_changed": False, "roles_or_clips_reselected": False,
                  "research_stage_completed": False, "exact_controlled_trajectories_available": False, "errors": [], "clips": []}
        write(output_audit, report)
        def check(record):
            return validate_record(record, retained, base_audit["timeline_calibrations"][record["match_id"]], protocol["targets"], publisher)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for number, result in enumerate(executor.map(check, records), 1):
                report["clips"].append(result)
                if number % 32 == 0:
                    report["verified_clips"] = number
                    write(output_audit, report)
                    print(json.dumps({"scope": scope, "verified_clips": number, "expected": len(records)}), flush=True)
        manifest = deepcopy(base_manifest)
        manifest.update(status="passed", records=deepcopy(records), protocol_sha256=registration["sha256"],
                        cohort_sha256=semantic_hash, split_manifest_sha256=registration["qualified_split_manifest_sha256"],
                        audit_report=str(ROOT / output_audit), original_full_audit_sha256=original_audit_hash,
                        original_full_audit_status="failed", excluded_match_ids=sorted(excluded),
                        scope="quality-qualified observational subset" if scope == "research" else "unchanged reserved engineering pilot")
        write(output_manifest, manifest)
        report.update(status="passed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      prepared_matches=len(expected_ids), prepared_clips=len(records), verified_clips=len(records),
                      match_counts=dict(Counter(retained[mid]["split"] for mid in expected_ids)),
                      clip_manifest_sha256=sha(ROOT / output_manifest))
        write(output_audit, report)
        outcomes.append({"scope": scope, "status": "passed", "matches": len(expected_ids), "clips": len(records),
                         "clip_manifest_sha256": report["clip_manifest_sha256"]})
    print(json.dumps(outcomes, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        for name in ("results/qualified_data_audit.json", "results/qualified_pilot_data_audit.json"):
            if (ROOT / name).exists():
                report = read(name)
                if report["status"] == "running":
                    report.update(status="failed", errors=[{"type": type(exc).__name__, "message": str(exc)}])
                    write(name, report)
        raise
