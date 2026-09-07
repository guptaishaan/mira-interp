#!/usr/bin/env python3
"""Freeze a separate observational cohort after complete source-quality auditing.

No position/velocity arrays, research activations, or probe scores are read.
This never changes the original registration or turns its failed audit into PASS.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_HASH = "2437f915ceb14a10c33a6b003b7273ee6ba135e8495d4b15b03ff72f0edbd6a3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def freeze(name, value):
    with (ROOT / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    names = ["data/qualified_split_manifest.json", "configs/observational_probe_v2.json",
             "results/observational_registration_v2.json"]
    if any((ROOT / name).exists() for name in names):
        raise RuntimeError("Registration is immutable; an output already exists")
    assert sha(ROOT / "configs/observational_probe_v1.json") == V1_HASH
    protocol = read("configs/observational_probe_v1.json")
    splits = read("data/split_manifest.json")
    audit = read("results/cohort_data_audit.json")
    clips = read("data/clip_manifest.json")
    assert audit["status"] == "failed" and audit["finished_at_utc"]
    assert audit["split_manifest_sha256"] == sha(ROOT / "data/split_manifest.json")
    assert audit["clip_manifest_sha256"] == sha(ROOT / "data/clip_manifest.json")
    assigned = {row["match_id"]: row for row in splits["matches"]}
    original_ids = {mid for mid, row in assigned.items() if row["split"] != "pilot"}
    assert len(original_ids) == 61
    assert set(audit["completed_match_ids"]) == original_ids
    assert len(audit["completed_match_ids"]) == 61
    assert all(error["stage"] == "source_clock_calibration" for error in audit["errors"])
    excluded = {error["match_id"] for error in audit["errors"]}
    assert len(excluded) == len(audit["errors"]) == 8
    qualified = original_ids - excluded
    counts = Counter(row["match_id"] for row in clips["records"])
    assert set(counts) == qualified and set(counts.values()) == {8}
    assert audit["prepared_matches"] == 53 and audit["prepared_clips"] == 424
    role_counts = Counter(assigned[mid]["split"] for mid in qualified)
    assert dict(role_counts) == {"discovery": 31, "selection": 11, "confirmation": 11}
    attrition = [{**error, "original_role": assigned[error["match_id"]]["split"]}
                 for error in sorted(audit["errors"], key=lambda row: row["match_id"])]
    parent_hashes = {name: sha(ROOT / name) for name in (
        "configs/observational_probe_v1.json", "data/split_manifest.json",
        "data/clip_manifest.json", "results/cohort_data_audit.json")}
    timestamp = datetime.now(timezone.utc).isoformat()
    amendment = {
        "registered_at_utc": timestamp,
        "reason": "Original 61-match quality gate failed: 7 inconsistent score clocks and 1 insufficient-anchor match. Separate quality-qualified observational cohort before any research capture or fit.",
        "original_protocol_status": "FAILED_DATA_GATE",
        "parent_artifact_sha256": parent_hashes,
        "excluded_matches": attrition,
        "inclusion_rule": "Every original research match that passes all unchanged source-quality gates and all eight prescribed clips; original roles retained, no replacement or rebalancing.",
        "timing_gates": {"minimum_goal_pairs_per_view": 3, "maximum_clock_residual_seconds": 0.1},
        "research_capture_or_probe_performance_used": False,
        "limits": [
            "The estimand is the quality-qualified subset, not all Rocket Science matches.",
            "Clock qualification can bias toward matches with at least three recorded goals and consistent clients.",
            "Frozen-state and demolition filtering further restricts the sampled live-play population.",
            "31/11/11 matches satisfy operational minima; this is not an a priori power calculation.",
            "Own-view one-to-one video/state alignment is a publisher assumption plus count/PTS/identity checks; zero visual-state latency is not independently established.",
            "Shared players can occur across matches; match-disjoint does not mean player-disjoint.",
            "Recorded observations are not single-variable simulator counterfactuals.",
        ],
    }
    subset = deepcopy(splits)
    subset["matches"] = [row for row in splits["matches"] if row["match_id"] not in excluded]
    subset["match_counts"] = dict(role_counts)
    subset["assignment"] = "Original roles retained after complete source-quality qualification; no replacement"
    subset["sha256"] = hashlib.sha256(json.dumps(subset["matches"], sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
    subset["amendment"] = amendment
    subset["research_dataset_ready"] = False  # Independent qualified-data audit is still required.
    protocol["version"] = 2
    protocol["scope"] = "Quality-qualified observational, teacher-forced physical-state reconstruction decoding only"
    protocol["registration_basis"] = "After full source-quality audit, before any research activation capture, probe fitting, selection, or confirmation"
    protocol["cohort"].update(dict(role_counts))
    protocol["cohort"]["balanced_manifest_sha256"] = subset["sha256"]
    protocol["cohort"]["source"] = "Exactly all quality-passing members of the original official-test cohort"
    protocol["cohort"]["amendment"] = amendment
    protocol["reason_for_adjustment"] = amendment["reason"]
    protocol["reporting"]["population"] = "Only quality-qualified matches and eligible live clips; original 61-match plan failed its data gate"
    freeze(names[0], subset)
    freeze(names[1], protocol)
    freeze(names[2], {"state": "REGISTERED_BEFORE_RESEARCH_CAPTURE_AND_FITTING",
                      "registered_at_utc": timestamp, "protocol_path": names[1],
                      "sha256": sha(ROOT / names[1]), "qualified_split_manifest_path": names[0],
                      "qualified_split_manifest_sha256": sha(ROOT / names[0]),
                      "qualified_cohort_sha256": subset["sha256"],
                      "parent_artifact_sha256": parent_hashes,
                      "registration_script_sha256": sha(Path(__file__)),
                      "original_protocol_status": "FAILED_DATA_GATE"})
    print(json.dumps(read(names[2]), indent=2))


if __name__ == "__main__":
    main()
