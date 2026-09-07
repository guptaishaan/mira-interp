#!/usr/bin/env python3
"""Describe failed source-clock fits using discrete scoreboard events only.

Diagnostic output does not relax the preparation gate or fit target variables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "results/cohort_data_audit.json")
    parser.add_argument("--index", type=Path, default=Path("/data2/ishaangp/mira-interp/rocket-science/test/index.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/source_clock_failures.json")
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit["status"] == "running":
        raise ValueError("wait for the complete cohort audit before summarizing failures")
    index = json.loads(args.index.read_text())
    entries = {entry["match_id"]: entry for entry in index["entries"]}
    frozen = json.loads((ROOT / "data/split_manifest.json").read_text())
    roles = {entry["match_id"]: entry["split"] for entry in frozen["matches"]}
    failed = sorted({error["match_id"] for error in audit["errors"] if error.get("stage") == "source_clock_calibration"})
    findings = []
    for match_id in failed:
        entry = entries[match_id]
        perspectives = sorted(entry["perspectives"], key=lambda p: p["player_id"])
        per_view = []
        with tarfile.open(args.index.parent / entry["shard"], "r:") as archive:
            members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
            for view in range(4):
                anchors = sorted((a for a in perspectives[view]["anchors"] if a["event_name"] == "GoalScored"),
                                 key=lambda a: a["master_sec"])
                previous, source_frame, pairs, problems = None, 0, [], []
                score_hash = hashlib.sha256()
                for chunk, expected in zip(entry["chunk_indices"], entry["chunk_frames"]):
                    member = members[f"{match_id}_c{chunk:05d}.p{view}.physics.jsonl"]
                    lines = archive.extractfile(member).read().splitlines()
                    if len(lines) != expected:
                        problems.append("chunk_row_count_mismatch")
                    for line in lines:
                        state = json.loads(line)["game"]
                        score_hash.update(json.dumps(state, sort_keys=True).encode())
                        total = int(state["score_blue"]) + int(state["score_orange"])
                        if previous is not None and total != previous:
                            if total != previous + 1 or not 0 < total <= len(anchors):
                                problems.append("ambiguous_score_increment")
                            else:
                                seconds = source_frame / 20
                                anchor_seconds = float(anchors[total - 1]["master_sec"])
                                pairs.append({"score_total": total, "source_frame": source_frame,
                                              "source_seconds": seconds, "anchor_seconds": anchor_seconds,
                                              "offset_seconds": anchor_seconds - seconds})
                        previous = total
                        source_frame += 1
                median = float(np.median([p["offset_seconds"] for p in pairs])) if pairs else None
                for pair in pairs:
                    pair["residual_seconds"] = pair["offset_seconds"] - median
                maximum = max((abs(p["residual_seconds"]) for p in pairs), default=None)
                per_view.append({"view": view, "player_id": perspectives[view]["player_id"],
                                 "paired_goals": len(pairs), "median_origin_seconds": median,
                                 "max_absolute_residual_seconds": maximum, "problems": problems,
                                 "passes_original_clock_gate": not problems and len(pairs) >= 3 and maximum <= .1,
                                 "goal_pairs": pairs, "score_track_sha256": score_hash.hexdigest(),
                                 "anchor_sha256": hashlib.sha256(json.dumps(anchors, sort_keys=True).encode()).hexdigest()})
        findings.append({"match_id": match_id, "role": roles[match_id], "views": per_view,
                         "source_shard": entry["shard"],
                         "source_shard_sha256": audit["source_shard_hashes"][str(args.index.parent / entry["shard"])]})
    output = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "diagnostic_only": True, "target_positions_velocities_or_model_outputs_analyzed": False,
              "cohort_gate_relaxed": False, "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
              "failed_matches": len(findings), "findings": findings}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"diagnosed_matches": len(findings), "output": str(args.output), "cohort_gate_relaxed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
