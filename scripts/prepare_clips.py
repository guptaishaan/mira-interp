#!/usr/bin/env python3
"""Prepare a frozen, metadata-selected observational cohort without changing roles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira_interp.clips import audit_physics_tracks, calibrate_timeline, decode_video, plan_clips, structural_validity_mask
from mira_interp.data import TARGET_NAMES


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def finish(args, report: dict, records: list[dict], shard_hashes: dict) -> int:
    records.sort(key=lambda row: (row["match_id"], row["clip_id"]))
    expected_clips = report["expected_matches"] * args.clips_per_match
    report.update(status="passed" if not report["errors"] and len(records) == expected_clips else "failed",
                  finished_at_utc=datetime.now(timezone.utc).isoformat(), prepared_clips=len(records),
                  expected_clips=expected_clips, prepared_matches=len({r["match_id"] for r in records}),
                  source_shard_hashes=shard_hashes)
    save(args.manifest, {"schema_version": 1, "status": report["status"], "records": records,
                         "roles": args.roles, "clips_per_match": args.clips_per_match,
                         "split_manifest_sha256": report["split_manifest_sha256"],
                         "cohort_sha256": report["cohort_sha256"], "audit_report": str(args.report),
                         "source_revision": report["source_revision"],
                         "preparation_code_sha256": report["preparation_code_sha256"],
                         "physical_counterfactuals": False, "observational_preparation_only": True})
    report["clip_manifest_sha256"] = sha(args.manifest)
    save(args.report, report)
    print(json.dumps({k: report[k] for k in ("status", "prepared_matches", "prepared_clips", "expected_clips", "errors")}), flush=True)
    return 0 if report["status"] == "passed" else 2


def parallel_prepare(args, entries: list[dict], report: dict) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    workdir = args.output_dir / "worker_reports" / f"{args.report.stem}_{run_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    groups = [[] for _ in range(args.workers)]
    shard_groups = {}
    for entry in sorted(entries, key=lambda e: (e["shard"], e["match_id"])):
        shard_groups.setdefault(entry["shard"], []).append(entry["match_id"])
    for ids in shard_groups.values():
        min(groups, key=len).extend(ids)
    outputs = []
    def launch(worker: int, match_ids: list[str]):
        worker_report, worker_manifest = workdir / f"audit_{worker}.json", workdir / f"manifest_{worker}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--index", str(args.index),
                   "--split-manifest", str(args.split_manifest), "--output-dir", str(args.output_dir),
                   "--download-report", str(args.download_report),
                   "--clips-per-match", str(args.clips_per_match), "--workers", "1",
                   "--worker-threads", str(max(1, 8 // args.workers)),
                   "--manifest", str(worker_manifest), "--report", str(worker_report), "--roles", *args.roles]
        for match_id in match_ids:
            command.extend(["--match-id", match_id])
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS=str(max(1, 8 // args.workers)), OPENBLAS_NUM_THREADS="1")
        with (workdir / f"worker_{worker}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        return worker, result.returncode
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(launch, worker, ids) for worker, ids in enumerate(groups) if ids]
        last_count = -1
        while not all(future.done() for future in futures):
            partial = []
            for worker in range(args.workers):
                path = workdir / f"audit_{worker}.json"
                if path.is_file():
                    partial.append(json.loads(path.read_text()))
            completed = sum(len(part.get("completed_match_ids", [])) for part in partial)
            report["progress"] = {"completed_matches": completed,
                                   "passed_clips": sum(len(part.get("clips", [])) for part in partial),
                                   "errors": sum(len(part.get("errors", [])) for part in partial),
                                   "workers": args.workers, "total_cpu_thread_cap": 8}
            save(args.report, report)
            if completed != last_count:
                print(json.dumps(report["progress"]), flush=True)
                last_count = completed
            time.sleep(2)
        outputs = [future.result() for future in futures]
    records, shard_hashes = [], {}
    for worker, returncode in outputs:
        worker_report, worker_manifest = workdir / f"audit_{worker}.json", workdir / f"manifest_{worker}.json"
        if not worker_report.is_file() or not worker_manifest.is_file():
            report["errors"].append({"worker": worker, "type": "incomplete_worker", "exit_code": returncode})
            continue
        part, manifest = json.loads(worker_report.read_text()), json.loads(worker_manifest.read_text())
        report["errors"].extend(part["errors"])
        if returncode != 0 and not part["errors"]:
            report["errors"].append({"worker": worker, "type": "worker_exit", "exit_code": returncode})
        for field in ("timeline_calibrations", "structural_eligibility"):
            report[field].update(part[field])
        report["clips"].extend(part["clips"])
        report["completed_match_ids"].extend(part["completed_match_ids"])
        records.extend(manifest["records"])
        shard_hashes.update(part.get("source_shard_hashes", {}))
    return finish(args, report, records, shard_hashes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("/data2/ishaangp/mira-interp/rocket-science/test/index.json"))
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "data/split_manifest.json")
    parser.add_argument("--download-report", type=Path, default=ROOT / "results/dataset_download.json")
    parser.add_argument("--output-dir", type=Path, default=Path("/data2/ishaangp/mira-interp/prepared"))
    parser.add_argument("--roles", nargs="+", default=["pilot"], choices=["pilot", "discovery", "selection", "confirmation"])
    parser.add_argument("--clips-per-match", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--match-id", action="append", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/clip_manifest.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results/pilot_data_audit.json")
    args = parser.parse_args()
    import torch
    if not 1 <= args.workers <= 4 or not 1 <= args.worker_threads <= 8:
        parser.error("workers must be 1..4 and worker threads 1..8")
    torch.set_num_threads(args.worker_threads)
    index = json.loads(args.index.read_text())
    split_manifest = json.loads(args.split_manifest.read_text())
    download_report = json.loads(args.download_report.read_text())
    if download_report["status"] != "PASS" or download_report["revision"] != split_manifest["source_revision"]:
        raise ValueError("full source download verification has not passed at the pinned revision")
    verified_shards = {Path(row["path"]).name: row["sha256"] for row in download_report["files"]
                       if row.get("verified") and row.get("sha256") == row.get("publisher_sha256")}
    if sha(args.index) not in split_manifest["input_index_sha256"]:
        raise ValueError("input index differs from the frozen manifest source")
    roles = {row["match_id"]: row["split"] for row in split_manifest["matches"]}
    entries = [entry for entry in index["entries"] if roles[entry["match_id"]] in args.roles]
    if args.match_id:
        entries = [entry for entry in entries if entry["match_id"] in set(args.match_id)]
    if not entries:
        raise ValueError("no requested-role matches in the pinned index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "scope": "observational per-view data integrity",
              "research_stage_completed": False, "roles": args.roles, "expected_matches": len(entries),
              "clips_per_match": args.clips_per_match, "source_revision": split_manifest["source_revision"],
              "split_manifest_sha256": sha(args.split_manifest), "cohort_sha256": split_manifest["sha256"],
              "download_report_sha256": sha(args.download_report),
              "errors": [], "clips": [], "exact_controlled_trajectories_available": False,
              "timeline_calibrations": {},
              "structural_eligibility": {},
              "completed_match_ids": [],
              "preparation_code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                  (Path(__file__).resolve(), ROOT / "src/mira_interp/clips.py", ROOT / "src/mira_interp/data.py")},
              "cross_view_shared_timestamp_claim": False,
              "selection_rule": "structurally valid plus clock-corrected metadata-live windows; disjoint length16 at20FPS; center of eight equal candidate quantiles",
              "alignment_gate": "frame counts, per-view PTS, player/team identities, nonfrozen physics, no demolitions; cross-view ball mean residual<=100uu after+-4frame lag",
              "resize": "upstream-equivalent bilinear antialias=True align_corners=False round/clamp uint8; 288x512"}
    save(args.report, report)
    if args.workers > 1:
        return parallel_prepare(args, entries, report)
    records = []
    shard_hashes = {}
    for entry in entries:
        match_id = entry["match_id"]
        role = roles[match_id]
        shard = args.index.parent / entry["shard"]
        if not shard.is_file():
            report["errors"].append({"match_id": match_id, "type": "missing_shard", "shard": entry["shard"]})
            report["completed_match_ids"].append(match_id)
            save(args.report, report)
            continue
        if str(shard) not in shard_hashes:
            shard_hashes[str(shard)] = sha(shard)
        if shard_hashes[str(shard)] != verified_shards.get(shard.name):
            report["errors"].append({"match_id": match_id, "type": "shard_hash_mismatch", "shard": entry["shard"]})
            report["completed_match_ids"].append(match_id)
            save(args.report, report)
            continue
        perspectives = sorted(entry["perspectives"], key=lambda p: p["player_id"])
        player_ids = [p["player_id"] for p in perspectives]
        teams = [p["team"] for p in perspectives]
        with tarfile.open(shard, "r:") as archive:
            members = {}
            for member in archive.getmembers():
                if member.isfile():
                    basename = Path(member.name).name
                    if basename in members:
                        raise ValueError("duplicate tar member basename")
                    members[basename] = member
            try:
                game_tracks = [[] for _ in range(4)]
                structural_masks = [[] for _ in range(4)]
                structural_counts = [{"invalid_state_frames": 0, "demolition_frames": 0, "frozen_frames": 0} for _ in range(4)]
                for chunk in entry.get("chunk_indices") or range(len(entry["chunk_frames"])):
                    for view in range(4):
                        name = f"{match_id}_c{chunk:05d}.p{view}.physics.jsonl"
                        lines = archive.extractfile(members[name]).read().splitlines()
                        if len(lines) != entry["chunk_frames"][chunk]:
                            raise ValueError("individual physics chunk row count disagrees with source index")
                        decoded_physics = [json.loads(line) for line in lines]
                        game_tracks[view].extend(frame["game"] for frame in decoded_physics)
                        valid, counts = structural_validity_mask(decoded_physics, player_ids, teams, view)
                        structural_masks[view].extend(valid.tolist())
                        for flag, value in counts.items():
                            structural_counts[view][flag] += value
                calibration = calibrate_timeline(entry, game_tracks)
                origins = [view["calibrated_origin_seconds"] for view in calibration["views"]]
                report["timeline_calibrations"][match_id] = calibration
                structural_valid = np.all(np.asarray(structural_masks, dtype=bool), axis=0)
                report["structural_eligibility"][match_id] = {
                    "source_frames": len(structural_valid), "all_views_valid_frames": int(structural_valid.sum()),
                    "mask_sha256": hashlib.sha256(structural_valid.astype(np.uint8).tobytes()).hexdigest(),
                    "per_view_exclusion_counts": structural_counts,
                    "exclusion_uses_target_magnitude_or_model_performance": False}
                plan = plan_clips(entry, args.clips_per_match, calibrated_origins=origins,
                                  source_valid_mask=structural_valid)
            except Exception as exc:
                report["errors"].append({"match_id": match_id, "type": type(exc).__name__,
                                         "stage": "source_clock_calibration", "message": str(exc)[:300]})
                report["completed_match_ids"].append(match_id)
                save(args.report, report)
                continue
            for number, planned in enumerate(plan):
                clip_id = f"{match_id}_clip{number:02d}"
                key = f"{match_id}_c{planned['chunk_index']:05d}"
                local = list(range(planned["local_start"], planned["local_start"] + 16))
                try:
                    def read(suffix):
                        member = members[f"{key}.{suffix}"]
                        return archive.extractfile(member).read()
                    meta = json.loads(read("meta.json"))
                    if meta["match_id"] != match_id or meta["chunk"] != planned["chunk_index"] or meta["player_ids"] != player_ids or meta["teams"] != teams:
                        raise ValueError("chunk metadata disagrees with the pinned index")
                    physics, action_tracks, videos, component_hashes = [], [], [], {}
                    for view in range(4):
                        blobs = {suffix: read(f"p{view}.{suffix}") for suffix in ("physics.jsonl", "jsonl", "mp4")}
                        component_hashes.update({f"p{view}.{suffix}": hashlib.sha256(blob).hexdigest() for suffix, blob in blobs.items()})
                        phys_lines = blobs["physics.jsonl"].splitlines()
                        acts = blobs["jsonl"].decode().splitlines()
                        expected = entry["chunk_frames"][planned["chunk_index"]]
                        if len(phys_lines) != expected or len(acts) != expected:
                            raise ValueError("source JSONL counts disagree with indexed video frames")
                        physics.append([json.loads(phys_lines[i]) for i in local])
                        action_tracks.append([acts[i] for i in local])
                        videos.append(blobs["mp4"])
                    targets, actions, physical_audit = audit_physics_tracks(physics, action_tracks, player_ids, teams)
                    decoded = [decode_video(video, local, expected) for video in videos]
                    frames = np.stack([values[0] for values in decoded])
                    video_pts = np.stack([values[1] for values in decoded])
                    indices = np.arange(planned["source_start_frame"], planned["source_start_frame"] + 16)
                    timestamps = np.stack([indices / 20 + origin for origin in origins])
                    path = args.output_dir / role / f"{clip_id}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, frames=frames, actions=actions, targets=targets,
                                        source_frame_indices=indices, timestamps=timestamps,
                                        video_pts=video_pts, player_ids=np.asarray(player_ids),
                                        target_names=np.asarray(TARGET_NAMES))
                    record = {"clip_id": clip_id, "match_id": match_id, "role": role,
                              "artifact_path": str(path), "sha256": sha(path), "player_ids": player_ids,
                              "teams": teams, "chunk_index": planned["chunk_index"], "plan": planned,
                              "source_shard": entry["shard"], "source_shard_sha256": shard_hashes[str(shard)],
                              "source_component_sha256": component_hashes, "source_revision": split_manifest["source_revision"],
                              "frames_shape": list(frames.shape), "targets_shape": list(targets.shape),
                              "source_fps": 20, "timestamp_semantics": "per-view score-anchor calibrated origin + original frame index /20; origin residual<=0.1s; no exact shared-time assertion",
                              "timeline_calibration": calibration,
                              "physical_audit": physical_audit, "video_audit": [values[2] for values in decoded]}
                    save(path.with_suffix(".json"), record)
                    records.append(record)
                    report["clips"].append({"clip_id": clip_id, "role": role, "status": "passed"})
                except Exception as exc:
                    report["errors"].append({"clip_id": clip_id, "match_id": match_id,
                                             "type": type(exc).__name__, "message": str(exc)[:300]})
            report["completed_match_ids"].append(match_id)
            save(args.report, report)
        print(json.dumps({"match_id": match_id, "role": role, "prepared_clips": sum(r["match_id"] == match_id for r in records),
                          "cumulative_errors": len(report["errors"])}), flush=True)
    return finish(args, report, records, shard_hashes)


if __name__ == "__main__":
    raise SystemExit(main())
