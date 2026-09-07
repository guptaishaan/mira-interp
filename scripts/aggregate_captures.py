#!/usr/bin/env python3
"""Independently audit complete actual-data captures and assemble the probe NPZ.

A passed aggregate never changes the status of an earlier failed data cohort.
All source labels, per-row identities, hashes, split roles, and gates are checked
before an analysis archive is published. No GPU or probe fitting is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIRA_REV = "3d739ec2d31daf83559d33eb01727cea48fe90f7"
DINO_REV = "6876159a11b4df116f30f667f8c9888617df0751"
MODEL_REV = "d58ee2f9bca27289554c1e652943dc0e539e8971"
APPROVED_PROTOCOL_HASHES = {
    1: "2437f915ceb14a10c33a6b003b7273ee6ba135e8495d4b15b03ff72f0edbd6a3",
    2: "e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2",
}
EXPECTED_ASSETS = {
    "checkpoint-90000/checkpoint.pth": "13a684a5f64613ae640d846b25cb16ac3c4059f97c9ccedacb1121f5241a5be0",
    "codec/checkpoint-125000/checkpoint.pth": "0b56cc07c878ab231a2c1faea155f20b0d49dbc378e3b67e84b2c7026eec59f9",
}
CODE_FILES = ["scripts/capture_observations.py", "src/mira_interp/model_loading.py", "src/mira_interp/pretrained.py", "src/mira_interp/instrumentation.py"]
ROLES = ("discovery", "selection", "confirmation")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_loading(loading):
    require(loading.get("model_revision") == MODEL_REV, "Unexpected pretrained model revision")
    require(loading.get("model_repo") == "alakazamworld/mira-mini-4p", "Unexpected model repository")
    require(loading.get("source_revisions") == {"mira": MIRA_REV, "dinov3": DINO_REV}, "Unexpected architecture source pins")
    for key, count in [("codec_strict_load", 936), ("world_model_strict_load", 1183)]:
        row = loading.get(key, {})
        require(row.get("passed") is True and row.get("state_tensors") == count, f"Strict loading gate missing: {key}")
        require(not any(row.get(name) for name in ["missing_keys", "unexpected_keys", "shape_mismatch"]), f"Non-strict loading recorded: {key}")
    pair = loading.get("codec_pair_integrity", {})
    require(pair.get("passed") is True and pair.get("compared_tensors") == 936 and pair.get("mismatched_keys") == [], "Codec/normalization pairing did not pass")
    assets = {row["name"]: row["sha256"] for row in loading.get("verified_assets", [])}
    require(assets == EXPECTED_ASSETS, "Verified checkpoint hashes differ")


def verify_record(clip, source_record, protocol, protocol_hash, code_hashes, manifest_directory):
    """Independent row/label audit; returns read-back arrays and compact provenance."""
    clip_id = source_record["clip_id"]
    for key in ["clip_id", "match_id", "role", "player_ids"]:
        require(clip.get(key) == source_record[key], f"Source identity mismatch {clip_id}: {key}")
    require(clip.get("protocol_sha256") == protocol_hash, f"Clip protocol mismatch: {clip_id}")
    require(clip.get("code_sha256") == code_hashes, f"Clip code mismatch: {clip_id}")
    require(clip.get("script_sha256") == code_hashes["scripts/capture_observations.py"], f"Clip script hash mismatch: {clip_id}")
    require(clip.get("model_revision") == MODEL_REV, f"Clip model revision mismatch: {clip_id}")
    require({row["name"]: row["sha256"] for row in clip.get("verified_checkpoint_assets", [])} == EXPECTED_ASSETS, f"Clip checkpoint hash mismatch: {clip_id}")
    require(clip.get("rows") == 32 and clip.get("row_order") == "view_major_then_latent_time", f"Clip row contract mismatch: {clip_id}")
    require(clip.get("target_alignment") == "own_view_source_frame2t+1", f"Clip label alignment claim mismatch: {clip_id}")
    require(clip.get("site_names") == protocol["capture"]["sites"], f"Clip site order mismatch: {clip_id}")
    expected_seed = int.from_bytes(hashlib.sha256(f"mira-observation-v1\0{clip_id}".encode()).digest()[:8], "big") % (2**63 - 1)
    require(clip.get("noise_seed") == expected_seed and clip.get("tau") == 0.5, f"Clip noise protocol mismatch: {clip_id}")
    metadata = clip.get("metadata", {})
    require(metadata.get("condition") == "teacher_forced_noisy_observed_target_no_intervention", "Incorrect experiment scope")
    require(metadata.get("sample_ids") == [clip_id] and metadata.get("latent_frame_indices") == list(range(8)), "Capture sample/time metadata mismatch")
    require(metadata.get("noise_seed") == expected_seed and metadata.get("noise_id") == clip.get("noise_sha256"), "Capture noise identity mismatch")
    require(metadata.get("tau") == [0.5] * 8 and metadata.get("tau_shape") == [1, 8, 1, 1, 1], "Capture tau metadata mismatch")
    require(metadata.get("cache_mode") == "fresh", "Unexpected inference cache")
    source_path = Path(source_record["artifact_path"])
    if not source_path.is_absolute():
        source_path = manifest_directory / source_path
    capture_path = Path(clip["output_path"])
    require(digest(source_path) == source_record["sha256"] == clip["input_sha256"], f"Prepared input hash mismatch: {clip_id}")
    require(digest(capture_path) == clip["output_sha256"], f"Capture hash mismatch: {clip_id}")
    require(capture_path.stat().st_size == clip["output_bytes"], f"Capture length mismatch: {clip_id}")
    with np.load(capture_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    for key, shape, dtype in [("X", (32, 17, 2048), np.float16), ("y", (32, 30), np.float64), ("codec_X", (32, 32), np.float16), ("RGB_X", (32, 192), np.float16)]:
        require(values[key].shape == shape and values[key].dtype == dtype and np.isfinite(values[key]).all(), f"Invalid array {key}: {clip_id}")
    require(values["sites"].tolist() == protocol["capture"]["sites"] and values["target_names"].tolist() == protocol["targets"], f"NPZ column identity mismatch: {clip_id}")
    require(values["match_ids"].tolist() == [source_record["match_id"]] * 32, "Per-row match IDs disagree")
    require(values["split"].tolist() == [source_record["role"]] * 32 and values["clip_ids"].tolist() == [clip_id] * 32, "Per-row clip/split IDs disagree")
    require(np.array_equal(values["view_index"], np.repeat(np.arange(4), 8)), "Rows are not view-major")
    require(np.array_equal(values["latent_frame_index"], np.tile(np.arange(8), 4)), "Latent-time order differs")
    require(values["player_ids"].tolist() == source_record["player_ids"], "Player identity ordering differs")
    with np.load(source_path, allow_pickle=False) as source:
        require(source["targets"].shape == (4, 16, 30), "Source physical labels have the wrong shape")
        require(np.array_equal(values["y"], source["targets"][:, 1::2].reshape(32, 30)), f"Labels are not exact own-view pair-final targets: {clip_id}")
        require(source["target_names"].tolist() == protocol["targets"], "Source target columns differ")
        require(source["player_ids"].tolist() == source_record["player_ids"], "Source player ordering differs")
        require(source["timestamps"].shape == (4, 16), "Source timestamps are not per-view")
        require(np.array_equal(values["timestamps"], source["timestamps"][:, 1::2].reshape(32)), f"Timestamp join mismatch: {clip_id}")
        indices = source["source_frame_indices"]
        if indices.shape == (16,):
            indices = np.broadcast_to(indices, (4, 16))
        require(indices.shape == (4, 16), "Unexpected source frame index shape")
        require(np.array_equal(values["source_frame_index"], indices[:, 1::2].reshape(32)), f"Source-frame join mismatch: {clip_id}")
        require(np.isfinite(source["timestamps"]).all() and np.all(np.diff(source["timestamps"], axis=1) > 0), "Invalid source timing")
    require(np.isfinite(values["timestamps"]).all(), "Nonfinite saved timestamps")
    values["player_id"] = np.repeat(values.pop("player_ids"), 8)
    values["canonical_player_ids"] = np.tile(np.asarray(source_record["player_ids"]), (32, 1))
    values["chunk_index"] = np.full(32, source_record["chunk_index"], dtype=np.int64)
    summary = {key: clip[key] for key in ["clip_id", "match_id", "role", "output_path", "input_sha256", "output_sha256", "noise_seed", "noise_sha256"]}
    summary.update(rows=32, exact_labels_checked=True, exact_timestamps_checked=True, exact_row_identity_checked=True)
    return values, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--registration-record", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--pilot-data-manifest", type=Path, required=True)
    parser.add_argument("--pilot-data-audit", type=Path, required=True)
    parser.add_argument("--capture-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--original-data-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = {"status": "running", "stage": "prerequisites", "gpu_used": False, "clips": [], "scope": "observational teacher-forced physical-state reconstruction decoding"}

    def stage(name):
        report.update(stage=name, elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(args.report, report)
        print(f"[{report['elapsed_seconds']:.1f}s] {name}: {len(report['clips'])} clips verified", flush=True)

    stage("prerequisites")
    try:
        require(not args.output.exists(), "Analysis output already exists; use a new explicit output path")
        protocol_hash = digest(args.protocol)
        require(protocol_hash == args.protocol_sha256 and len(protocol_hash) == 64, "Caller-supplied protocol hash mismatch")
        protocol, registration = read_json(args.protocol), read_json(args.registration_record)
        require(APPROVED_PROTOCOL_HASHES.get(protocol.get("version")) == protocol_hash, "Protocol hash has not been approved")
        require(registration.get("sha256") == protocol_hash, "Frozen registration record does not match protocol")
        require(registration.get("state", "").startswith("REGISTERED_BEFORE_RESEARCH_CAPTURE"), "Protocol was not registered before research capture")
        manifest, audit, split = read_json(args.data_manifest), read_json(args.data_audit), read_json(args.split_manifest)
        require(audit.get("status") == "passed" and manifest.get("status") == "passed", "The supplied data cohort did not pass; failed original audits cannot be promoted")
        require(audit.get("clip_manifest_sha256") == digest(args.data_manifest), "Data audit is not tied to this clip manifest")
        split_hash = digest(args.split_manifest)
        require(audit.get("split_manifest_sha256") == split_hash == manifest.get("split_manifest_sha256"), "Data split hashes differ")
        cohort_hash = hashlib.sha256(json.dumps(split["matches"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(cohort_hash == split["sha256"] == protocol["cohort"]["balanced_manifest_sha256"] == audit.get("cohort_sha256") == manifest.get("cohort_sha256"), "Registered cohort identity differs")
        require(protocol["source_revision"] == manifest.get("source_revision"), "Dataset revision mismatch")
        if "source_revision" in audit:
            require(audit["source_revision"] == protocol["source_revision"], "Data audit dataset revision mismatch")
        if protocol.get("version", 1) > 1:
            require(audit.get("protocol_sha256") == manifest.get("protocol_sha256") == protocol_hash, "Qualified data is not bound to this protocol")
            require(audit.get("registration_sha256") == digest(args.registration_record), "Qualified data uses another registration record")
        require(protocol.get("model_revision") == MODEL_REV, "Protocol model pin differs")
        assignments = {row["match_id"]: row for row in split["matches"]}
        require(len(assignments) == len(split["matches"]), "Duplicate match assignment")
        expected_matches = {mid: row for mid, row in assignments.items() if row["split"] in ROLES}
        counts = Counter(row["split"] for row in expected_matches.values())
        require(dict(counts) == {role: protocol["cohort"][role] for role in ROLES}, "Registered match counts differ")
        records = sorted(manifest["records"], key=lambda row: row["clip_id"])
        records_by_id = {row["clip_id"]: row for row in records}
        require(len(records_by_id) == len(records), "Duplicate prepared clip ID")
        per_match = Counter()
        for row in records:
            require(row["match_id"] in expected_matches, "Unregistered or pilot match in analysis input")
            assigned = expected_matches[row["match_id"]]
            require(row["role"] == assigned["split"] and row["player_ids"] == assigned["player_ids"], "Source role/player assignment differs")
            per_match[row["match_id"]] += 1
            require(row.get("source_revision") == protocol["source_revision"], "Clip source revision mismatch")
            require(len(row.get("video_audit", [])) == 4 and all(v.get("pts_monotonic_and_uniform") is True for v in row["video_audit"]), "Missing per-view video timing audit")
            physical = row.get("physical_audit", {})
            require(physical.get("all_player_identities_checked") is True and physical.get("all_team_ids_checked") is True, "Missing physical identity audit")
            require(physical.get("frozen_transitions") == 0 and physical.get("demolished_entity_frames") == 0, "Source physical eligibility gate failed")
        require(set(per_match) == set(expected_matches) and all(n == 8 for n in per_match.values()), "Missing match or incomplete eight-clip cohort")
        require(audit.get("expected_clips") == audit.get("prepared_clips") == len(records), "Data audit clip counts are incomplete")
        require(audit.get("expected_matches") == audit.get("prepared_matches") == len(expected_matches), "Data audit match counts are incomplete")
        audited_clip_ids = [row["clip_id"] for row in audit.get("clips", []) if row.get("status") == "passed"]
        require(len(audited_clip_ids) == len(records) and set(audited_clip_ids) == set(records_by_id), "Passed data audit lacks exact clip coverage")
        require(not audit.get("errors"), "Passed data audit retains unresolved errors")
        if protocol.get("version", 1) > 1:
            require(args.original_data_audit is not None, "Amended subset requires the original cohort audit provenance")
        if args.original_data_audit is not None:
            original = read_json(args.original_data_audit)
            original_hash = digest(args.original_data_audit)
            original_passed_clips = {row["clip_id"] for row in original.get("clips", []) if row.get("status") == "passed"}
            require(set(records_by_id) <= original_passed_clips, "Qualified records were not all passed in the original data audit")
            report["original_data_audit"] = {"path": str(args.original_data_audit), "sha256": original_hash, "status_preserved": original.get("status"), "expected_matches": original.get("expected_matches"), "prepared_matches": original.get("prepared_matches")}
            require(original.get("status") == "failed", "The original failed cohort audit must retain its failed status")
            require(audit.get("original_full_audit_sha256") == manifest.get("original_full_audit_sha256") == original_hash, "Qualified audit original-cohort link differs")
            require(audit.get("original_full_audit_status") == manifest.get("original_full_audit_status") == "failed", "Qualified cohort silently changed original cohort status")
            parent_hashes = protocol["cohort"]["amendment"]["parent_artifact_sha256"]
            require(parent_hashes.get("results/cohort_data_audit.json") == original_hash, "Original failure differs from the registered amendment")
            excluded = {row["match_id"] for row in protocol["cohort"]["amendment"]["excluded_matches"]}
            require(excluded == set(audit.get("excluded_match_ids", [])) == set(manifest.get("excluded_match_ids", [])) and not excluded.intersection(expected_matches), "Qualified exclusion identities differ")
        code_hashes = {name: digest(ROOT / name) for name in CODE_FILES}
        pilot = read_json(args.pilot_report)
        pilot_hash = digest(args.pilot_report)
        require(pilot.get("status") == "passed" and pilot.get("pilot") is True, "GPU pilot did not pass")
        require(pilot.get("protocol_sha256") == protocol_hash and pilot.get("code_sha256") == code_hashes, "Pilot code/protocol mismatch")
        pilot_manifest, pilot_audit = read_json(args.pilot_data_manifest), read_json(args.pilot_data_audit)
        require(pilot_manifest.get("status") == pilot_audit.get("status") == "passed", "Reserved pilot data did not pass")
        require(pilot.get("manifest_sha256") == pilot_audit.get("clip_manifest_sha256") == digest(args.pilot_data_manifest), "Pilot data manifest differs")
        require(pilot.get("data_audit_sha256") == digest(args.pilot_data_audit), "Pilot data audit differs")
        require(pilot.get("split_manifest_sha256") == pilot_audit.get("split_manifest_sha256") == pilot_manifest.get("split_manifest_sha256") == split_hash, "Pilot split identity differs")
        require(pilot.get("cohort_sha256") == pilot_audit.get("cohort_sha256") == pilot_manifest.get("cohort_sha256") == cohort_hash, "Pilot cohort identity differs")
        if protocol.get("version", 1) > 1:
            require(pilot_audit.get("protocol_sha256") == pilot_manifest.get("protocol_sha256") == protocol_hash, "Pilot qualified protocol differs")
        pilot_records = sorted(pilot_manifest["records"], key=lambda row: row["clip_id"])
        require(len(pilot_records) == 8 and len({row["clip_id"] for row in pilot_records}) == 8, "Expected eight prepared reserved-pilot clips")
        for row in pilot_records:
            require(row["match_id"] in assignments and assignments[row["match_id"]]["split"] == row["role"] == "pilot", "Pilot source is not reserved")
            require(row["player_ids"] == assignments[row["match_id"]]["player_ids"], "Pilot player identities differ")
        verify_loading(pilot.get("model_loading", {}))
        require(pilot.get("completed_clips") == 1 and len(pilot.get("clips", [])) == 1, "Expected one reserved GPU pilot")
        _, pilot_row_audit = verify_record(pilot["clips"][0], pilot_records[0], protocol, protocol_hash, code_hashes, args.pilot_data_manifest.parent)
        report["pilot_row_audit"] = pilot_row_audit
        controls = pilot["clips"][0].get("controls", {})
        require(all(controls.get(name) is True for name in ["noop_bitwise_equal", "repeat_bitwise_equal", "full_tensor_pooling_passed"]), "Pilot forward/pooling control failed")
        require(len(controls.get("full_tensor_pooling", [])) == 17, "Pilot reference does not cover all sites")
        require({row["site"] for row in controls["full_tensor_pooling"]} == set(range(17)), "Pilot reference sites differ")
        for row in controls["full_tensor_pooling"]:
            require(digest(Path(row["path"])) == row["sha256"], "Pilot reference tensor hash differs")
        collected, worker_indices, loading_reference = {}, set(), None
        worker_paths = sorted(args.capture_reports)
        for worker_path in worker_paths:
            worker = read_json(worker_path)
            require(worker.get("status") == "passed" and worker.get("pilot") is False, "Incomplete or pilot capture worker supplied")
            require(worker.get("workers") == len(worker_paths), "Capture worker count differs")
            index = worker.get("worker_index")
            require(isinstance(index, int) and index not in worker_indices and 0 <= index < len(worker_paths), "Duplicate/invalid worker index")
            worker_indices.add(index)
            require(worker.get("protocol_sha256") == protocol_hash and worker.get("code_sha256") == code_hashes, "Worker code/protocol mismatch")
            require(worker.get("pilot_report_sha256") == pilot_hash, "Worker used a different pilot gate")
            require(worker.get("manifest_sha256") == digest(args.data_manifest) and worker.get("data_audit_sha256") == digest(args.data_audit) and worker.get("split_manifest_sha256") == split_hash, "Worker input audit chain differs")
            loading = worker.get("model_loading", {})
            verify_loading(loading)
            if loading_reference is None:
                loading_reference = loading
            else:
                require(loading == loading_reference, "Workers loaded different model sources/configs/checkpoints")
            expected_ids = {row["clip_id"] for i, row in enumerate(records) if i % len(worker_paths) == index}
            actual_ids = [row["clip_id"] for row in worker.get("clips", [])]
            require(len(actual_ids) == len(expected_ids) and set(actual_ids) == expected_ids, "Worker has duplicate/missing/wrong modulo clip assignment")
            require(worker.get("completed_clips") == worker.get("expected_clips") == len(expected_ids) and worker.get("rows") == 32 * len(expected_ids), "Worker counts disagree")
            for clip in worker["clips"]:
                require(clip["clip_id"] not in collected, "Duplicate clip across workers")
                collected[clip["clip_id"]] = clip
        require(set(collected) == set(records_by_id), "Capture workers do not cover the exact registered clip set")
        frozen_paths = [args.protocol, args.registration_record, args.data_manifest, args.data_audit, args.split_manifest, args.pilot_report, args.pilot_data_manifest, args.pilot_data_audit, Path(__file__), *worker_paths]
        if args.original_data_audit is not None:
            frozen_paths.append(args.original_data_audit)
        frozen_inputs = {str(path): digest(path) for path in frozen_paths}
        report.update(registration_sha256=protocol_hash, registration_record_sha256=digest(args.registration_record), data_manifest_sha256=digest(args.data_manifest), data_audit_sha256=digest(args.data_audit), split_manifest_sha256=split_hash, cohort_sha256=cohort_hash, pilot_report_sha256=pilot_hash, pilot_data_audit_sha256=digest(args.pilot_data_audit), pilot_data_manifest_sha256=digest(args.pilot_data_manifest), capture_report_sha256={str(path): digest(path) for path in worker_paths}, capture_code_sha256=code_hashes, aggregate_script_sha256=digest(Path(__file__)), match_counts=dict(counts), expected_clips=len(records), alignment_basis=audit.get("alignment_basis"), per_view_zero_visual_state_latency_independently_proven=False)
        chunks, seen_rows = {}, set()
        for ordinal, record in enumerate(records):
            values, summary = verify_record(collected[record["clip_id"]], record, protocol, protocol_hash, code_hashes, args.data_manifest.parent)
            for i in range(32):
                identity = (record["match_id"], int(values["view_index"][i]), int(values["source_frame_index"][i]))
                require(identity not in seen_rows, "Duplicate source view/frame row across clips")
                seen_rows.add(identity)
            for key, value in values.items():
                if key not in {"sites", "target_names"}:
                    chunks.setdefault(key, []).append(value)
            summary.update(row_start=ordinal * 32, row_stop=(ordinal + 1) * 32)
            report["clips"].append(summary)
            if ordinal % 25 == 0:
                stage("audit_source_and_capture_rows")
        arrays = {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}
        arrays.update(sites=np.asarray(protocol["capture"]["sites"]), target_names=np.asarray(protocol["targets"]))
        require(arrays["X"].shape == (len(records) * 32, 17, 2048), "Aggregate residual shape mismatch")
        require(len(seen_rows) == len(records) * 32, "Aggregate row count mismatch")
        require(all(np.count_nonzero(arrays["match_ids"] == mid) == 256 for mid in expected_matches), "Aggregate match row count mismatch")
        stage("write_analysis_archive")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(temporary, allow_pickle=False) as saved:
            for name, value in arrays.items():
                require(np.array_equal(saved[name], value), f"Aggregate write/readback changed {name}")
        require({name: digest(ROOT / name) for name in CODE_FILES} == code_hashes, "Capture helpers changed during aggregation")
        require(all(digest(Path(path)) == expected for path, expected in frozen_inputs.items()), "Protocol, audit, worker report, or aggregate code changed during aggregation")
        temporary.replace(args.output)
        report.update(status="passed", stage="complete", analysis_npz_sha256=digest(args.output), analysis_npz_path=str(args.output.resolve()), analysis_npz_bytes=args.output.stat().st_size, rows=len(seen_rows), real_data=True, video_alignment_checked=True, physics_alignment_checked=True, all_layers_complete=True, match_splits_disjoint=True, checkpoint_integrity_checked=True, all_source_label_joins_exact=True, duplicate_or_missing_rows=False, row_order="sorted clip_id, then view index, then latent time", finished_at_utc=datetime.now(timezone.utc).isoformat())
        stage("complete")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        stage(report["stage"])
        raise


if __name__ == "__main__":
    main()
