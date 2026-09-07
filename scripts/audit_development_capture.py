#!/usr/bin/env python3
"""Independent CPU audit of development descriptors, provenance and source joins.

No fitting and no confirmation NPZ reads. A pilot-only report cannot satisfy the
full development gate. The audit never changes a capture or preparation file.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "607aef226c2cb241e66d00c9a3b7f323802c6bce869f96345c3a0b014a220755"
OLD_HELPER_HASH = "70603eb01a7c8996bf26f4b69193044a417377820244edbd173cb9f079f61e70"
ROLES = {"discovery", "selection"}
SITES = ["block_0_input"] + [f"block_{i}_output" for i in range(16)]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aggregate_captures import digest, read_json, require, verify_loading, write_json


def descriptor_checks():
    """Check every spatial bin independently with an explicit slice implementation."""
    import torch
    from mira_interp.development_capture import channel_projection, descriptors
    torch.set_num_threads(4)
    projection = channel_projection("cpu")
    q = projection.numpy()
    orthogonal_error = float(np.abs(q.T @ q - np.eye(128)).max())
    require(q.shape == (2048, 128) and orthogonal_error < 1e-5, "Channel map is not orthonormal")
    # Pure mathematical fixture; never a research observation or fitting input.
    rng = np.random.default_rng(79123)
    raw = torch.from_numpy(rng.standard_normal((1, 2, 36, 16, 2048)).astype(np.float32))
    means, spatial = descriptors(raw, projection)
    reference_means, reference_bins = [], []
    for view in range(4):
        for timestep in range(2):
            tile = raw[0, timestep, view * 9:(view + 1) * 9]
            reference_means.append(tile.mean((0, 1)))
            reference_bins.append(torch.cat([tile[by * 3:(by + 1) * 3, bx * 4:(bx + 1) * 4].mean((0, 1)) @ projection for by in range(3) for bx in range(4)]))
    ref_m, ref_s = torch.stack(reference_means), torch.stack(reference_bins)
    mean_error = float((means - ref_m).abs().max())
    bin_error = float((spatial - ref_s).abs().max())
    require(torch.allclose(means, ref_m, atol=1e-6, rtol=1e-5), "Descriptor mean/view/time order differs from explicit slices")
    require(torch.allclose(spatial, ref_s, atol=2e-6, rtol=1e-5), "Descriptor bin/channel order differs from explicit slices")
    return q, {"status": "passed", "fixture_is_research_data": False, "view_time_order": "view then latent time", "bin_order": "row bin then column bin then projected channel", "mean_max_abs_difference": mean_error, "spatial_max_abs_difference": bin_error, "orthogonality_max_abs_difference": orthogonal_error, "projection_seed": 20260907, "projection_sha256": hashlib.sha256(q.tobytes()).hexdigest(), "projection_hash_scope": "independent reconstruction; workers did not serialize the matrix"}


def check_helper_history(preparation):
    """Allow the recorded RGB-input tolerance repair only outside preparation code."""
    current = ROOT / "src/mira_interp/development_capture.py"
    require(preparation["code_sha256"] == digest(ROOT / "scripts/prepare_development.py"), "Preparation script changed")
    if preparation["helper_sha256"] == digest(current):
        return {"status": "identical"}
    old = ROOT / "results/development_attempts/helper_before_rgb_tolerance.py.txt"
    require(preparation["helper_sha256"] == OLD_HELPER_HASH == digest(old), "Unknown preparation helper change")
    trees = [ast.parse(path.read_text()) for path in [old, current]]
    functions = [{node.name: ast.dump(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))} for tree in trees]
    changed = sorted(key for key in set(functions[0]) | set(functions[1]) if functions[0].get(key) != functions[1].get(key))
    require(changed == ["make_inputs"], "Helper changes affect preparation or descriptor code")
    for tree in trees:
        tree.body = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "make_inputs"]
    require(ast.dump(trees[0]) == ast.dump(trees[1]), "Non-preparation function exclusion does not explain helper change")
    return {"status": "reviewed_nonpreparation_change", "old_path": str(old), "old_sha256": OLD_HELPER_HASH, "current_sha256": digest(current), "changed_functions": changed, "preparation_used_code_and_globals_identical": True}


def audit_record(side, prepared, original, protocol_hash, code_hashes, projection, *, rgb_check=False):
    import torch
    role, clip_id = original["role"], original["clip_id"]
    require(role in ROLES | {"pilot"}, "Confirmation input is forbidden before opening any NPZ")
    require(side["clip_id"] == prepared["clip_id"] == clip_id and side["match_id"] == prepared["match_id"] == original["match_id"], "Clip/match identity mismatch")
    require(side["split"] == prepared["split"] == role, "Role mismatch")
    require(side["registration_sha256"] == protocol_hash and side["code_sha256"] == code_hashes, "Capture code/protocol mismatch")
    require(side.get("parameter_dtype") == side.get("z_t_dtype") == "torch.float32" and side.get("no_labels_passed_to_model") is True, "Incorrect numerical or label-exclusion path")
    capture_path, input_path, original_path = Path(side["path"]), Path(prepared["artifact_path"]), Path(original["artifact_path"])
    require(digest(capture_path) == side["sha256"], "Capture NPZ hash mismatch")
    require(digest(input_path) == prepared["sha256"] == side["input_sha256"], "Float input NPZ hash mismatch")
    require(digest(original_path) == original["sha256"] == prepared["parent_sha256"], "Original input NPZ hash mismatch")
    require(prepared.get("source_component_sha256") == original["source_component_sha256"], "Source video/action/physics component hashes differ")
    require(all(prepared.get(key) is True for key in ["source_pts_exact", "round_to_previous_pixels_exact", "row_metadata_unchanged"]), "Preparation did not pass pixel/PTS/row controls")
    with np.load(capture_path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    shapes = {"mean_X": (32, 17, 2048), "spatial_X": (32, 17, 1536), "codec_spatial_X": (32, 4608), "codec_mean_X": (32, 32), "RGB_X": (32, 1536), "y": (32, 30)}
    for key, shape in shapes.items():
        require(saved[key].shape == shape and np.isfinite(saved[key]).all(), f"Invalid {key} array")
        require(saved[key].dtype == (np.float64 if key == "y" else np.float16), f"Unexpected {key} storage dtype")
    require(saved["sites"].tolist() == SITES, "Incorrect residual site columns")
    require(saved["match_ids"].tolist() == [original["match_id"]] * 32 and saved["clip_ids"].tolist() == [clip_id] * 32 and saved["split"].tolist() == [role] * 32, "Per-row identity mismatch")
    require(np.array_equal(saved["view_index"], np.repeat(np.arange(4), 8)) and np.array_equal(saved["latent_frame_index"], np.tile(np.arange(8), 4)), "View/time ordering differs")
    require(saved["canonical_player_ids"].shape == (32, 4) and np.array_equal(saved["canonical_player_ids"], np.tile(original["player_ids"], (32, 1))), "Canonical player IDs differ")
    with np.load(original_path, allow_pickle=False) as source, np.load(input_path, allow_pickle=False) as prepared_npz:
        for key in ["actions", "targets", "timestamps", "source_frame_indices", "video_pts", "player_ids", "target_names"]:
            require(np.array_equal(source[key], prepared_npz[key]), f"Float preparation changed {key}")
        require(source["targets"].shape == (4, 16, 30), "Original labels are not per-view")
        require(np.array_equal(saved["y"], source["targets"][:, 1::2].reshape(32, 30)), "Labels are not exact own-view pair-final targets")
        require(saved["target_names"].tolist() == source["target_names"].tolist(), "Target columns differ")
        indices = source["source_frame_indices"]
        if indices.shape == (16,):
            indices = np.broadcast_to(indices, (4, 16))
        require(indices.shape == (4, 16) and np.array_equal(saved["source_frame_index"], indices[:, 1::2].reshape(32)), "Wrong source frame join")
        require(np.array_equal(saved["timestamps"], source["timestamps"][:, 1::2].reshape(32)) and np.isfinite(saved["timestamps"]).all(), "Wrong own-view timestamp join")
        rgb_error = None
        if rgb_check:
            frames = prepared_npz["frames"]
            require(frames.shape == (4, 16, 3, 288, 512) and frames.dtype == np.float32 and np.isfinite(frames).all(), "Invalid float RGB")
            require(np.array_equal(np.rint(frames).clip(0, 255).astype(np.uint8), source["frames"]), "Float RGB does not reproduce previous uint8 frames")
            rgb = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(frames[:, 1::2].copy()).reshape(32, 3, 288, 512) / 255, (16, 32)).flatten(1).half().numpy()
            require(np.array_equal(rgb, saved["RGB_X"]), "RGB baseline differs from exact source frames")
            rgb_error = 0.0
    codec_reference = saved["codec_spatial_X"].astype(np.float32).reshape(32, 9, 16, 32).mean((1, 2))
    require(np.allclose(codec_reference, saved["codec_mean_X"].astype(np.float32), rtol=1e-3, atol=2e-5), "Codec spatial/mean baselines are inconsistent")
    # Equal-area bins imply average projected bins = projected global mean.
    # Both saved arrays are FP16, so compare with an explicit quantization allowance.
    saved_mean = saved["mean_X"].astype(np.float64)
    saved_bins = saved["spatial_X"].astype(np.float64).reshape(32, 17, 12, 128)
    q = projection.astype(np.float64)
    global_projection, bin_average = saved_mean @ q, saved_bins.mean(2)
    # For FP16 round-to-nearest, |saved-true| <= u/(1-u)*|saved|+eta,
    # where u=2^-11 and eta is half the minimum subnormal. Propagate each
    # saved mean's error through |Q| and each bin's error through its average.
    # A relative tolerance on the final sum is invalid near cancellation.
    u, eta = 2.**-11, 2.**-25
    rounding_budget = (u/(1-u)*np.abs(saved_mean)+eta) @ np.abs(q) + (u/(1-u)*np.abs(saved_bins)+eta).mean(2)
    spatial_difference = np.abs(global_projection - bin_average)
    require(np.all(spatial_difference <= rounding_budget), "Spatial/mean discrepancy exceeds the propagated FP16 rounding budget")
    require(all(np.any(saved["mean_X"][:, site] != 0) and np.any(saved["spatial_X"][:, site] != 0) for site in range(17)), "Missing or zero-filled site")
    identities = [(original["match_id"], int(v), int(t)) for v, t in zip(saved["view_index"], saved["source_frame_index"])]
    return {"clip_id": clip_id, "match_id": original["match_id"], "split": role, "capture_sha256": side["sha256"], "prepared_sha256": prepared["sha256"], "original_sha256": original["sha256"], "rows": 32, "exact_label_timestamp_identity_join": True, "codec_mean_max_abs_difference": float(np.abs(codec_reference - saved["codec_mean_X"]).max()), "spatial_global_consistency_max_abs_difference": float(spatial_difference.max()), "spatial_global_max_rounding_budget_fraction": float(np.max(spatial_difference / rounding_budget)), "RGB_source_recomputed": rgb_check, "RGB_max_abs_difference": rgb_error}, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/development_capture_manifest.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results/development_capture_audit.json")
    args = parser.parse_args()
    started = time.monotonic()
    report = {"status": "running", "scope": "reserved descriptor pilot" if args.pilot_only else "development only independent capture audit", "pilot_only": args.pilot_only, "gpu_used": False, "confirmation_npz_read": False, "clips": []}
    def stage(name):
        report.update(stage=name, elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(args.report, report)
        print(f"[{report['elapsed_seconds']:.1f}s] {name}: {len(report['clips'])} clips", flush=True)
    stage("prerequisites")
    try:
        protocol_path, split_path = ROOT / "configs/development_v3.json", ROOT / "data/qualified_split_manifest.json"
        require(digest(protocol_path) == PROTOCOL_HASH, "Unregistered development protocol")
        split = read_json(split_path)
        source_manifest_path = ROOT / ("data/qualified_pilot_clip_manifest.json" if args.pilot_only else "data/qualified_clip_manifest.json")
        source_manifest = read_json(source_manifest_path)
        require(source_manifest["status"] == "passed" and source_manifest["split_manifest_sha256"] == digest(split_path), "Original qualified data/split gate failed")
        allowed = {"pilot"} if args.pilot_only else ROLES
        originals = [row for row in source_manifest["records"] if row["role"] in allowed]
        if args.pilot_only:
            originals = originals[:1]
        by_id = {row["clip_id"]: row for row in originals}
        require(len(by_id) == len(originals), "Duplicate original clip IDs")
        assignments = {row["match_id"]: row for row in split["matches"]}
        for row in originals:
            require(assignments[row["match_id"]]["split"] == row["role"] and assignments[row["match_id"]]["player_ids"] == row["player_ids"], "Original role/player assignment differs")
        counts = Counter(row["role"] for row in {row["match_id"]: row for row in originals}.values())
        require(dict(counts) == ({"pilot": 1} if args.pilot_only else {"discovery": 31, "selection": 11}), "Incorrect exact development match cohort")
        require(len(originals) == (1 if args.pilot_only else 336), "Incorrect exact clip cohort")
        prep_path = ROOT / ("results/development_prepare_pilot.json" if args.pilot_only else "results/development_prepare.json")
        preparation = read_json(prep_path)
        require(preparation["status"] == "passed" and not preparation["errors"] and preparation["input_manifest_sha256"] == digest(source_manifest_path), "Float preparation gate differs")
        prepared = {row["clip_id"]: row for row in preparation["records"]}
        require(len(prepared) == len(preparation["records"]) == len(originals) and set(prepared) == set(by_id), "Float preparation is incomplete or includes other clips")
        report["preparation_helper_review"] = check_helper_history(preparation)
        pilot_path = ROOT / "results/development_capture_pilot.json"
        pilot = read_json(pilot_path)
        require(pilot["status"] == "passed" and pilot["registration_sha256"] == PROTOCOL_HASH, "Corrected numerical pilot did not pass")
        code_hashes = pilot["code_sha256"]
        require(all(digest(ROOT / name) == expected for name, expected in code_hashes.items()), "Capture/helper source changed since pilot")
        verify_loading(pilot["model_loading"])
        require(len(pilot["completed"]) == 1 and all(pilot["completed"][0]["controls"].get(key) is True for key in ["noop_bitwise_equal", "repeat_bitwise_equal"]), "Pilot controls failed")
        require(pilot["model_loading"]["adapter_sha256"] == digest(ROOT / "src/mira_interp/pretrained.py"), "Adapter changed since pilot")
        frozen_paths = [protocol_path, split_path, source_manifest_path, prep_path, pilot_path, Path(__file__)]
        if args.pilot_only:
            records = pilot["completed"]
            manifest_hash = digest(pilot_path)
            require(pilot["input_manifest_sha256"] == digest(prep_path), "Pilot prepared input manifest differs")
        else:
            combined = read_json(args.manifest)
            require(combined["status"] == "passed" and combined["registration_sha256"] == PROTOCOL_HASH and combined["code_sha256"] == code_hashes, "Combined capture gate differs")
            require(combined["confirmation_used"] is False and combined["rows"] == 10752 and combined["counts"] == {"discovery": 31, "selection": 11}, "Combined scope/counts differ")
            records = combined["records"]
            manifest_hash = digest(args.manifest)
            frozen_paths.append(args.manifest)
            completed = []
            for worker in range(2):
                path = ROOT / f"results/development_capture_worker{worker}.json"
                value = read_json(path)
                require(value["status"] == "passed" and not value["errors"] and value["registration_sha256"] == PROTOCOL_HASH and value["code_sha256"] == code_hashes, "Worker did not pass matching protocol/code")
                require(value["input_manifest_sha256"] == digest(prep_path), "Worker input preparation differs")
                verify_loading(value["model_loading"])
                require(value["model_loading"] == pilot["model_loading"], "Worker/pilot checkpoint loading differs")
                expected_ids = [row["clip_id"] for row in preparation["records"][worker::2]]
                require([row["clip_id"] for row in value["completed"]] == expected_ids and value["expected"] == len(expected_ids), "Worker modulo coverage differs")
                completed.extend(value["completed"])
                frozen_paths.append(path)
            require({row["clip_id"]: row for row in completed} == {row["clip_id"]: row for row in records}, "Combined manifest differs from worker completion records")
        require(len(records) == len(by_id) and len({row["clip_id"] for row in records}) == len(by_id) and {row["clip_id"] for row in records} == set(by_id), "Missing/duplicate/unregistered captured clips")
        projection, math_audit = descriptor_checks()
        report.update(descriptor_math=math_audit, registration_sha256=PROTOCOL_HASH, capture_manifest_sha256=manifest_hash, split_manifest_sha256=digest(split_path), source_manifest_sha256=digest(source_manifest_path), preparation_sha256=digest(prep_path), pilot_report_sha256=digest(pilot_path), code_sha256=code_hashes, audit_script_sha256=digest(Path(__file__)), match_counts=dict(counts))
        initial_hashes = {str(path): digest(path) for path in frozen_paths}
        seen = set()
        for index, side in enumerate(records):
            original = by_id[side["clip_id"]]
            row, identities = audit_record(side, prepared[side["clip_id"]], original, PROTOCOL_HASH, code_hashes, projection, rgb_check=args.pilot_only or index == 0)
            require(not seen.intersection(identities) and len(set(identities)) == 32, "Duplicate physical source view/frame rows")
            seen.update(identities)
            report["clips"].append(row)
            if index % 25 == 0:
                stage("audit_source_and_descriptor_arrays")
        require(all(digest(Path(path)) == expected for path, expected in initial_hashes.items()), "Audit inputs/code changed during verification")
        report.update(status="passed", rows=len(seen), all_source_label_joins_exact=True, all_layers_complete=True, all_feature_arrays_finite=True, exact_clip_cohort=True, match_splits_disjoint=True, checkpoint_integrity_checked=True, confirmation_used=False)
        stage("complete")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        stage(report.get("stage", "failed"))
        raise


if __name__ == "__main__":
    main()
