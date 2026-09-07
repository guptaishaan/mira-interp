#!/usr/bin/env python3
"""Register, pilot, then measure all frozen dictionaries' internal causal fidelity.

This replays the original 11 selection matches and paired seeds at block15.
No dictionary is selected, fitted, or renamed using these effects. A frozen
clean-codec readout on edited model endpoints is not validated video physics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mira/src"), str(ROOT / "scripts")]
import causal_development as causal
from mira_interp.causal_development import ResidualHooks, tensor_hash
from mira_interp.development_capture import channel_projection, forward
from mira_interp.model_loading import load_pretrained_four_player, sha256
from mira_interp.sparse_causal_fidelity import (load_dictionary, reconstruct_descriptor,
    reconstruct_tile, spatial_descriptor)
from analyze_development import require, write

SITE = 16
IDENTITY = "spatial/block_15_output"
CONDITIONS = ("recipient_reconstruction", "donor_reconstruction")
MEAN_CONTROL = "discovery_mean"
CODE = causal.CODE + ["scripts/sparse_causal_fidelity.py", "src/mira_interp/sparse_causal_fidelity.py",
                     "src/mira_interp/feature_geometry.py"]


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(Path(path))}


def current_code():
    return {name: sha256(ROOT / name) for name in CODE}


def checked(entry):
    require(sha256(Path(entry["path"])) == entry["sha256"], f"Bound file changed: {entry['path']}")
    return read(entry["path"])


def original_args(args):
    return argparse.Namespace(output_dir=args.causal_dir, probe_dir=args.probe_dir, probe_audit=args.probe_audit)


def feature_inputs(args):
    """Require the full 3-seed x 2-budget x 5-variant set and independent audit."""
    audit = read(args.feature_audit)
    require(audit["status"] == "passed" and audit["all_reconstruction_metrics_recomputed"] is True,
            "Independent sparse-training audit must pass before fidelity registration")
    reports = [binding(path) for path in sorted(args.feature_report)]
    require(len(reports) == 3 and len({row["path"] for row in reports}) == 3, "Exactly three distinct seed reports required")
    pairs = lambda rows: {(str(Path(row["path"]).resolve()), row["sha256"]) for row in rows}
    require(len(audit["feature_reports"]) == 3 and pairs(reports) == pairs(audit["feature_reports"]), "Sparse audit/report set differs")
    checkpoints, bundles, metadata = [], {}, []
    seen = set()
    for entry in reports:
        report = checked(entry)
        require(report["status"] == "passed_development_feature_analysis" and report["scope"] == "development_only"
                and report["descriptor_identity"] == IDENTITY and report["dimension"] == 1536
                and report["temporal_readout"] == "current" and report["confirmation_rows_loaded"] is False,
                "Feature report scope or descriptor differs")
        require(report["match_counts"] == {"discovery": 31, "selection": 11}, "Wrong dictionary cohort")
        require(report["input_npz_sha256"] == audit["input_npz_sha256"]
                and report["selected_probe_sha256"] == audit["selected_probe_sha256"], "Sparse input provenance differs")
        require(report["protocol_sha256"] == audit["protocol_sha256"], "Sparse training protocol differs")
        for path, expected in report["code_sha256"].items():
            require(sha256(ROOT / path) == expected, "Dictionary training source changed")
        require(len(report["outputs"]) == len(report["sparse"]) == 10, "Each seed must include all 10 frozen dictionaries")
        report_seeds = set()
        for filename, artifact in sorted(report["outputs"].items()):
            require(Path(filename).name == filename and filename.endswith(".pt"), "Unsafe or noncheckpoint feature output")
            path = Path(entry["path"]).parent / filename
            item = binding(path)
            require(item["sha256"] == artifact["sha256"] and path.stat().st_size == artifact["bytes"], "Dictionary checkpoint changed")
            bundle = load_dictionary(path)
            details = bundle[3]
            require(details in report["sparse"], "Checkpoint metadata is absent from feature report")
            kind, active, seed = details["kind"], details["active_scalar_budget"], details["seed"]
            temporal = details["temporal_weight"] > 0
            variant = kind + ("_temporal" if temporal else "")
            if temporal and details["temporal_control"] == "shuffled_within_match":
                variant += "_shuffled"
            require(seed in {0, 1, 2} and active in {32, 64}, "Unexpected training seed or active budget")
            require(variant in {"relu", "signed", "block", "block_temporal", "block_temporal_shuffled"}, "Unexpected dictionary variant")
            require(details["group_size"] == (8 if kind == "block" else 1), "Dictionary block size differs")
            key = (seed, active, variant)
            require(key not in seen, "Duplicate sparse variant")
            seen.add(key)
            report_seeds.add(seed)
            identifier = f"seed{seed}_{variant}_active{active}"
            bundles[identifier] = bundle
            checkpoints.append(item)
            metadata.append({"id": identifier, "seed": seed, "variant": variant, "active": active,
                             "checkpoint": item, "feature_report": entry, "training": details})
        require(len(report_seeds) == 1, "Each feature report must contain one seed")
    require(len(seen) == 30 and len(audit["checkpoints"]) == 30 and pairs(checkpoints) == pairs(audit["checkpoints"]),
            "Independent sparse audit must bind exactly all 30 checkpoints")
    require(len({bundle[3]["normalizer_sha256"] for bundle in bundles.values()}) == 1, "Dictionary normalizers differ")
    return {"audit": binding(args.feature_audit), "reports": reports, "checkpoints": checkpoints}, metadata, bundles


def prerequisite_bindings(args):
    original, reg_hash, models = causal.verify_registration(original_args(args))
    paths = {name: args.causal_dir / name for name in
             ("registration.json", "discovery_selection.json", "selection_results.json", "selection_audit.json", "pilot.json")}
    lock, result, audit = (read(paths[name]) for name in ("discovery_selection.json", "selection_results.json", "selection_audit.json"))
    require(lock["chosen_site_indices"][0] == SITE, "Block15 must remain the original top discovery site")
    require(result["status"] == "passed_development_internal_causal_tests" and result["n_matches"] == 11
            and result["n_pairs_seeds"] == 22 and result["registration_sha256"] == reg_hash, "Original exact causal stage incomplete")
    require(audit["status"] == "passed" and audit["selection_results_sha256"] == sha256(paths["selection_results.json"])
            and audit["registration_sha256"] == reg_hash and audit["conditions_checked"] == 660,
            "Independent causal selection audit differs")
    expected = {(pair["match_id"], seed) for pair in original["pairs"] if pair["split"] == "selection" for seed in causal.SEEDS}
    require(len(result["pairs"]) == 22 and {(row["match_id"], row["seed"]) for row in result["pairs"]} == expected,
            "Exact fixed 11-match paired-seed set required")
    for entry in result["pairs"]:
        row = checked(entry)
        require(row["status"] == "passed" and row["registration_sha256"] == reg_hash and row["split"] == "selection"
                and sha256(Path(row["artifact_path"])) == row["artifact_sha256"], "Original causal pair changed")
    return original, result, models, {name: binding(path) for name, path in paths.items()}


def register(args):
    original, result, _, causal_bindings = prerequisite_bindings(args)
    feature_bindings, metadata, _ = feature_inputs(args)
    record = {
        "status": "registered_before_sparse_causal_execution", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development-only internal model-output causal fidelity", "physical_control_established": False,
        "confirmation_used": False, "new_confirmation_required": True,
        "descriptor_identity": IDENTITY, "dimension": 1536, "temporal_readout": "current", "site_index": SITE,
        "support": {"view": 0, "latent": 7, "height": 9, "width": 16}, "conditions": list(CONDITIONS),
        "lift": "native + broadcast_3x4_tokens((reconstruction - descriptor(native)).reshape(3,4,128) @ Q.T); no 1/12 factor",
        "nullspace": "Native within-bin/channel information outside the descriptor is retained, up to floating point roundoff.",
        "fidelity_limit": "Conditional reconstruction fidelity only: retained native nullspace can preserve effects even if the dictionary discards descriptor information. This does not establish feature sufficiency.",
        "discovery_mean_control": "Two shared controls per pair replace recipient/donor descriptors by the identical frozen discovery mean, preserving each native nullspace. These reveal effects surviving without descriptor deviations from the mean.",
        "control_id": MEAN_CONTROL, "conditions_per_pair": 62, "full_study_forwards": 1364,
        "precision": "Original FP32 model, BF16 autocast, FP32 source preparation and tau; dictionary CPU FP32 using frozen FP64 global-RMS normalization.",
        "reference_conditions": {"recipient_reconstruction": "native recipient baseline", "donor_reconstruction": "original full donor tile replacement"},
        "objective": original["objective"], "output_probe": original["output_probe"],
        "output_proxy_limit": original["output_probe_distribution_note"],
        "pair_rule": "Reuse all original selection pairs and two seeds; no new donor/site/dictionary selection.",
        "pairs": result["pairs"], "n_matches": 11, "n_pairs_seeds": 22, "dictionary_count": 30,
        "dictionary_selection": "Every checkpoint is evaluated and reported; no causal-performance winner is selected.",
        "normalizer": "Frozen equal-match discovery mean and one global RMS, never refitted.",
        "dictionary_input_rounding": "Native FP32 descriptors are reconstructed; dictionary training descriptors were saved FP16. This small precision-distribution difference is recorded, not fitted away.",
        "metrics": "Raw standardized donor-error gain and difference from native reference; full-flow/endpoint distances, all15 position-proxy differences and descriptor reconstruction errors. No recovery fractions.",
        "uncertainty": "Average paired seeds within each of11matches;500 wholematch bootstrap draws seed20260907. Descriptive conditional intervals; no fresh confirmation or multiple-comparison claims.",
        "pilot": "Reserved original pilot only, all30dictionaries bothconditions; native baseline/full-donor/zero-lift controls must replay bitwise.",
        "feature_bindings": feature_bindings, "dictionaries": metadata, "causal_bindings": causal_bindings,
        "code_sha256": current_code(),
        "artifact_policy": "No raw source labels, actions or pixels copied to fidelity archives; original source references remain hash bound.",
    }
    write(args.output_dir / "registration.json", record, exclusive=True)
    print("Sparse causal protocol registered; no GPU execution occurred.", flush=True)


def verify(args, *, load=True):
    path = args.output_dir / "registration.json"
    registration = read(path)
    require(registration["code_sha256"] == current_code(), "Sparse fidelity implementation changed after registration")
    for entry in registration["causal_bindings"].values():
        checked(entry)
    features = registration["feature_bindings"]
    checked(features["audit"])
    for entry in features["reports"]:
        checked(entry)
    for entry in features["checkpoints"]:
        require(sha256(Path(entry["path"])) == entry["sha256"], "Registered dictionary changed")
    for entry in registration["pairs"]:
        row = checked(entry)
        require(sha256(Path(row["artifact_path"])) == row["artifact_sha256"], "Original causal archive changed")
    if not load:
        return registration
    original, result, models, causal_bindings = prerequisite_bindings(args)
    current_features, metadata, bundles = feature_inputs(args)
    require(causal_bindings == registration["causal_bindings"] and current_features == features
            and metadata == registration["dictionaries"] and result["pairs"] == registration["pairs"], "Registered inputs differ")
    return registration, sha256(path), original, models, bundles


def matched_summary(rows, identifier, condition):
    selected = [(row["match_id"], next(item for item in row["conditions"]
                 if item["dictionary_id"] == identifier and item["condition"] == condition)) for row in rows]
    matches = sorted({match for match, _ in selected})
    require(len(matches) == 11 and all(sum(match == mid for mid, _ in selected) == 2 for match in matches), "Incomplete fidelity pairs")
    draws = np.random.default_rng(20260907).integers(0, 11, (500, 11))
    result = {"dictionary_id": identifier, "condition": condition, "n_matches": 11, "n_paired_seeds": 2, "metrics": {}}
    for metric in ("standardized_error_gain", "gain_difference_from_native_reference", "flow_distance_from_native_reference_l2",
                   "flow_mse_from_native_reference", "proxy_distance_from_native_reference_standardized_rms",
                   "descriptor_reconstruction_rmse", "endpoint_recipient_source_mse"):
        values = np.asarray([np.mean([item[metric] for mid, item in selected if mid == match]) for match in matches])
        result["metrics"][metric] = {"mean": float(values.mean()), "match_bootstrap_ci95": np.quantile(values[draws].mean(1), [.025, .975]).tolist(),
                                     "match_values": dict(zip(matches, values.tolist()))}
    return result


def paired_mean_summary(rows, identifier, condition):
    """Compare each dictionary with the same pair's shared mean-descriptor edit."""
    matches = sorted({row["match_id"] for row in rows})
    require(len(matches) == 11, "Expected eleven independent matches")
    values = []
    for row in rows:
        indexed = {(item["dictionary_id"], item["condition"]): item for item in row["conditions"]}
        model, control = indexed[identifier, condition], indexed[MEAN_CONTROL, condition]
        values.append((row["match_id"], model["standardized_error_gain"]-control["standardized_error_gain"],
                       abs(control["gain_difference_from_native_reference"])-abs(model["gain_difference_from_native_reference"])))
    draws = np.random.default_rng(20260907).integers(0, 11, (500, 11))
    result = {"dictionary_id": identifier, "condition": condition, "control_id": MEAN_CONTROL, "metrics": {}}
    for index, name in ((1, "donor_objective_gain_above_mean_control"), (2, "absolute_native_gain_error_improvement_over_mean_control")):
        per_match = np.asarray([np.mean([value[index] for value in values if value[0] == match]) for match in matches])
        result["metrics"][name] = {"mean": float(per_match.mean()),
            "match_bootstrap_ci95": np.quantile(per_match[draws].mean(1), [.025, .975]).tolist(),
            "match_values": dict(zip(matches, per_match.tolist()))}
    return result


def run_pair(model, pair, seed, old, models, bundles, projection):
    arrays = causal.input_arrays(pair["recipient"])
    inputs, clean, noise = causal.prepare(model, arrays, seed)
    require(causal.input_hashes(inputs) == old["input_sha256"] and tensor_hash(noise) == old["paired_noise_sha256"], "Original fixed inputs/noise did not replay")
    require(sha256(Path(old["artifact_path"])) == old["artifact_sha256"], "Original causal tensors changed")
    names = ["baseline_flow", f"site{SITE}_exact_donor_flow", f"site{SITE}_recipient_tile", f"site{SITE}_donor_tile"]
    with np.load(old["artifact_path"], allow_pickle=False) as archive:
        native = {name: torch.from_numpy(archive[name]).to("cuda") for name in names}
    baseline, donor_flow = native["baseline_flow"], native[f"site{SITE}_exact_donor_flow"]
    recipient_tile, donor_tile = native[f"site{SITE}_recipient_tile"], native[f"site{SITE}_donor_tile"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with ResidualHooks(model.world_model.transformer, capture=True) as trace:
            replay = forward(model, inputs)
        require(torch.equal(replay.float(), baseline) and torch.equal(trace.values[SITE], recipient_tile), "Native model baseline/tile differs")
        with ResidualHooks(model.world_model.transformer, site=SITE, replacement=donor_tile):
            donor_replay = forward(model, inputs)
        require(torch.equal(donor_replay.float(), donor_flow), "Native full-donor model output differs")
        identity_tile = reconstruct_tile(recipient_tile, spatial_descriptor(recipient_tile, projection), projection)
        require(torch.equal(identity_tile, recipient_tile), "Descriptor identity lift is not exact")
        with ResidualHooks(model.world_model.transformer, site=SITE, replacement=identity_tile):
            identity = forward(model, inputs)
        require(torch.equal(identity.float(), baseline), "Zero-lift model output differs")
    probe, donor_z = models[causal.OUTPUT_PROBE], old["donor_source_ball_z"]
    descriptors = torch.cat([spatial_descriptor(value, projection) for value in (recipient_tile, donor_tile)])
    result = {"match_id": pair["match_id"], "split": pair["split"], "seed": seed,
              "recipient_clip_id": pair["recipient"]["clip_id"], "donor_clip_id": pair["donor"]["clip_id"],
              "input_sha256": causal.input_hashes(inputs), "paired_noise_sha256": tensor_hash(noise),
              "native_baseline_replay_bitwise_equal": True, "native_donor_replay_bitwise_equal": True,
              "zero_lift_replay_bitwise_equal": True, "conditions": []}
    saved = {"baseline_flow": baseline.cpu().numpy(), "native_donor_flow": donor_flow.cpu().numpy(),
             "recipient_tile": recipient_tile.cpu().numpy(), "donor_tile": donor_tile.cpu().numpy(),
             "recipient_z_t": inputs["z_t"].float().cpu().numpy(), "recipient_clean_codec": clean.float().cpu().numpy(),
             "native_descriptors": descriptors.cpu().numpy()}
    with torch.no_grad():
        native_metrics = [causal.metric(inputs, flow, baseline, probe, donor_z, clean) for flow in (baseline, donor_flow)]
        result["native_reference_metrics"] = dict(zip(CONDITIONS, native_metrics))
        for identifier in [MEAN_CONTROL, *bundles]:
            if identifier == MEAN_CONTROL:
                reconstructions = next(iter(bundles.values()))[1].float().reshape(1, -1).repeat(2, 1)
                codes = torch.empty(2, 0)
            else:
                reconstructions, codes = reconstruct_descriptor(bundles[identifier], descriptors)
            saved[f"{identifier}_reconstructed_descriptors"] = reconstructions.numpy()
            saved[f"{identifier}_codes"] = codes.numpy()
            for index, condition in enumerate(CONDITIONS):
                original_tile = (recipient_tile, donor_tile)[index]
                target = reconstructions[index:index+1].to("cuda")
                replacement = reconstruct_tile(original_tile, target, projection)
                roundtrip = spatial_descriptor(replacement, projection)
                delta = target - descriptors[index:index+1]
                residual = roundtrip - target
                require(float(torch.linalg.vector_norm(residual)) <= 1e-4 * (1 + float(torch.linalg.vector_norm(target))), "Spatial lift roundtrip exceeds FP32 allowance")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with ResidualHooks(model.world_model.transformer, site=SITE, replacement=replacement):
                        prediction = forward(model, inputs)
                reference = (baseline, donor_flow)[index]
                require(torch.equal(prediction[:, :7].float(), baseline[:, :7]), "Earlier flow frames changed")
                measured = causal.metric(inputs, prediction, baseline, probe, donor_z, clean)
                flow_difference = prediction.float() - reference
                reference_metric = native_metrics[index]
                proxy_gap = (np.asarray(measured["output_position_proxy"]) - reference_metric["output_position_proxy"]) / probe.y_scale
                measured.update(dictionary_id=identifier, condition=condition, is_discovery_mean_control=identifier == MEAN_CONTROL,
                    standardized_error_gain=measured["objective"] - native_metrics[0]["objective"],
                    gain_difference_from_native_reference=measured["objective"] - reference_metric["objective"],
                    flow_distance_from_native_reference_l2=float(torch.linalg.vector_norm(flow_difference)),
                    flow_mse_from_native_reference=float(flow_difference.square().mean()),
                    endpoint_distance_from_native_reference_l2=float(torch.linalg.vector_norm(flow_difference * (1-inputs["tau"]))),
                    proxy_distance_from_native_reference_standardized_rms=float(np.sqrt(np.square(proxy_gap).mean())),
                    proxy_difference_from_native_reference_standardized=proxy_gap.tolist(),
                    descriptor_reconstruction_rmse=float(delta.square().mean().sqrt()),
                    descriptor_roundtrip_max_abs_error=float(residual.abs().max()),
                    lifted_delta_l2=float(torch.linalg.vector_norm(replacement-original_tile)),
                    all_earlier_flow_frames_bitwise_unchanged=True)
                result["conditions"].append(measured)
                saved[f"{identifier}_{condition}_flow"] = prediction.float().cpu().numpy()
    require(causal.input_hashes(inputs) == old["input_sha256"], "Recipient inputs changed during fidelity evaluation")
    return result, saved


def execute(args):
    registration, reg_hash, original, models, bundles = verify(args)
    complete_path = args.output_dir / ("pilot.json" if args.phase == "pilot" else "fidelity_results.json")
    require(not complete_path.exists(), "Completed fidelity stages are immutable")
    if args.phase == "evaluate":
        pilot = read(args.output_dir / "pilot.json")
        require(pilot["status"] == "passed" and pilot["registration_sha256"] == reg_hash, "Matching reserved sparse-fidelity pilot must pass")
        checked(pilot["pair"])
        pilot_audit = read(args.output_dir / "pilot_audit.json")
        require(pilot_audit["status"] == "passed" and pilot_audit["pilot_sha256"] == sha256(args.output_dir / "pilot.json")
                and pilot_audit["registration_sha256"] == reg_hash, "Independent reserved fidelity pilot audit must pass")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose only the single authorized GPU")
    model, loading = load_pretrained_four_player(args.assets)
    model = model.to("cuda")
    require(all(parameter.dtype == torch.float32 for parameter in model.parameters()), "Original FP32 model parameters required")
    projection = channel_projection("cuda")
    require(float((projection.T @ projection - torch.eye(128, device="cuda")).abs().max()) < 2e-6, "Channel projection lost orthonormality")
    if args.phase == "pilot":
        record = original["pilot_record"]
        pair = {"match_id": record["match_id"], "split": "pilot", "recipient": record, "donor": record}
        source = read(args.causal_dir / "pilot.json")["pair"]
        jobs = [(pair, causal.SEEDS[0], source)]
    else:
        pairs = {pair["match_id"]: pair for pair in original["pairs"] if pair["split"] == "selection"}
        jobs = [(pairs[entry["match_id"]], entry["seed"], entry) for entry in registration["pairs"]]
    started, rows, entries = time.monotonic(), [], []
    for pair, seed, source in jobs:
        path = args.output_dir / pair["split"] / f"{pair['match_id']}_seed{seed}.json"
        old = checked(source)
        if path.exists():
            row = read(path)
            require(row["status"] == "passed" and row["registration_sha256"] == reg_hash and row["source_causal_pair"] == source
                    and row["match_id"] == pair["match_id"] and row["seed"] == seed and row["split"] == pair["split"], "Stale cached fidelity pair")
            require(sha256(Path(row["artifact_path"])) == row["artifact_sha256"], "Cached fidelity tensors changed")
        else:
            row, arrays = run_pair(model, pair, seed, old, models, bundles, projection)
            artifact = args.artifact_dir / pair["split"] / path.with_suffix(".npz").name
            artifact.parent.mkdir(parents=True, exist_ok=True)
            require(not artifact.exists(), "Uncommitted fidelity tensor archive exists; preserve/audit before resume")
            with artifact.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            row.update(status="passed", registration_sha256=reg_hash, source_causal_pair=source,
                       artifact_path=str(artifact.resolve()), artifact_sha256=sha256(artifact),
                       raw_source_labels_included=False, physical_control_established=False, confirmation_used=False)
            write(path, row, exclusive=True)
        require(len(row["conditions"]) == 62 and {(entry["dictionary_id"], entry["condition"]) for entry in row["conditions"]}
                == {(identifier, condition) for identifier in [MEAN_CONTROL, *bundles] for condition in CONDITIONS}, "Incomplete dictionary/control conditions")
        rows.append(row)
        entries.append(binding(path))
        print(json.dumps({"phase": args.phase, "completed_pairs": len(rows), "expected_pairs": len(jobs),
                          "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
    verify(args, load=False)
    for entry in entries:
        row = checked(entry)
        require(sha256(Path(row["artifact_path"])) == row["artifact_sha256"], "Fidelity output changed during run")
    result = {"status": "passed" if args.phase == "pilot" else "passed_development_sparse_causal_fidelity",
              "registration_sha256": reg_hash, "scope": registration["scope"], "physical_control_established": False,
              "confirmation_used": False, "new_confirmation_required": True, "model_loading": loading,
              "n_pairs_seeds": len(rows), "dictionary_count": len(bundles), "conditions_checked": sum(len(row["conditions"]) for row in rows),
              "shared_mean_control_count_per_pair": 2, "conditional_reconstruction_fidelity_only": True,
              "elapsed_seconds": time.monotonic()-started, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "output_proxy_limit": registration["output_proxy_limit"], "uncertainty": registration["uncertainty"]}
    if args.phase == "pilot":
        result.update(pair=entries[0], engineering_only=True)
    else:
        result.update(pairs=entries, n_matches=11, comparisons=[matched_summary(rows, identifier, condition)
                      for identifier in [MEAN_CONTROL, *bundles] for condition in CONDITIONS],
                      paired_discovery_mean_comparisons=[paired_mean_summary(rows, identifier, condition)
                      for identifier in bundles for condition in CONDITIONS])
    write(complete_path, result, exclusive=True)
    print(f"Completed sparse fidelity {args.phase}; internal developmental scope only.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("register", "pilot", "evaluate"))
    parser.add_argument("--feature-report", type=Path, nargs="+", required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, default=ROOT / "results/causal_development_v1")
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "results/development_probes_v3")
    parser.add_argument("--probe-audit", type=Path, default=ROOT / "results/development_probes_v3/probe_audit.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sparse_causal_fidelity_v1")
    parser.add_argument("--artifact-dir", type=Path, default=Path("/data2/ishaangp/mira-interp/sparse_causal_fidelity_v1"))
    parser.add_argument("--assets", type=Path, default=Path("/data2/ishaangp/mira-interp/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "register":
        register(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
