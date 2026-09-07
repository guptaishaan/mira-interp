#!/usr/bin/env python3
"""Read-only independent bookkeeping/output audit before causal selection tests.

Recompute the frozen ranking from all discovery scores and the output objective
from saved original-model flow arrays. This does not independently reconstruct
the model's backward pass: all-layer discovery gradient tiles were not persisted.
No selection or confirmation arrays are accessed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira_interp.probes import load_selected_models


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/causal_development_v1")
    args = parser.parse_args()
    reg_path, lock_path = args.directory / "registration.json", args.directory / "discovery_selection.json"
    registration, lock = read(reg_path), read(lock_path)
    reg_hash = sha(reg_path)
    require(lock["status"] == "frozen_discovery_site_selection" and lock["registration_sha256"] == reg_hash,
            "Discovery stage has not completed for this registration")
    require(lock["n_matches"] == 31 and lock["n_pairs_seeds"] == 62, "Expected31discovery matches and2seeds")
    for name, digest in registration["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Registered causal implementation changed")
    for section in ("input_bindings", "probe_bindings"):
        for entry in registration[section].values():
            require(sha(entry["path"]) == entry["sha256"], "Registered source/probe/audit changed")
    selection = read(registration["probe_bindings"]["selection"]["path"])
    models = load_selected_models(registration["probe_bindings"]["archive"]["path"], selection)
    probe = next(model.main for model in models if model.name == registration["output_probe"])
    pair_by_id = {pair["match_id"]: pair for pair in registration["pairs"] if pair["split"] == "discovery"}
    require(len(pair_by_id) == 31, "Original discovery match count differs")
    expected = {(match, seed) for match in pair_by_id for seed in registration["paired_noise_seeds"]}
    require(len(lock["pairs"]) == len(expected) == 62, "Discovery pair count differs")
    require({(row["match_id"], row["seed"]) for row in lock["pairs"]} == expected, "Missing/duplicated discovery pair")
    scores, checked, frozen = [], [], {str(reg_path): reg_hash, str(lock_path): sha(lock_path)}
    for entry in lock["pairs"]:
        path = Path(entry["path"])
        require(sha(path) == entry["sha256"], "Discovery pair report changed")
        frozen[str(path)] = entry["sha256"]
        result = read(path)
        require(result["split"] == "discovery" and result["match_id"] in pair_by_id,
                "Reject selection/confirmation report before opening target arrays")
        pair = pair_by_id[result["match_id"]]
        require(result["seed"] == entry["seed"] and result["match_id"] == entry["match_id"]
                and result["registration_sha256"] == reg_hash and result["status"] == "passed", "Pair identity/status differs")
        for side in ("recipient", "donor"):
            require(result[f"{side}_clip_id"] == pair[side]["clip_id"]
                    and result[f"{side}_prepared_sha256"] == pair[side]["sha256"], "Pair/source identity differs")
            source_path = Path(pair[side]["artifact_path"])
            require(sha(source_path) == pair[side]["sha256"], "Registered source clip changed")
            frozen[str(source_path)] = pair[side]["sha256"]
            with np.load(source_path, allow_pickle=False) as source:
                require(float(source["targets"][0, 15, 2]) == result[f"{side}_source_ball_z"]
                        and float(source["timestamps"][0, 15]) == result[f"{side}_target_timestamp"],
                        "Objective source target/time does not match own-view final frame")
                indices = source["source_frame_indices"]
                frame = int(indices[15] if indices.ndim == 1 else indices[0, 15])
                require(frame == result[f"{side}_source_frame_index"], "Wrong original source frame")
                require(source["player_ids"].astype(str).tolist() == result["canonical_player_ids"], "Persistent entity order differs")
        require(result["gradient_forward_repeat_bitwise_equal"] is True and result["no_input_or_action_changes"] is True
                and result["conditions"] == [] and result["engineering_only_pilot"] is False, "Discovery controls/scope differ")
        rows = result["scores"]
        require(len(rows) == 17 and [row["site"] for row in rows] == registration["sites"]
                and [row["site_index"] for row in rows] == list(range(17)), "Missing or reordered attribution site")
        require(all(np.isfinite([row["attribution"], row["gradient_l2"], row["donor_delta_l2"]]).all()
                    and row["gradient_l2"] >= 0 and row["donor_delta_l2"] >= 0 for row in rows), "Invalid attribution values")
        scores.append([row["attribution"] for row in rows])
        artifact = Path(result["artifact_path"])
        require(sha(artifact) == result["artifact_sha256"], "Discovery tensor archive changed")
        frozen[str(artifact)] = result["artifact_sha256"]
        with np.load(artifact, allow_pickle=False) as arrays:
            flow, z_tau = arrays["baseline_flow"], arrays["recipient_z_t"]
            require(flow.shape == z_tau.shape == (1, 8, 36, 16, 32)
                    and flow.dtype == z_tau.dtype == np.float32 and np.isfinite(flow).all(), "Unexpected original output/input layout")
            require(hashlib.sha256(z_tau.tobytes()).hexdigest() == result["input_sha256"]["z_t"], "Saved recipient input differs from fixed-input hash")
            require(hashlib.sha256(flow.tobytes()).hexdigest() == result["baseline"]["prediction_sha256"], "Saved original flow differs from report")
            noise, clean = arrays["recipient_noise"], arrays["recipient_clean_codec"]
            require(hashlib.sha256(noise.tobytes()).hexdigest() == result["paired_noise_sha256"], "Saved paired noise differs")
            require(clean.shape == (4, 8, 9, 16, 32) and noise.shape == flow.shape, "Saved codec/noise layout differs")
            clean_joint = clean.transpose(1, 0, 2, 3, 4).reshape(1, 8, 36, 16, 32)
            require(np.array_equal(z_tau, .5*clean_joint+.5*noise), "Saved interpolant does not match registered tau/noise/clean input")
            donor_z = float(arrays["donor_source_targets_final"][0, 2])
            require(donor_z == result["donor_source_ball_z"]
                    and float(arrays["recipient_source_targets_final"][0, 2]) == result["recipient_source_ball_z"], "Own-view final source label differs")
            estimate = z_tau+.5*flow
            decoded = probe.predict(estimate[0, 7, :9].reshape(1, 4608))[0]
            objective = -float(((decoded[2]-donor_z)/probe.y_scale[2])**2)
            proxy_error = float(np.max(np.abs(decoded-np.asarray(result["baseline"]["output_position_proxy"]))))
            require(np.allclose(decoded, result["baseline"]["output_position_proxy"], rtol=1e-10, atol=1e-8), "Frozen endpoint proxy does not reproduce")
            require(np.isclose(objective, result["objective"], rtol=1e-10, atol=1e-10)
                    and np.isclose(objective, result["baseline"]["objective"], rtol=1e-10, atol=1e-10), "Donor-target objective does not reproduce")
        checked.append({"match_id": result["match_id"], "seed": result["seed"], "output_proxy_max_abs_difference": proxy_error,
                        "objective_abs_difference": abs(objective-result["objective"]), "all17_scores_finite": True})
    values = np.asarray(scores)
    means = values.mean(0)
    ranks = sorted(range(17), key=lambda site: (-means[site], site))
    require(ranks[:3] == lock["chosen_site_indices"] and [registration["sites"][i] for i in ranks[:3]] == lock["chosen_sites"], "Frozen top3 differs from all-discovery ranking")
    require(all(float(means[index]) == lock["all_site_mean_attributions"][site] for index, site in enumerate(registration["sites"])), "Stored mean attribution differs")
    require(all(sha(path) == digest for path, digest in frozen.items()), "Causal evidence changed during audit")
    report = {"status": "passed", "registration_sha256": reg_hash, "discovery_selection_sha256": sha(lock_path),
              "audit_script_sha256": sha(Path(__file__)), "n_matches": 31, "n_pairs_seeds": 62,
              "all17_attribution_means_recomputed": True, "output_objectives_recomputed": True,
              "selection_arrays_read": False, "confirmation_arrays_read": False, "gpu_used": False,
              "backward_pass_independently_recomputed": False,
              "scope_limit": "This audits complete pair/site coverage, stored score aggregation and saved original output objectives. Gradient dot activation differences are validated by the registered runtime and engineering tests, not rerun here.",
              "chosen_sites": lock["chosen_sites"], "ranked_sites": [
                  {"site": registration["sites"][i], "site_index": i, "mean_attribution": float(means[i]),
                   "positive_seed_pairs": int((values[:, i]>0).sum()), "negative_seed_pairs": int((values[:, i]<0).sum())} for i in ranks],
              "pair_checks": checked}
    out = args.directory / "discovery_audit.json"
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "n_matches", "n_pairs_seeds", "chosen_sites", "ranked_sites")}, indent=2))


if __name__ == "__main__":
    main()
