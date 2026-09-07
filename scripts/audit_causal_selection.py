#!/usr/bin/env python3
"""Independently audit exact causal outputs, paired controls and match summaries.

No model forward, gradient recomputation, fitting, site selection or confirmation
data access. The saved selected-site gradients permit an independent AtP dot
product check; metrics are recomputed from original model flow tensors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira_interp.probes import load_selected_models

CONDITIONS = ["self", "exact_donor", "component_transfer", "ablate_to_discovery_mean", "restore",
              "norm_matched_random", "norm_matched_random_full", "norm_matched_random_ablation",
              "wrong_variable_ball_x", "wrong_view_component"]
CONTROL_PAIRS = {"norm_matched_random": "component_transfer", "norm_matched_random_full": "exact_donor",
                 "norm_matched_random_ablation": "ablate_to_discovery_mean", "wrong_variable_ball_x": "component_transfer",
                 "wrong_view_component": "component_transfer"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, *, atol=1e-8, rtol=1e-8):
    require(np.shape(actual) == np.shape(expected) and np.allclose(actual, expected, atol=atol, rtol=rtol), name)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tile(array, view=0):
    return array[0, 7, view*9:(view+1)*9]


def norm(array):
    return float(np.linalg.norm(np.asarray(array, dtype=np.float64)))


def probe_direction(probe, projection, target):
    raw = probe.coefficient[:, target]*probe.y_scale[target]/probe.x_scale
    bins = raw.reshape(3, 4, 128) @ projection.astype(np.float64).T / 12
    return np.repeat(np.repeat(bins, 3, axis=0), 4, axis=1).astype(np.float32)


def bin_features(raw, projection):
    # Explicit independent slices; FP64 accumulation permits a small, declared
    # raw-unit allowance when comparing to the execution's FP32 descriptors.
    return np.concatenate([raw[y*3:(y+1)*3, x*4:(x+1)*4].astype(np.float64).mean((0, 1)) @ projection
                           for y in range(3) for x in range(4)])[None]


def correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def ranks(values):
    values = np.asarray(values)
    return np.asarray([sum(values < x)+(sum(values == x)-1)/2 for x in values], dtype=float)


def match_summary(rows, site, condition):
    subset = [row for row in rows if row["site_index"] == site and row["condition"] == condition]
    matches = sorted({row["match_id"] for row in subset})
    require(len(matches) == 11 and all(sum(row["match_id"] == match for row in subset) == 2 for match in matches), "Incomplete paired condition")
    gain = np.asarray([np.mean([row["gain"] for row in subset if row["match_id"] == match]) for match in matches])
    draws = np.random.default_rng(20260907).integers(0, len(matches), (500, len(matches)))
    shifts = np.asarray([row["proxy_shift_standardized"] for row in subset])
    side = np.ones(15, dtype=bool)
    side[2] = False
    return {"site_index": site, "condition": condition, "n_matches": 11,
            "mean_standardized_error_gain": float(gain.mean()), "match_bootstrap_ci95": np.quantile(gain[draws].mean(1), [.025, .975]).tolist(),
            "positive_gain_matches": int((gain>0).sum()), "match_gains": dict(zip(matches, gain.tolist())),
            "mean_flow_delta_l2": float(np.mean([row["flow_delta_l2"] for row in subset])),
            "mean_endpoint_source_mse_change": float(np.mean([row["endpoint_source_mse_change"] for row in subset])),
            "proxy_shift_standardized_mean_abs": np.abs(shifts).mean(0).tolist(),
            "proxy_shift_standardized_rms": np.sqrt(np.square(shifts).mean(0)).tolist(),
            "nontarget_position_shift_standardized_rms": float(np.sqrt(np.square(shifts[:, side]).mean())),
            "all_past_flows_bitwise_unchanged": all(row["past_flow_unchanged"] for row in subset)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/causal_development_v1")
    args = parser.parse_args()
    out = args.directory / "selection_audit.json"
    require(not out.exists(), "Prior audit is immutable")
    reg_path, lock_path, result_path = [args.directory/name for name in ("registration.json", "discovery_selection.json", "selection_results.json")]
    registration, lock, result = read(reg_path), read(lock_path), read(result_path)
    discovery_audit_path = args.directory / "discovery_audit.json"
    discovery_audit = read(discovery_audit_path)
    reg_hash = sha(reg_path)
    require(discovery_audit["status"] == "passed" and discovery_audit["discovery_selection_sha256"] == sha(lock_path), "Discovery audit changed")
    require(result["status"] == "passed_development_internal_causal_tests" and result["registration_sha256"] == reg_hash
            and lock["registration_sha256"] == reg_hash and result["discovery_selection_sha256"] == sha(lock_path), "Registered causal stages incomplete")
    require(result["n_matches"] == 11 and result["n_pairs_seeds"] == 22 and result["chosen_sites"] == lock["chosen_sites"], "Wrong selection cohort or sites")
    require(registration["selection_conditions"] == CONDITIONS and registration["matched_control_comparisons"] == CONTROL_PAIRS,
            "Condition protocol changed")
    frozen = {str(path): sha(path) for path in (reg_path, lock_path, result_path, discovery_audit_path, Path(__file__))}
    for name, expected in registration["code_sha256"].items():
        require(sha(ROOT/name) == expected, "Execution code changed")
        frozen[str(ROOT/name)] = expected
    for section in ("input_bindings", "probe_bindings"):
        for entry in registration[section].values():
            require(sha(entry["path"]) == entry["sha256"], "Registered input changed")
            frozen[entry["path"]] = entry["sha256"]
    selected = read(registration["probe_bindings"]["selection"]["path"])
    models = {model.name: model.main for model in load_selected_models(registration["probe_bindings"]["archive"]["path"], selected)}
    output_probe = models[registration["output_probe"]]
    projection, _ = np.linalg.qr(np.random.default_rng(20260907).standard_normal((2048, 128)))
    projection = projection.astype(np.float32)
    pairs = {pair["match_id"]: pair for pair in registration["pairs"] if pair["split"] == "selection"}
    expected = {(match, seed) for match in pairs for seed in registration["paired_noise_seeds"]}
    require(len(pairs) == 11 and len(result["pairs"]) == 22
            and {(entry["match_id"], entry["seed"]) for entry in result["pairs"]} == expected, "Missing/duplicate selection pairs")
    sites = lock["chosen_site_indices"]
    checked, exact_rows, attributions = [], [], []
    max_proxy_error, max_attribution_error, max_ablation_error = 0., 0., 0.
    for pair_entry in result["pairs"]:
        path = Path(pair_entry["path"])
        require(sha(path) == pair_entry["sha256"], "Pair report changed")
        frozen[str(path)] = pair_entry["sha256"]
        pair_result = read(path)
        match, seed = pair_result["match_id"], pair_result["seed"]
        require(pair_result["split"] == "selection" and (match, seed) in expected
                and pair_result["registration_sha256"] == reg_hash and pair_result["status"] == "passed", "Forbidden/incorrect role before targets")
        pair = pairs[match]
        for side in ("recipient", "donor"):
            require(pair_result[f"{side}_clip_id"] == pair[side]["clip_id"] and pair_result[f"{side}_prepared_sha256"] == pair[side]["sha256"], "Pair source ID differs")
            source_path = Path(pair[side]["artifact_path"])
            require(sha(source_path) == pair[side]["sha256"], "Source clip changed")
            frozen[str(source_path)] = pair[side]["sha256"]
            with np.load(source_path, allow_pickle=False) as source:
                require(float(source["targets"][0, 15, 2]) == pair_result[f"{side}_source_ball_z"]
                        and float(source["timestamps"][0, 15]) == pair_result[f"{side}_target_timestamp"]
                        and source["player_ids"].astype(str).tolist() == pair_result["canonical_player_ids"], "Wrong own-view target/time/entity")
        records = {(row["site_index"], row["condition"]): row for row in pair_result["conditions"]}
        require(len(pair_result["conditions"]) == len(records) == len(sites)*len(CONDITIONS)
                and set(records) == {(site, condition) for site in sites for condition in CONDITIONS}, "Missing/extra/duplicate exact conditions")
        require(pair_result["gradient_forward_repeat_bitwise_equal"] and pair_result["no_input_or_action_changes"], "Input or repeat control failed")
        artifact = Path(pair_result["artifact_path"])
        require(sha(artifact) == pair_result["artifact_sha256"], "Pair tensor archive changed")
        frozen[str(artifact)] = pair_result["artifact_sha256"]
        with np.load(artifact, allow_pickle=False) as arrays:
            baseline, z_tau, noise, clean = [arrays[key] for key in ("baseline_flow", "recipient_z_t", "recipient_noise", "recipient_clean_codec")]
            require(baseline.shape == z_tau.shape == noise.shape == (1, 8, 36, 16, 32) and clean.shape == (4, 8, 9, 16, 32), "Unexpected original flow/input layout")
            require(array_sha(z_tau) == pair_result["input_sha256"]["z_t"] and array_sha(noise) == pair_result["paired_noise_sha256"]
                    and array_sha(baseline) == pair_result["baseline"]["prediction_sha256"], "Saved original input/noise/output differs")
            clean_joint = clean.transpose(1, 0, 2, 3, 4).reshape(1, 8, 36, 16, 32)
            require(np.array_equal(z_tau, .5*clean_joint+.5*noise), "Incorrect tau0.5 interpolant")
            target = float(arrays["donor_source_targets_final"][0, 2])
            require(target == pair_result["donor_source_ball_z"], "Wrong final donor height target")
            base_endpoint = z_tau+.5*baseline
            base_proxy = output_probe.predict(tile(base_endpoint).reshape(1, -1))[0]
            base_objective = -float(((base_proxy[2]-target)/output_probe.y_scale[2])**2)
            close(base_objective, pair_result["baseline"]["objective"], "Baseline objective differs")
            base_mse = float(np.square(base_endpoint.astype(float)-clean_joint).mean())
            for site in sites:
                name = registration["sites"][site]
                rec, donor, grad = [arrays[f"site{site}_{suffix}"] for suffix in ("recipient_tile", "donor_tile", "objective_gradient_tile")]
                require(rec.shape == donor.shape == grad.shape == (9, 16, 2048)
                        and all(np.isfinite(x).all() for x in (rec, donor, grad)), "Invalid site tensors")
                attribution = float(np.sum((donor-rec).astype(float)*grad.astype(float)))
                original_attribution = pair_result["scores"][site]["attribution"]
                close(attribution, original_attribution, "Saved gradient/donor difference does not reproduce AtP", atol=1e-9)
                max_attribution_error = max(max_attribution_error, abs(attribution-original_attribution))
                probe = models[f"spatial/{name}/absolute30/position"]
                direction = probe_direction(probe, projection, 2)
                close(direction, arrays[f"site{site}_ball_z_pullback"], "Frozen spatial pullback changed", atol=2e-8, rtol=2e-5)
                unit = direction/np.float32(norm(direction))
                wrong = probe_direction(probe, projection, 0)
                wrong_unit = wrong/np.float32(norm(wrong))
                dose = float(np.sum((donor-rec).astype(float)*unit.astype(float)))
                component = arrays[f"site{site}_component_transfer_replacement_tile"]
                close(component, rec+np.float32(dose)*unit, "Component transfer direction/dose differs", atol=2e-5, rtol=1e-6)
                ablated = arrays[f"site{site}_ablate_to_discovery_mean_replacement_tile"]
                ablation_error = abs(float(probe.predict(bin_features(ablated, projection))[0, 2])-probe.y_mean[2])
                require(ablation_error < .01, "Ablation does not remove centered height to within0.01 raw units")
                max_ablation_error = max(max_ablation_error, ablation_error)
                close(arrays[f"site{site}_wrong_variable_ball_x_replacement_tile"], rec+np.float32(dose)*wrong_unit,
                      "Wrong-variable control direction/dose differs", atol=2e-5, rtol=1e-6)
                require(np.array_equal(arrays[f"site{site}_exact_donor_replacement_tile"], donor), "Exact donor tensor differs")
                for condition in ("self", "restore"):
                    require(np.array_equal(arrays[f"site{site}_{condition}_replacement_tile"], rec), "No-op/restoration tile differs")
                actual_gains = {}
                for condition in CONDITIONS:
                    row = records[site, condition]
                    prediction = arrays[f"site{site}_{condition}_flow"]
                    replacement = arrays[f"site{site}_{condition}_replacement_tile"]
                    require(prediction.shape == baseline.shape and np.isfinite(prediction).all()
                            and replacement.shape == rec.shape and np.isfinite(replacement).all(), "Invalid edited tensors")
                    require(array_sha(prediction) == row["prediction_sha256"], "Edited original output hash differs")
                    if condition in ("self", "restore"):
                        require(np.array_equal(prediction, baseline), "No-op/restoration fails bitwise original-output equality")
                    past_equal = np.array_equal(prediction[:, :7], baseline[:, :7])
                    require(past_equal, "Final-latent edit changed earlier flow outputs")
                    estimate = z_tau+.5*prediction
                    require(np.array_equal(tile(estimate), arrays[f"site{site}_{condition}_endpoint_tile"]), "Saved endpoint differs from original flow")
                    predicted = output_probe.predict(tile(estimate).reshape(1, -1))[0]
                    proxy_error = float(np.max(np.abs(predicted-np.asarray(row["output_position_proxy"]))))
                    max_proxy_error = max(max_proxy_error, proxy_error)
                    close(predicted, row["output_position_proxy"], "Edited frozen output proxy differs")
                    gain = -float(((predicted[2]-target)/output_probe.y_scale[2])**2)-base_objective
                    close(gain, row["standardized_error_gain"], "Edited donor-target error gain differs")
                    actual_gains[condition] = gain
                    shift = (predicted-base_proxy)/output_probe.y_scale
                    close(shift, row["proxy_shift_standardized"], "Collateral position shift differs")
                    flow_norm, end_norm = norm(prediction.astype(float)-baseline), norm(estimate.astype(float)-base_endpoint)
                    close(flow_norm, row["flow_delta_l2"], "Flow delta norm differs", rtol=1e-5)
                    close(end_norm, row["endpoint_delta_l2"], "Endpoint delta norm differs", rtol=1e-5)
                    source_mse = float(np.square(estimate.astype(float)-clean_joint).mean())
                    close(source_mse, row["endpoint_recipient_source_mse"], "Endpoint source reconstruction MSE differs", rtol=1e-5)
                    views = np.stack([tile(estimate, view) for view in range(4)]).reshape(4, -1)
                    close(output_probe.predict(views), row["view_final_position_proxies"], "Other-view position proxy differs")
                    if condition != "wrong_view_component":
                        close(norm(replacement.astype(float)-rec), row["realized_tile_delta_l2"], "Realized edit norm differs", rtol=1e-5, atol=1e-5)
                    exact_rows.append({"match_id": match, "seed": seed, "site_index": site, "site": name,
                                       "condition": condition, "gain": gain, "flow_delta_l2": flow_norm,
                                       "proxy_shift_standardized": shift.tolist(), "endpoint_source_mse_change": source_mse-base_mse,
                                       "past_flow_unchanged": past_equal})
                for control, intervention in CONTROL_PAIRS.items():
                    if control != "wrong_view_component":
                        a = norm(arrays[f"site{site}_{control}_replacement_tile"].astype(float)-rec)
                        b = norm(arrays[f"site{site}_{intervention}_replacement_tile"].astype(float)-rec)
                        close(a, b, "Matched control norm differs", atol=1e-4, rtol=1e-4)
                    else:
                        close(records[site, control]["realized_tile_delta_l2"], records[site, intervention]["realized_tile_delta_l2"],
                              "Recorded wrong-view and component norms differ", atol=1e-4, rtol=1e-4)
                attributions.append({"match_id": match, "seed": seed, "site": name, "site_index": site,
                                     "attribution": attribution, "exact_donor_gain": actual_gains["exact_donor"]})
        checked.append({"match_id": match, "seed": seed, "conditions_checked": len(records), "all_sites_checked": sites})
        print(f"Audited {len(checked)}/22 paired runs", flush=True)
    summaries, paired, agreement = [], [], []
    for site in sites:
        for condition in CONDITIONS:
            values = match_summary(exact_rows, site, condition)
            published = next(row for row in result["comparisons"] if row["site"] == registration["sites"][site] and row["condition"] == condition)
            for key in ("mean_standardized_error_gain", "match_bootstrap_ci95", "positive_gain_matches", "mean_flow_delta_l2"):
                close(values[key], published[key], f"Published match summary differs: {key}", rtol=1e-5 if key == "mean_flow_delta_l2" else 1e-8)
            close(list(values["match_gains"].values()), [published["match_gains"][match] for match in values["match_gains"]], "Published match gains differ")
            values["site"] = registration["sites"][site]
            summaries.append(values)
        by_condition = {row["condition"]: row for row in summaries if row["site_index"] == site}
        matches = sorted(by_condition["self"]["match_gains"])
        draws = np.random.default_rng(20260907).integers(0, 11, (500, 11))
        for control, intervention in CONTROL_PAIRS.items():
            differences = np.asarray([by_condition[intervention]["match_gains"][match]-by_condition[control]["match_gains"][match] for match in matches])
            values = {"site": registration["sites"][site], "intervention": intervention, "control": control,
                      "mean_error_gain_above_control": float(differences.mean()),
                      "paired_match_bootstrap_ci95": np.quantile(differences[draws].mean(1), [.025, .975]).tolist(),
                      "match_differences": dict(zip(matches, differences.tolist()))}
            published = next(row for row in result["paired_control_comparisons"] if row["site"] == values["site"] and row["control"] == control)
            for key in ("mean_error_gain_above_control", "paired_match_bootstrap_ci95"):
                close(values[key], published[key], "Published paired-control gain differs")
            paired.append(values)
        a = [row for row in attributions if row["site_index"] == site]
        approx = np.asarray([np.mean([row["attribution"] for row in a if row["match_id"] == match]) for match in matches])
        exact = np.asarray([np.mean([row["exact_donor_gain"] for row in a if row["match_id"] == match]) for match in matches])
        agreement.append({"site": registration["sites"][site], "n_matches": 11, "mean_attribution": float(approx.mean()),
                          "mean_exact_donor_gain": float(exact.mean()), "pearson_across_seed_averaged_matches": correlation(approx, exact),
                          "spearman_across_seed_averaged_matches": correlation(ranks(approx), ranks(exact)),
                          "positive_sign_agreement_fraction": float(((approx>0)==(exact>0)).mean()),
                          "approximation_rmse": float(np.sqrt(np.square(approx-exact).mean())),
                          "match_values": [{"match_id": match, "attribution": float(ap), "exact_donor_gain": float(ex)} for match, ap, ex in zip(matches, approx, exact)]})
    require(all(sha(path) == expected for path, expected in frozen.items()), "Evidence changed during independent audit")
    report = {"status": "passed", "scope": "development-only internal output-proxy causal interventions; no physical control claim",
              "selection_results_sha256": sha(result_path), "discovery_selection_sha256": sha(lock_path), "registration_sha256": reg_hash,
              "audit_script_sha256": sha(Path(__file__)), "n_matches": 11, "n_pairs_seeds": 22, "conditions_checked": len(exact_rows),
              "all_noop_and_restore_outputs_bitwise_equal": True, "all_earlier_flow_frames_bitwise_unchanged": True,
              "selected_site_attribution_dot_products_recomputed": True, "all_saved_output_metrics_recomputed": True,
              "paired_match_summaries_and_intervals_recomputed": True, "max_output_proxy_abs_error": max_proxy_error,
              "max_attribution_abs_error": max_attribution_error, "max_ablation_target_raw_error": max_ablation_error,
              "wrong_view_audit_limit": "Original view1 residual tile was not persisted; its replacement location and unchanged-input condition rely on registered hook code/tests. Saved runtime wrong-view norms are checked against component norms, not independently reconstructed.",
              "ablation_precision_note": "Independent explicit FP64 bin projection must reach the discovery mean within0.01 raw units; execution used FP32 bin arithmetic.",
              "uncertainty": registration["uncertainty"], "physical_control_established": False, "gpu_used": False,
              "confirmation_arrays_read": False, "target_names": next(row["target_names"] for row in selected["models"] if row["name"] == registration["output_probe"]),
              "pairs": checked, "comparisons": summaries, "paired_control_comparisons": paired,
              "attribution_vs_exact": agreement, "per_seed_condition_metrics": exact_rows}
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "n_matches", "n_pairs_seeds", "conditions_checked", "max_output_proxy_abs_error", "max_attribution_abs_error", "max_ablation_target_raw_error", "attribution_vs_exact")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
