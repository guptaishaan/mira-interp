#!/usr/bin/env python3
"""Register and run narrow internal causal tests after the development probe audit.

Phases are deliberately separate: register -> reserved pilot -> discovery
attribution ranking -> development selection exact interventions. No confirmation
match is accepted. A codec-trained output proxy is not a generated-video physics
measurement. This path uses a single teacher-forced flow evaluation without cache.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "8"
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mira/src"), str(ROOT / "scripts")]
from mira_interp.causal_development import (SITES, ResidualHooks, endpoint, intervention_tiles,
    matched_pairs, output_objective, spatial_pullback, tensor_hash, tile, tile_features, torch_predict)
from mira_interp.development_capture import channel_projection, forward, make_inputs
from mira_interp.model_loading import load_pretrained_four_player, sha256
from mira_interp.probes import load_selected_models
from analyze_development import require, write

SEEDS = [2026090701, 2026090702]
CODE = ["scripts/causal_development.py", "src/mira_interp/causal_development.py",
        "src/mira_interp/development_capture.py", "src/mira_interp/model_loading.py",
        "src/mira_interp/pretrained.py", "src/mira_interp/probes.py", "scripts/analyze_development.py"]
OUTPUT_PROBE = "codec_spatial_X/absolute30/position"
CONDITIONS = ["self", "exact_donor", "component_transfer", "ablate_to_discovery_mean", "restore",
              "norm_matched_random", "norm_matched_random_full", "norm_matched_random_ablation",
              "wrong_variable_ball_x", "wrong_view_component"]
MATCHED_CONTROLS = {"norm_matched_random": "component_transfer",
                    "norm_matched_random_full": "exact_donor",
                    "norm_matched_random_ablation": "ablate_to_discovery_mean",
                    "wrong_variable_ball_x": "component_transfer", "wrong_view_component": "component_transfer"}
FD_RELATIVE_STEPS = [1e-4, 1e-3, 1e-2]


def read(path):
    return json.loads(Path(path).read_text())


def current_code():
    return {name: sha256(ROOT / name) for name in CODE}


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def bind_probes(args):
    selection_path = args.probe_dir / "development_selection.json"
    archive_path = args.probe_dir / "development_models.npz"
    report_path = args.probe_dir / "development_report.json"
    selection, audit, report = read(selection_path), read(args.probe_audit), read(report_path)
    require(selection["status"] == "frozen_development_choices" and selection["confirmation_used"] is False,
            "Frozen development probes excluding confirmation are required")
    require(report["status"] == "passed_development_analysis" and report["development_selection_sha256"] == sha256(selection_path),
            "Development probe analysis did not complete")
    require(audit["status"] == "passed" and audit.get("development_selection_sha256") == sha256(selection_path)
            and audit.get("model_archive_sha256") == sha256(archive_path), "Independent probe audit must bind these exact choices and coefficients")
    require(selection["model_archive_sha256"] == sha256(archive_path), "Coefficient archive changed")
    require(selection["provenance"]["match_counts"] == {"discovery": 31, "selection": 11}, "Wrong development cohort")
    require(all(sha256(ROOT / name) == expected for name, expected in selection["analysis_code_sha256"].items()),
            "Probe implementation changed after fitting")
    all_models = load_selected_models(archive_path, selection)
    models = {model.name: model.main for model in all_models}
    metadata = {entry["name"]: entry for entry in selection["models"]}
    names = [OUTPUT_PROBE] + [f"spatial/{site}/absolute30/position" for site in SITES]
    require(all(name in models for name in names), "Output codec probe and all17 spatial site probes are required")
    for name in names:
        require(metadata[name]["target_names"][:3] == ["ball.location.x", "ball.location.y", "ball.location.z"], "Position column order changed")
        require(metadata[name]["feature_spec"]["temporal"] == "current", "Temporal probes are outside this causal protocol")
    return {name: models[name] for name in names}, selection, {
        "selection": binding(selection_path), "archive": binding(archive_path),
        "report": binding(report_path), "audit": binding(args.probe_audit)}


def register(args):
    _, probes, probe_bindings = bind_probes(args)
    paths = {"capture_manifest": ROOT / "results/development_capture_manifest.json",
             "capture_audit": ROOT / "results/development_capture_audit.json",
             "preparation": ROOT / "results/development_prepare.json",
             "pilot_preparation": ROOT / "results/development_prepare_pilot.json",
             "split": ROOT / "data/qualified_split_manifest.json",
             "development_protocol": ROOT / "configs/development_v3.json"}
    capture, audit, prepared, split = (read(paths[name]) for name in ("capture_manifest", "capture_audit", "preparation", "split"))
    require(capture["status"] == audit["status"] == prepared["status"] == "passed", "Capture/preparation audit gates must pass")
    require(audit.get("pilot_only") is False and audit["capture_manifest_sha256"] == sha256(paths["capture_manifest"])
            and audit["all_source_label_joins_exact"] and audit["all_layers_complete"], "Independent full capture audit differs")
    require(probes["provenance"]["capture_manifest_sha256"] == sha256(paths["capture_manifest"])
            and probes["provenance"]["capture_audit_sha256"] == sha256(paths["capture_audit"])
            and probes["provenance"]["preparation_manifest_sha256"] == sha256(paths["preparation"]), "Probe/capture/source lineage differs")
    assignments = {entry["match_id"]: entry for entry in split["matches"]}
    pairs = matched_pairs(prepared["records"], assignments)
    counts = {role: sum(pair["split"] == role for pair in pairs) for role in ("discovery", "selection")}
    require(counts == {"discovery": 31, "selection": 11}, "Exact original31/11 matches required")
    capture_inputs = {row["clip_id"]: row["input_sha256"] for row in capture["records"]}
    for row in prepared["records"]:
        require(capture_inputs.get(row["clip_id"]) == row["sha256"], "Prepared input differs from probed capture")
    pilot = read(paths["pilot_preparation"])
    require(pilot["status"] == "passed" and len(pilot["records"]) == 1 and pilot["records"][0]["split"] == "pilot", "Reserved single-clip pilot required")
    registration = {
        "status": "registered_before_causal_execution", "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": 1, "scope": "development-only internal output-proxy causality; not physical trajectory control",
        "primary_target": "ball.location.z", "objective": "-((f_codec(z_tau+(1-tau)*v_pred)[ball_z]-donor_source_ball_z)/discovery_ball_z_std)^2",
        "output_probe": OUTPUT_PROBE, "output_probe_distribution_note": "Probe was trained on observed clean codec tokens. Its application to model endpoint estimates and edited outputs is an unvalidated distribution shift.",
        "input_dtype": "checkpoint FP32, BF16 autocast, FP32 fractional RGB and tau mixing", "tau": 0.5,
        "sites": SITES, "edit_support": {"view": 0, "latent": 7, "spatial": "entire9x16 tile"},
        "pair_rule": "Within each original match, lexicographically first clip is recipient and last is donor, independent of labels and intervention outcomes.",
        "natural_donor_note": "Donor video/actions/state can differ in many variables. Recipient video/actions/past/noise are fixed across every intervention. No one-variable counterfactual claim.",
        "paired_noise_seeds": SEEDS, "match_counts": counts, "pairs": pairs, "pilot_record": pilot["records"][0],
        "discovery_rank": "Mean gradient dot donor-minus-recipient tile across31matches and2pairedseeds; retain all17 signed scores. Top3 descending score, ties by canonical site index.",
        "selection_conditions": CONDITIONS, "component_direction": "Each site's fixed current spatial absolute-position probe, ball-z coefficient with discovery normalization pulled back through projection and equal-area bin means.",
        "ablation": "Remove centered ball-z probe component toward discovery target mean. Restoration reinstalls exact cached recipient, an implementation equality control.",
        "random_control": "Same edit support and L2 norm as component transfer, isotropic normal direction with paired deterministic seed.",
        "additional_random_controls": "Same random unit direction, separately scaled to full donor delta norm and ablation delta norm; each exact intervention has its own norm control.",
        "matched_control_comparisons": MATCHED_CONTROLS,
        "pilot_finite_difference": {"sites": [0, 8, 16], "direction": "unit frozen spatial ball-z pullback", "central_difference_relative_tile_l2_steps": FD_RELATIVE_STEPS,
                                    "interpretation": "Report directional derivative, finite central difference and linearization error at every fixed step. BF16 forward is quantized; agreement is diagnostic, not a success-based research exclusion rule."},
        "wrong_variable": "Ball-x probe direction with same signed component dose/norm; report direction cosine.",
        "wrong_view": "Apply same component delta in view1, preserving its original tile. This is another-view control, not a wrong-entity control because the ball is global.",
        "metrics": "Raw standardized error gain toward donor target; all15 output position proxy shifts; original model flow/endpoint L2 change; source endpoint reconstruction MSE. No recovery ratio.",
        "uncertainty": "Aggregate paired seeds within matches; descriptive whole-match bootstrap across11developmentselection matches, conditional on all choices. No multiplicity or fresh-confirmation claim.",
        "secondary_velocity": "Not included in primary execution; any ball-velocity-z extension requires a separately documented exploratory scope and codec-probe adequacy assessment after primary.",
        "artifact_scope": "All discovery rankings; selection recipient/donor/gradient tiles at chosen sites; all condition full original flow outputs and endpoint tiles for subsequent geometry comparisons.",
        "sampling": "Single teacher-forced forward only; no streaming KV cache, rollout or decoded trajectory validation.",
        "confirmation_used": False, "new_confirmation_required": True,
        "probe_bindings": probe_bindings, "input_bindings": {name: binding(path) for name, path in paths.items()},
        "code_sha256": current_code(),
    }
    write(args.output_dir / "registration.json", registration, exclusive=True)
    print("Registered causal protocol; no GPU execution occurred.", flush=True)


def verify_registration(args):
    path = args.output_dir / "registration.json"
    registration = read(path)
    require(registration["code_sha256"] == current_code(), "Causal implementation changed after registration")
    for section in ("probe_bindings", "input_bindings"):
        for entry in registration[section].values():
            require(sha256(Path(entry["path"])) == entry["sha256"], "Registered input/probe/audit changed")
    models, _, bindings = bind_probes(args)
    require(bindings == registration["probe_bindings"], "Different probes/audit supplied")
    require(registration["paired_noise_seeds"] == SEEDS and registration["sites"] == SITES, "Causal constants differ")
    return registration, sha256(path), models


def input_arrays(record):
    require(record["split"] in {"discovery", "selection", "pilot"}, "Confirmation inputs forbidden")
    path = Path(record["artifact_path"])
    require(sha256(path) == record["sha256"], "Registered prepared clip changed")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("frames", "actions", "targets", "player_ids", "target_names", "timestamps", "source_frame_indices")}
    require(arrays["target_names"].tolist()[2] == "ball.location.z" and arrays["targets"].shape == (4, 16, 30), "Invalid registered source target layout")
    return arrays


def prepare(model, arrays, seed):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inputs, clean, noise = make_inputs(model, arrays["frames"], arrays["actions"], seed)
    inputs = {key: value.detach() for key, value in inputs.items()}
    return inputs, clean.detach(), noise.detach()


def input_hashes(inputs):
    return {key: tensor_hash(value) for key, value in inputs.items()}


def source_frame(arrays):
    indices = arrays["source_frame_indices"]
    require(indices.shape in {(16,), (4, 16)}, "Unexpected source frame-index layout")
    return int(indices[15] if indices.ndim == 1 else indices[0, 15])


def attribution(model, recipient, donor, probe, donor_z):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with ResidualHooks(model.world_model.transformer, capture=True) as donor_hooks:
            donor_prediction = forward(model, donor)
    require(len(donor_hooks.values) == 17, "Missing donor residual sites")
    recipient["z_t"] = recipient["z_t"].detach().requires_grad_(True)
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with ResidualHooks(model.world_model.transformer, capture=True, differentiable=True) as hooks:
            prediction = forward(model, recipient)
        objective, values, _ = output_objective(recipient, prediction, probe, donor_z)
        gradients = torch.autograd.grad(objective, [hooks.values[index] for index in range(17)])
    result, recipients, grad_tiles = [], {}, {}
    for index, gradient in enumerate(gradients):
        raw, grad = tile(hooks.values[index]).detach().float(), tile(gradient).detach().float()
        require(torch.isfinite(raw).all() and torch.isfinite(grad).all(), "Nonfinite activation/gradient")
        difference = donor_hooks.values[index] - raw
        score = float((difference.double() * grad.double()).sum())
        result.append({"site": SITES[index], "site_index": index, "attribution": score,
                       "donor_delta_l2": float(torch.linalg.vector_norm(difference)), "gradient_l2": float(torch.linalg.vector_norm(grad))})
        recipients[index], grad_tiles[index] = raw.clone(), grad.clone()
    return {"scores": result, "objective": float(objective.detach()), "output_position_proxy": values.detach().cpu().tolist()}, prediction.detach(), donor_prediction.detach(), recipients, donor_hooks.values, grad_tiles


def metric(inputs, prediction, baseline, probe, donor_z, clean):
    objective, values, estimate = output_objective(inputs, prediction, probe, donor_z)
    _, baseline_values, baseline_estimate = output_objective(inputs, baseline, probe, donor_z)
    view_endpoints = torch.stack([tile(estimate, view=view) for view in range(4)])
    clean_joint = clean.permute(1, 0, 2, 3, 4).reshape(1, 8, 36, 16, 32)
    return {"objective": float(objective), "output_position_proxy": values.cpu().tolist(),
            "proxy_shift_raw": (values-baseline_values).cpu().tolist(),
            "proxy_shift_standardized": ((values-baseline_values)/torch.as_tensor(probe.y_scale, device=values.device)).cpu().tolist(),
            "flow_delta_l2": float(torch.linalg.vector_norm(prediction.float()-baseline.float())),
            "endpoint_delta_l2": float(torch.linalg.vector_norm(estimate-baseline_estimate)),
            "endpoint_recipient_source_mse": float((estimate-clean_joint.float()).square().mean()),
            "view_final_position_proxies": torch_predict(probe, view_endpoints.reshape(4, -1)).cpu().tolist(),
            "prediction_sha256": tensor_hash(prediction)}


def run_pair(model, pair, seed, models, projection, sites, *, pilot=False):
    rec, don = input_arrays(pair["recipient"]), input_arrays(pair["donor"])
    require(np.array_equal(rec["player_ids"], don["player_ids"]), "Natural pair must preserve canonical player identities")
    recipient, clean, noise = prepare(model, rec, seed)
    donor, donor_clean, donor_noise = prepare(model, don, seed)
    require(torch.equal(noise, donor_noise), "Paired donor/recipient noise differs")
    hashes = input_hashes(recipient)
    output_probe = models[OUTPUT_PROBE]
    donor_z, recipient_z = float(don["targets"][0, 15, 2]), float(rec["targets"][0, 15, 2])
    result, baseline, donor_prediction, rec_tiles, don_tiles, gradients = attribution(model, recipient, donor, output_probe, donor_z)
    recipient["z_t"] = recipient["z_t"].detach()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        repeated = forward(model, recipient)
    require(torch.equal(baseline, repeated), "Gradient capture changed original model prediction")
    result.update(match_id=pair["match_id"], split=pair["split"], seed=seed, input_sha256=hashes,
                  recipient_clip_id=pair["recipient"]["clip_id"], donor_clip_id=pair["donor"]["clip_id"],
                  donor_source_ball_z=donor_z, recipient_source_ball_z=recipient_z,
                  recipient_target_timestamp=float(rec["timestamps"][0, 15]), donor_target_timestamp=float(don["timestamps"][0, 15]),
                  recipient_source_frame_index=source_frame(rec), donor_source_frame_index=source_frame(don),
                  canonical_player_ids=rec["player_ids"].astype(str).tolist(), source_view=0, source_local_frame=15, latent_frame=7,
                  recipient_prepared_sha256=pair["recipient"]["sha256"], donor_prepared_sha256=pair["donor"]["sha256"],
                  paired_noise_sha256=tensor_hash(noise), gradient_forward_repeat_bitwise_equal=True,
                  conditions=[], controls={"repeat": True}, no_input_or_action_changes=True)
    arrays = {"baseline_flow": baseline.float().cpu().numpy(), "donor_flow": donor_prediction.float().cpu().numpy(),
              "recipient_clean_codec": clean.float().cpu().numpy(), "donor_clean_codec": donor_clean.float().cpu().numpy(),
              "recipient_noise": noise.float().cpu().numpy(), "recipient_z_t": recipient["z_t"].float().cpu().numpy(),
              "recipient_source_targets_final": rec["targets"][:, 15], "donor_source_targets_final": don["targets"][:, 15]}
    with torch.no_grad():
        base_metric = metric(recipient, baseline, baseline, output_probe, donor_z, clean)
        result["baseline"] = base_metric
        clean_pred = torch_predict(output_probe, clean[0, 7].reshape(1, -1))[0]
        result["recipient_clean_codec_position_proxy"] = clean_pred.cpu().tolist()
        result["clean_codec_vs_endpoint_ball_z_shift"] = float(base_metric["output_position_proxy"][2]-clean_pred[2])
        for site in sites:
            name = SITES[site]
            site_probe = models[f"spatial/{name}/absolute30/position"]
            tiles, delta, details = intervention_tiles(rec_tiles[site], don_tiles[site], site_probe, projection, seed=seed+1000+site)
            # Wrong-view recipient baseline must be captured directly at that view.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with ResidualHooks(model.world_model.transformer, capture=True, view=1) as wrong_hooks:
                    wrong_capture_prediction = forward(model, recipient)
            require(torch.equal(baseline, wrong_capture_prediction), "Wrong-view capture is not a no-op")
            tiles["wrong_view_component"] = wrong_hooks.values[site] + delta
            arrays[f"site{site}_recipient_tile"] = rec_tiles[site].cpu().numpy()
            arrays[f"site{site}_donor_tile"] = don_tiles[site].cpu().numpy()
            arrays[f"site{site}_objective_gradient_tile"] = gradients[site].cpu().numpy()
            direction = spatial_pullback(site_probe, projection, 2)
            arrays[f"site{site}_ball_z_pullback"] = direction.cpu().numpy()
            # Independent scalar agreement of raw-space derivative and descriptor
            # readout is an engineering check, including normalization/bin area.
            before = torch_predict(site_probe, tile_features(rec_tiles[site], projection))[0, 2]
            after = torch_predict(site_probe, tile_features(tiles["component_transfer"], projection))[0, 2]
            predicted_shift = ((tiles["component_transfer"]-rec_tiles[site]).double()*direction.double()).sum()
            require(abs(float(after-before-predicted_shift)) <= 2e-3 + 1e-4*abs(float(predicted_shift)), "Spatial pullback/readout finite change disagrees")
            for condition in CONDITIONS:
                view = 1 if condition == "wrong_view_component" else 0
                replacement = tiles[condition]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with ResidualHooks(model.world_model.transformer, site=site, replacement=replacement, view=view):
                        prediction = forward(model, recipient)
                require(torch.isfinite(prediction).all(), "Nonfinite intervention prediction")
                if condition in {"self", "restore"}:
                    require(torch.equal(prediction, baseline), f"{condition} is not exactly the baseline")
                values = metric(recipient, prediction, baseline, output_probe, donor_z, clean)
                values.update(site=name, site_index=site, condition=condition, **details,
                              standardized_error_gain=values["objective"]-base_metric["objective"],
                              realized_tile_delta_l2=float(torch.linalg.vector_norm(replacement-(wrong_hooks.values[site] if view == 1 else rec_tiles[site]))))
                result["conditions"].append(values)
                arrays[f"site{site}_{condition}_flow"] = prediction.float().cpu().numpy()
                arrays[f"site{site}_{condition}_endpoint_tile"] = tile(endpoint(recipient, prediction)).cpu().numpy()
                arrays[f"site{site}_{condition}_replacement_tile"] = replacement.cpu().numpy()
            result["controls"][name] = {"self_bitwise_equal": True, "restore_bitwise_equal": True,
                                         "pullback_scalar_abs_error": abs(float(after-before-predicted_shift))}
            if pilot:
                unit = direction / torch.linalg.vector_norm(direction)
                derivative = float((gradients[site].double()*unit.double()).sum())
                radius = float(torch.linalg.vector_norm(rec_tiles[site]))
                require(radius > 0, "Pilot residual radius is zero")
                checks = []
                for relative in FD_RELATIVE_STEPS:
                    step = radius*relative
                    objectives = []
                    for sign in (-1, 1):
                        replacement = rec_tiles[site]+sign*step*unit
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            with ResidualHooks(model.world_model.transformer, site=site, replacement=replacement):
                                prediction = forward(model, recipient)
                        objective = float(output_objective(recipient, prediction, output_probe, donor_z)[0])
                        require(np.isfinite(objective), "Pilot finite-difference output is nonfinite")
                        objectives.append(objective)
                    finite = (objectives[1]-objectives[0])/(2*step)
                    checks.append({"relative_tile_l2_step": relative, "absolute_l2_step": step,
                                   "analytic_directional_derivative": derivative, "central_finite_difference": finite,
                                   "absolute_derivative_error": abs(finite-derivative),
                                   "relative_derivative_error": abs(finite-derivative)/max(abs(finite), abs(derivative), 1e-12),
                                   "minus_objective": objectives[0], "plus_objective": objectives[1],
                                   "plus_first_order_prediction": base_metric["objective"]+step*derivative,
                                   "plus_linearization_abs_error": abs(objectives[1]-base_metric["objective"]-step*derivative)})
                result["controls"][name]["finite_difference_diagnostics"] = checks
    require(input_hashes(recipient) == hashes, "Recipient inputs/actions changed during interventions")
    result["engineering_only_pilot"] = pilot
    return result, arrays


def save_pair(args, registration_hash, result, arrays):
    stem = f"{result['match_id']}_seed{result['seed']}"
    folder = args.output_dir / result["split"]
    folder.mkdir(parents=True, exist_ok=True)
    artifact_folder = args.artifact_dir / result["split"]
    artifact_folder.mkdir(parents=True, exist_ok=True)
    array_path = artifact_folder / (stem + ".npz")
    report_path = folder / (stem + ".json")
    require(not report_path.exists() and not array_path.exists(), "Existing causal pair artifacts are immutable")
    np.savez_compressed(array_path, **arrays)
    result.update(status="passed", registration_sha256=registration_hash,
                  artifact_path=str(array_path.resolve()), artifact_sha256=sha256(array_path),
                  matched_control_comparisons=MATCHED_CONTROLS,
                  artifact_source_labels_private=True)
    write(report_path, result, exclusive=True)
    return {"path": str(report_path.resolve()), "sha256": sha256(report_path), "match_id": result["match_id"], "seed": result["seed"]}


def summary(rows, sites):
    out = []
    for site in sites:
        for condition in CONDITIONS:
            values = [(row["match_id"], next(x for x in row["conditions"] if x["site_index"] == site and x["condition"] == condition)) for row in rows]
            matches = sorted({match for match, _ in values})
            gains = np.asarray([np.mean([x["standardized_error_gain"] for match, x in values if match == mid]) for mid in matches])
            draws = np.random.default_rng(20260907).integers(0, len(gains), (500, len(gains)))
            out.append({"site": SITES[site], "condition": condition, "n_matches": len(matches), "n_paired_seeds": 2,
                        "mean_standardized_error_gain": float(gains.mean()), "match_bootstrap_ci95": np.quantile(gains[draws].mean(1), [.025, .975]).tolist(),
                        "positive_gain_matches": int((gains>0).sum()), "match_gains": dict(zip(matches, gains.tolist())),
                        "mean_flow_delta_l2": float(np.mean([x["flow_delta_l2"] for _, x in values]))})
    return out


def paired_control_summary(rows, sites):
    comparisons = []
    for site in sites:
        matches = sorted({row["match_id"] for row in rows})
        for control, intervention in MATCHED_CONTROLS.items():
            per_seed = []
            for row in rows:
                values = {entry["condition"]: entry for entry in row["conditions"] if entry["site_index"] == site}
                per_seed.append((row["match_id"], values[intervention]["standardized_error_gain"]-values[control]["standardized_error_gain"]))
            gains = np.asarray([np.mean([gain for match, gain in per_seed if match == mid]) for mid in matches])
            draws = np.random.default_rng(20260907).integers(0, len(matches), (500, len(matches)))
            comparisons.append({"site": SITES[site], "intervention": intervention, "control": control,
                                "n_matches": len(matches), "mean_error_gain_above_control": float(gains.mean()),
                                "paired_match_bootstrap_ci95": np.quantile(gains[draws].mean(1), [.025, .975]).tolist(),
                                "positive_means": "intervention reduces donor-target standardized error more than its paired control",
                                "match_differences": dict(zip(matches, gains.tolist()))})
    return comparisons


def execute(args):
    registration, reg_hash, models = verify_registration(args)
    pilot_path, lock_path = args.output_dir / "pilot.json", args.output_dir / "discovery_selection.json"
    if args.phase != "pilot":
        pilot = read(pilot_path)
        require(pilot["status"] == "passed" and pilot["registration_sha256"] == reg_hash, "Matching reserved causal pilot must pass")
        require(sha256(Path(pilot["pair"]["path"])) == pilot["pair"]["sha256"], "Pilot artifact report changed")
    sites = []
    if args.phase == "evaluate":
        lock = read(lock_path)
        require(lock["status"] == "frozen_discovery_site_selection" and lock["registration_sha256"] == reg_hash
                and lock["n_matches"] == 31 and lock["n_pairs_seeds"] == 62, "All discovery rankings must complete before selection")
        for entry in lock["pairs"]:
            require(sha256(Path(entry["path"])) == entry["sha256"], "Discovery ranking report changed")
        sites = lock["chosen_site_indices"]
        require(len(sites) == 3 and len(set(sites)) == 3, "Exactly three frozen sites required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Set CUDA_VISIBLE_DEVICES to the single authorized GPU")
    model, loading = load_pretrained_four_player(args.assets)
    model = model.to("cuda")
    require(all(parameter.dtype == torch.float32 for parameter in model.parameters()), "Keep released FP32 parameters")
    projection = channel_projection("cuda")
    started = time.monotonic()
    if args.phase == "pilot":
        require(not pilot_path.exists(), "Pilot report already exists")
        record = registration["pilot_record"]
        pair = {"match_id": record["match_id"], "split": "pilot", "recipient": record, "donor": record}
        result, arrays = run_pair(model, pair, SEEDS[0], models, projection, [0, 8, 16], pilot=True)
        require(all(abs(row["attribution"]) == 0 for row in result["scores"]), "Identical pilot donor has nonzero activation difference")
        entry = save_pair(args, reg_hash, result, arrays)
        verify_registration(args)
        write(pilot_path, {"status": "passed", "registration_sha256": reg_hash, "engineering_only": True,
                         "pair": entry, "model_loading": loading, "all17_finite_gradients": True,
                         "input_middle_final_patch_controls": True, "elapsed_seconds": time.monotonic()-started,
                         "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}, exclusive=True)
        print("Reserved causal engineering pilot passed; no discovery ranking executed.", flush=True)
        return
    role = "discovery" if args.phase == "discover" else "selection"
    complete_path = lock_path if args.phase == "discover" else args.output_dir / "selection_results.json"
    require(not complete_path.exists(), "Completed causal stages are immutable")
    pairs = [pair for pair in registration["pairs"] if pair["split"] == role]
    results, entries = [], []
    for pair in pairs:
        for seed in SEEDS:
            cached = args.output_dir / role / f"{pair['match_id']}_seed{seed}.json"
            if cached.exists():
                result = read(cached)
                require(result["status"] == "passed" and result["registration_sha256"] == reg_hash
                        and result["recipient_clip_id"] == pair["recipient"]["clip_id"] and result["donor_clip_id"] == pair["donor"]["clip_id"]
                        and result["seed"] == seed and result["split"] == role and result["match_id"] == pair["match_id"], "Stale causal pair")
                require(sha256(Path(result["artifact_path"])) == result["artifact_sha256"], "Cached causal tensors changed")
                entry = {"path": str(cached.resolve()), "sha256": sha256(cached), "match_id": pair["match_id"], "seed": seed}
            else:
                result, arrays = run_pair(model, pair, seed, models, projection, sites)
                entry = save_pair(args, reg_hash, result, arrays)
            results.append(result)
            entries.append(entry)
            print(json.dumps({"phase": args.phase, "completed": len(results), "expected": len(pairs)*2,
                              "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
    verify_registration(args)
    report = {"status": "frozen_discovery_site_selection" if args.phase == "discover" else "passed_development_internal_causal_tests",
              "registration_sha256": reg_hash, "n_matches": len(pairs), "n_pairs_seeds": len(results), "pairs": entries,
              "model_loading": loading, "elapsed_seconds": time.monotonic()-started,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "scope": registration["scope"], "confirmation_used": False, "new_confirmation_required": True}
    if args.phase == "discover":
        scores = np.asarray([[row["attribution"] for row in result["scores"]] for result in results])
        means = scores.mean(0)
        chosen = sorted(range(17), key=lambda site: (-means[site], site))[:3]
        report.update(chosen_site_indices=chosen, chosen_sites=[SITES[site] for site in chosen],
                      all_site_mean_attributions=dict(zip(SITES, means.tolist())), all_signed_scores_retained=True)
    else:
        report.update(discovery_selection_sha256=sha256(lock_path), chosen_sites=[SITES[site] for site in sites],
                      comparisons=summary(results, sites), uncertainty_note=registration["uncertainty"],
                      paired_control_comparisons=paired_control_summary(results, sites),
                      output_proxy_limit=registration["output_probe_distribution_note"])
    write(complete_path, report, exclusive=True)
    print(f"Completed {args.phase}; scope remains internal developmental output-proxy causality.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("register", "pilot", "discover", "evaluate"))
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "results/development_probes_v3")
    parser.add_argument("--probe-audit", type=Path, default=ROOT / "results/development_probes_v3/probe_audit.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/causal_development_v1")
    parser.add_argument("--artifact-dir", type=Path, default=Path("/data2/ishaangp/mira-interp/causal_development_v1"),
                        help="Private tensor storage; source targets must be excluded from public exports")
    parser.add_argument("--assets", type=Path, default=Path("/data2/ishaangp/mira-interp/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "register":
        register(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
