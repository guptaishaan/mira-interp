#!/usr/bin/env python3
"""Prepare and execute frozen one-shot steering using published model.inference.

The reserved pilot benchmarks five rollouts before the full budget is approved.
This script saves generated video and intervention measurements, not physical
success claims. An independent frozen video evaluator consumes its manifests.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np
import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mira/src"), str(ROOT / "scripts")]
import causal_development as causal
from analyze_development import require, write
from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionTensors
from mira.world_model.config import WorldModelInferenceConfig
from mira.world_model.schedule import build_inference_schedule
from mira_interp.causal_development import tensor_hash
from mira_interp.development_capture import channel_projection
from mira_interp.model_loading import load_pretrained_four_player, sha256
from mira_interp.probes import RidgeModel
from mira_interp.array_storage import write_npz
from mira_interp.rollout_steering import (DOSES, PATHS, HEIGHT_SUPPORT, FirstGeneratedEdit,
    conditions, hidden_video, path_delta)
from mira_interp.sparse_causal_fidelity import spatial_descriptor, spatial_lift

PROBE_NAME = "spatial/block_15_output/absolute30/position"
SEEDS = causal.SEEDS
CODE = causal.CODE + ["src/mira_interp/array_storage.py", "scripts/rollout_steering_v2.py", "src/mira_interp/rollout_steering.py",
    "src/mira_interp/sparse_causal_fidelity.py", "src/mira_interp/feature_geometry.py",
    "external/mira/src/mira/world_model/multi_wrapper_world_model.py",
    "external/mira/src/mira/world_model/latent_world_model.py", "external/mira/src/mira/world_model/schedule.py"]


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(Path(path))}


def checked(entry):
    require(sha256(Path(entry["path"])) == entry["sha256"], f"Bound file changed: {entry['path']}")
    return read(entry["path"])


def code_hashes():
    return {name: sha256(ROOT / name) for name in CODE}


def schedule_values():
    return build_inference_schedule(10, torch.device("cpu"), "linear_quadratic").tolist()


def load_maps(path, export_path):
    exported = read(export_path)
    report = read(path.parent / "report.json")
    require(exported["status"] == "passed" and exported["maps_sha256"] == sha256(path)
            and exported["report_sha256"] == sha256(path.parent / "report.json"), "Frozen geometry export changed")
    require(report["status"] == "passed_development_feature_analysis" and report["descriptor_identity"] == "spatial/block_15_output"
            and report["geometry"]["variable"] == "height" and report["confirmation_rows_loaded"] is False,
            "Expected development-only block15 height geometry")
    with np.load(path, allow_pickle=False) as archive:
        maps = {name: archive[name] for name in archive.files}
    require(str(maps["variable"]) == "height" and np.isfinite(maps["physical_mean"])
            and np.isfinite(maps["physical_scale"]) and maps["physical_scale"] > 0, "Invalid map coordinates")
    require(float(maps["physical_mean"]) == report["geometry"]["physical_mean_fit_only"]
            and float(maps["physical_scale"]) == report["geometry"]["physical_scale_fit_only"], "Saved physical-coordinate normalization differs")
    for kind, width in (("affine", 1), ("quadratic", 2)):
        arrays = [maps[f"{kind}__{key}"] for key in ("x_mean", "x_scale", "y_mean", "y_scale", "coefficient")]
        model = RidgeModel(report["geometry"]["models"][kind]["alpha"], *arrays)
        require(model.coefficient.shape == (width, 1536) and all(np.isfinite(a).all() for a in arrays)
                and np.all(model.x_scale > 0) and np.all(model.y_scale > 0), "Invalid geometry coefficients")
        require(model.fingerprint() == exported["forward_models_reproduced"][kind], "Frozen geometry model fingerprint differs")
    return maps


def prerequisites(args):
    manifest, audit = read(args.input_manifest), read(args.input_audit)
    require(manifest["status"] == "passed_quality_qualified_rollout_inputs" and manifest["match_counts"]
            == {"pilot": 1, "selection": 10, "confirmation": 22} and not manifest["errors"], "Qualified33 inputs required")
    require(audit["status"] == "passed_rollout_input_audit" and audit["input_report_sha256"] == sha256(args.input_manifest)
            and audit["registration_sha256"] == manifest["registration_sha256"], "Independent rollout input audit differs")
    records = manifest["records"]
    require(len(records) == len({row["clip_id"] for row in records}) == len({row["match_id"] for row in records}) == 33,
            "Exactly one unique clip per qualified match required")
    audited = {(row["clip_id"], row["match_id"], row["role"], row["extended_sha256"]) for row in audit["records"]}
    require(len(audit["records"]) == 33 and audited == {(row["clip_id"], row["match_id"], row["role"], row["sha256"]) for row in records},
            "Input audit role/clip/array identity differs")
    for row in records:
        require(sha256(Path(row["path"])) == row["sha256"], "Qualified input archive changed")
    models, _, probe_bindings = causal.bind_probes(args)
    maps = load_maps(args.maps, args.geometry_export)
    fidelity = read(args.fidelity_results)
    fidelity_audit = read(args.fidelity_audit)
    require(fidelity["status"] == "passed_development_sparse_causal_fidelity" and fidelity["n_matches"] == 11
            and fidelity["n_pairs_seeds"] == 22 and fidelity["dictionary_count"] == 30, "Completed internal fidelity execution required")
    require(fidelity_audit["status"] == "passed" and fidelity_audit["fidelity_results_sha256"] == sha256(args.fidelity_results),
            "Independent completed fidelity audit required; positive effects are not a gate")
    fidelity_registration_path = args.fidelity_results.parent / "registration.json"
    require(sha256(fidelity_registration_path) == fidelity["registration_sha256"], "Fidelity registration differs")
    feature_audit_binding = read(fidelity_registration_path)["feature_bindings"]["audit"]
    feature_audit = checked(feature_audit_binding)
    require(feature_audit["status"] == "passed" and feature_audit["saved_geometry_physical_metadata_independently_recomputed"] is True,
            "Independent sparse/geometry metadata audit must pass")
    bindings = {name: binding(path) for name, path in {
        "inputs": args.input_manifest, "input_audit": args.input_audit, "maps": args.maps,
        "geometry_export": args.geometry_export, "geometry_report": args.maps.parent / "report.json",
        "fidelity_results": args.fidelity_results, "fidelity_audit": args.fidelity_audit,
        "fidelity_registration": fidelity_registration_path, "feature_audit": Path(feature_audit_binding["path"])}.items()}
    for name,path in {
        "storage_benchmark": ROOT/"results/rollout_storage_benchmark.json",
        "prior_successful_pilot": ROOT/"results/rollout_steering_v1/pilot.json",
        "prior_successful_pilot_audit": ROOT/"results/rollout_steering_v1/pilot_audit.json"}.items():
        bindings[name]=binding(path)
    benchmark=checked(bindings["storage_benchmark"])
    prior=checked(bindings["prior_successful_pilot"])
    prior_audit=checked(bindings["prior_successful_pilot_audit"])
    require(benchmark["status"]=="passed_storage_benchmark" and benchmark["all_reloaded_arrays_bitwise_equal"]
            and benchmark["original_artifacts_and_manifest_unchanged"], "Lossless storage benchmark must pass")
    require(prior["status"]=="passed_rollout_steering_pilot" and prior_audit["status"]=="passed_rollout_generation_audit"
            and prior_audit["manifest_sha256"]==bindings["prior_successful_pilot"]["sha256"]
            and benchmark["pilot_manifest_sha256"]==bindings["prior_successful_pilot"]["sha256"], "Prior engineering pilot differs")
    return records, models[PROBE_NAME], maps, bindings, probe_bindings


def register(args):
    records, probe, _, bindings, probe_bindings = prerequisites(args)
    schedule = schedule_values()
    registration = {"status": "registered_before_rollout_steering", "created_utc": datetime.now(timezone.utc).isoformat(),
        "storage": "Lossless NPZ DEFLATE level1; same arrays and model arithmetic as successful v1 pilot",
        "model_api": "Published Alakazam-compatible model.inference; fresh local streaming KV cache each condition",
        "scope": "Frozen candidate generated-video steering; independent video evaluation required",
        "physical_control_established": False, "records": records, "match_counts": {"pilot": 1, "selection": 10, "confirmation": 22},
        "context_frames": 16, "generated_frames": 8, "future_video_placeholder": 0, "alternate_pilot_placeholder": 255,
        "n_diffusion_steps": 10, "schedule_type": "linear_quadratic", "schedule_values_fp32": schedule,
        "noise_level": 0., "intervention_step_index": 8, "intervention_tau": schedule[8],
        "schedule_values_bf16": torch.as_tensor(schedule, dtype=torch.bfloat16).float().tolist(),
        "intervention_tau_bf16": float(torch.tensor(schedule[8], dtype=torch.bfloat16)),
        "sampler_precision": "Published sampler retains codec latent dtype. Tau is nominal FP32 schedule cast to that dtype; under BF16 step8 is actually0.52734375. Record actual tau/dtype every call; do not rewrite sampler to FP32.",
        "tau_amendment": "Before any rollout: published10-step default has no0.5 step; preserve original schedule and edit nearest actual tau0.527789294719696. Teacher-forced tau0.5 and generated cached tau~0.528 are different distributions.",
        "site": "block_15_output", "view": 0, "generated_latent_index": 0,
        "persistence": "One edit only; remaining denoising steps and next3 generated latents unedited. tau1 cache initialization/update never edited.",
        "paths": list(PATHS), "doses_uu": list(DOSES), "condition_grid": conditions(), "n_conditions_per_pair": 21,
        "height_support_uu": list(HEIGHT_SUPPORT), "base_height": "Frozen spatial ball-z probe on baseline first generated latent at actual registered tau; no source state used.",
        "clipping": "Clamp base probe height to fixed discovery support, then clamp base+requested dose. Report both clipping flags and effective dose; retain all21 nominal conditions.",
        "path_definition": "Probe minimum-norm feature offset or frozen forward map f(clippedtarget)-f(clippedbase); lift with Q, preserving native descriptor nullspace. Random and wrong-ball-x directions match quadratic raw lift norm.",
        "normalization": "Frozen discovery probe and forward-map normalizers; no new fits or selected paths",
        "actions": "Original24-frame action batch fixed. Upstream windows slice [1:17],[3:19],[5:21],[7:23]; final generated latent action pairs [15:17],[17:19],[19:21],[21:23]. Frame23 action is unused by published alignment.",
        "action_dtype": "Convert a separate keyboard-action copy to int32 for upstream embeddings; source uint8 arrays and all numeric action values remain unchanged.",
        "engineering_amendment": "Initial reserved pilot failed before baseline because ByteTensor indices reached the keyboard embedding. Original registration/code/log/exit are preserved in results/rollout_attempts/01_action_dtype. This dtype-only correction changes no research choices.",
        "seeds": SEEDS, "pilot_seed": SEEDS[0], "random_direction_seed_offset": 1777,
        "pilot_dose_candidates_uu": [300., -300.],
        "pilot_dose_rule": "Use+300 if clipping leaves a positive effective dose; otherwise use-300. Reserved engineering coverage only; main dose grid unchanged.",
        "pilot": "Five inferences: original unhooked, hooked baseline, alternate255 placeholders, zero lift, quadratic active edit. Engineering dose is+300 if clipped effective dose is positive, otherwise-300. Require nonzero actual edit. Compare exact latents/raw decoder outputs and measure runtime/storage.",
        "full_budget": {"selection_rollouts": 420, "confirmation_rollouts": 924,
            "workers": 2, "partition": "registered phase match order, indices worker_index::2",
            "agent_resource_review_after_pilot_required": True, "minimum_free_bytes": 25*1024**3},
        "output": "Private NPZ frames FP32[4,24,3,288,512] clipped0..1 with clipping counts and raw output hashes, latents FP32[4,12,9,16,32]. Public export must use generated last8 frames only.",
        "context_control": "Report decoded first16 pixel equality and latent context equality for each condition; evaluator windows contain decoded context.",
        "interpretation": "Geometry comparisons did not establish a nonlinear manifold. Physical response, monotonicity, nuisance changes and persistence require independent frozen evaluator analysis; code completion is not scientific success.",
        "confirmation_scope": "These matches were used for prior observational decoding confirmation, but not for these generated intervention outcomes. No rollout-based choice may precede their frozen evaluation.",
        "bindings": bindings, "probe_bindings": probe_bindings, "probe_sha256": probe.fingerprint(), "code_sha256": code_hashes()}
    write(args.output_dir / "registration.json", registration, exclusive=True)
    print("Registered rollout candidates; no model inference executed.", flush=True)


def verify(args, *, initial=True):
    registration = read(args.output_dir / "registration.json")
    require(registration["code_sha256"] == code_hashes() and registration["schedule_values_fp32"] == schedule_values(), "Rollout implementation or upstream schedule changed")
    for name, entry in registration["bindings"].items():
        require(sha256(Path(entry["path"])) == entry["sha256"], f"Rollout binding changed: {name}")
    for entry in registration["probe_bindings"].values():
        require(sha256(Path(entry["path"])) == entry["sha256"], "Frozen rollout probe binding changed")
    if not initial:
        return registration
    records, probe, maps, bindings, probe_bindings = prerequisites(args)
    require(records == registration["records"] and bindings == registration["bindings"]
            and probe_bindings == registration["probe_bindings"] and probe.fingerprint() == registration["probe_sha256"], "Registered rollout inputs differ")
    return registration, sha256(args.output_dir / "registration.json"), probe, maps


def source_arrays(record):
    require(sha256(Path(record["path"])) == record["sha256"], "Registered source changed")
    with np.load(record["path"], allow_pickle=False) as data:
        # Do not load source simulator targets: baseline height is model-probed.
        frames, actions = data["frames"], data["actions"]
    require(frames.shape == (4, 24, 3, 288, 512) and frames.dtype == np.float32
            and actions.shape == (4, 24, 9) and np.isfinite(actions).all(), "Invalid rollout source video/action dimensions")
    return torch.from_numpy(frames), torch.from_numpy(actions)


def infer(model, frames, action_array, seed, schedule, projection, *, delta=None, placeholder=0., hooked=True):
    video = hidden_video(frames, placeholder=placeholder)
    actions = ActionTensors(model.config.actions, batch_size=4)
    actions.key_presses = action_array.to(torch.int32).clone()
    require(torch.equal(actions.key_presses.float(), action_array.float()), "Action dtype conversion changed values")
    actions.mouse_movements = torch.zeros(4, 24, 2)
    batch = VideoActionBatch(video, actions).to("cuda")
    config = WorldModelInferenceConfig(n_diffusion_steps=10, schedule_type="linear_quadratic", noise_level=0.)
    torch.manual_seed(seed)
    started = time.monotonic()
    trace = FirstGeneratedEdit(model.world_model, schedule=schedule, descriptor_delta=delta, projection=projection) if hooked else None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if trace is None:
            output = model.inference(batch, config, progress_bar=False)
        else:
            with trace:
                output = model.inference(batch, config, progress_bar=False)
            trace.validate()
    raw = rearrange(output.output_video.float(), "b t c (p h) w -> (b p) t c h w", p=4)
    latents = output.z_t.float()
    require(raw.shape == (4, 24, 3, 288, 512) and latents.shape == (4, 12, 9, 16, 32)
            and torch.isfinite(raw).all() and torch.isfinite(latents).all(), "Incomplete/nonfinite published rollout")
    result = {"frames": raw.clamp(0, 1).cpu().numpy(), "latents": latents.cpu().numpy(),
              "raw_decoded_sha256": tensor_hash(raw), "raw_decoded_context_sha256": tensor_hash(raw[:, :16]),
              "latent_sha256": tensor_hash(latents), "raw_decoded_min": float(raw.min()), "raw_decoded_max": float(raw.max()),
              "action_batch_sha256": tensor_hash(action_array),
              "decoded_clipped_values": int(((raw < 0) | (raw > 1)).sum()), "elapsed_seconds": time.monotonic()-started,
              "trace": trace}
    return result


def exact_rollout(a, b, description):
    require(a["raw_decoded_sha256"] == b["raw_decoded_sha256"] and a["latent_sha256"] == b["latent_sha256"]
            and np.array_equal(a["frames"], b["frames"]) and np.array_equal(a["latents"], b["latents"]), description)


def record_id(source, seed, name, dose):
    return f"{source['clip_id']}_seed{seed}_{name}_dose{int(dose):+d}"


def save_condition(args, reg_hash, source, seed, name, dose, generated, baseline, base_height, probe, projection, height):
    identifier = record_id(source, seed, name, dose)
    baseline_id = record_id(source, seed, "baseline", 0)
    path = args.artifact_dir / source["role"] / f"{identifier}.npz"
    metadata_path = args.output_dir / source["role"] / f"{identifier}.json"
    require(not path.exists() and not metadata_path.exists(), "Existing generated condition must be audited/resumed explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = generated["trace"]
    before = spatial_descriptor(trace.native_tile, projection).cpu().numpy()
    after = spatial_descriptor(trace.edited_tile, projection).cpu().numpy()
    arrays = {"frames": generated["frames"], "latents": generated["latents"],
              "native_edit_descriptor": before, "edited_descriptor": after,
              "native_edit_tile": trace.native_tile.cpu().numpy(), "edited_tile": trace.edited_tile.cpu().numpy()}
    with path.open("xb") as stream:
        write_npz(stream, arrays, compresslevel=1)
    metrics = {key: generated[key] for key in ("raw_decoded_sha256", "raw_decoded_context_sha256", "latent_sha256",
                                              "raw_decoded_min", "raw_decoded_max", "decoded_clipped_values", "elapsed_seconds", "action_batch_sha256")}
    row = {"record_id": identifier, "match_id": source["match_id"], "clip_id": source["clip_id"], "split": source["role"],
           "seed": seed, "intervention_type": name, "dose": dose, "generated_artifact_path": str(path.resolve()),
           "generated_sha256": sha256(path), "baseline_record_id": baseline_id, "registration_sha256": reg_hash,
           "source_sha256": source["sha256"], "status": "passed_generation", "generated_bytes": path.stat().st_size,
           "physical_control_established": False, "source_targets_loaded": False, "height": height,
           "base_height_unclipped_from_baseline": base_height, "actual_site_position_proxy_delta": (probe.predict(after)-probe.predict(before))[0].tolist(),
           "actual_edit_l2": float(torch.linalg.vector_norm(trace.edited_tile-trace.native_tile)),
           "descriptor_delta_l2": float(np.linalg.norm(after-before)), "world_model_calls": trace.trace,
           "requested_descriptor_delta_l2": 0. if trace.details is None else trace.details["requested_descriptor_delta_l2"],
           "requested_raw_lift_l2": 0. if trace.delta is None else float(torch.linalg.vector_norm(spatial_lift(trace.delta, projection))),
           "descriptor_rounding_error_l2": float(np.linalg.norm(after-before-(0. if trace.delta is None else trace.delta.cpu().numpy()))),
           "decoded_context_bitwise_equal_to_baseline": generated["raw_decoded_context_sha256"] == baseline["raw_decoded_context_sha256"],
           "decoded_context_max_abs_difference": float(np.max(np.abs(generated["frames"][:, :16]-baseline["frames"][:, :16]))),
           "latent_context_bitwise_equal_to_baseline": np.array_equal(generated["latents"][:, :8], baseline["latents"][:, :8]),
           "future_latent_l2_by_time": np.sqrt(np.sum(np.square(generated["latents"][:, 8:].astype(float)-baseline["latents"][:, 8:]), axis=(0, 2, 3, 4))).tolist(),
           "public_export_rule": "Only frames[:,16:] and latents[:,8:]; never republish full observed context", **metrics}
    require(row["latent_context_bitwise_equal_to_baseline"], "Generated edit changed fixed context latents")
    require([call["action_features_sha256"] for call in trace.trace] ==
            [call["action_features_sha256"] for call in baseline["trace"].trace], "Action conditioning differs from baseline")
    write(metadata_path, row, exclusive=True)
    row["record_report"] = binding(metadata_path)
    return row


def load_cached(args, source, seed, name, dose, reg_hash):
    path = args.output_dir / source["role"] / f"{record_id(source, seed, name, dose)}.json"
    if not path.exists():
        return None
    row = read(path)
    require(row["status"] == "passed_generation" and row["registration_sha256"] == reg_hash
            and row["source_sha256"] == source["sha256"] and row["record_id"] == record_id(source, seed, name, dose)
            and sha256(Path(row["generated_artifact_path"])) == row["generated_sha256"], "Cached generated condition changed")
    row["record_report"] = binding(path)
    return row


def execute(args):
    registration, reg_hash, probe, maps = verify(args)
    if args.phase == "pilot":
        require(args.worker_index is None, "Pilot is a single unsharded reserved experiment")
    else:
        require(args.worker_index in range(2), "Select one of the two registered match workers")
    suffix = "" if args.phase == "pilot" else f"_worker{args.worker_index}"
    output_path = args.output_dir / f"{args.phase}{suffix}.json"
    require(not output_path.exists(), "Completed rollout phases are immutable")
    if args.phase != "pilot":
        pilot = read(args.output_dir / "pilot.json")
        require(pilot["status"] == "passed_rollout_steering_pilot" and pilot["registration_sha256"] == reg_hash, "Matching five-rollout pilot must pass")
    if args.phase == "confirmation":
        selected = read(args.output_dir / "selection.json")
        audit = read(args.selection_audit)
        require(selected["status"] == "passed_rollout_generation" and selected["registration_sha256"] == reg_hash
                and audit["status"] == "passed_rollout_generation_audit" and audit["manifest_sha256"] == sha256(args.output_dir / "selection.json"),
                "Selection generation and its independent completion audit must pass first")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose only one currently authorized GPU")
    model, loading = load_pretrained_four_player(args.assets)
    model.to("cuda").eval().requires_grad_(False)
    require(all(parameter.dtype == torch.float32 for parameter in model.parameters()), "Keep original FP32 checkpoint parameters")
    model.set_inference_context(16)
    projection = channel_projection("cuda")
    schedule = registration["schedule_values_fp32"]
    sources = [row for row in registration["records"] if row["role"] == args.phase]
    require(len(sources) == {"pilot": 1, "selection": 10, "confirmation": 22}[args.phase], "Wrong phase cohort")
    if args.phase != "pilot":
        sources = sources[args.worker_index::2]
    rows, started = [], time.monotonic()
    pilot_controls = None
    for source in sources:
        frames, actions = source_arrays(source)
        action_hash = tensor_hash(actions)
        for seed in (SEEDS[:1] if args.phase == "pilot" else SEEDS):
            # Baseline is recomputed when resuming to recover the exact generated
            # intervention tile; cached outputs must then replay bitwise.
            baseline = infer(model, frames, actions, seed, schedule, projection)
            descriptor = spatial_descriptor(baseline["trace"].native_tile, projection).cpu().numpy()
            base_height = float(probe.predict(descriptor)[0, 2])
            if args.phase == "pilot":
                original = infer(model, frames, actions, seed, schedule, projection, hooked=False)
                exact_rollout(original, baseline, "No-op capture changed published rollout")
                alternate = infer(model, frames, actions, seed, schedule, projection, placeholder=255.)
                exact_rollout(baseline, alternate, "Hidden future placeholders changed rollout")
                zero = infer(model, frames, actions, seed, schedule, projection, delta=torch.zeros(1, 1536, device="cuda"))
                exact_rollout(baseline, zero, "Zero edit changed rollout")
                pilot_controls = {"unhooked_vs_capture_bitwise_equal": True, "future_placeholder_bitwise_invariance": True,
                    "zero_edit_bitwise_equal": True, "future_actions_unchanged": tensor_hash(actions) == action_hash,
                    "persisted_control_hashes": {name: {key:value[key] for key in
                        ("raw_decoded_sha256", "raw_decoded_context_sha256", "latent_sha256", "action_batch_sha256")}
                        for name,value in (("unhooked",original),("hooked",baseline),("alternate255",alternate),("zero_edit",zero))},
                    "four_control_inference_seconds": [value["elapsed_seconds"] for value in (original, baseline, alternate, zero)]}
                del original, alternate, zero
            pilot_dose = 300. if min(max(base_height, HEIGHT_SUPPORT[0]), HEIGHT_SUPPORT[1]) < HEIGHT_SUPPORT[1] else -300.
            grid = [("baseline", 0.), ("quadratic_forward", pilot_dose)] if args.phase == "pilot" else conditions()
            for name, dose in grid:
                delta, height = path_delta(name, base_height, dose, probe, maps, projection, seed=seed+1777)
                cached = load_cached(args, source, seed, name, dose, reg_hash)
                if cached is not None:
                    require(cached["base_height_unclipped_from_baseline"] == base_height and cached["height"] == height,
                            "Resumed steering coordinates differ from baseline")
                    if name == "baseline":
                        require(cached["raw_decoded_sha256"] == baseline["raw_decoded_sha256"] and cached["latent_sha256"] == baseline["latent_sha256"], "Resumed baseline differs")
                    row = cached
                else:
                    require(shutil.disk_usage(args.artifact_dir.parent).free >= 25*1024**3,
                            "Stop generation before consuming the25GiB disk reserve")
                    generated = baseline if name == "baseline" else infer(model, frames, actions, seed, schedule, projection, delta=delta)
                    require(torch.equal(generated["trace"].native_tile, baseline["trace"].native_tile), "Pre-edit trajectory differs despite paired inputs/seed")
                    row = save_condition(args, reg_hash, source, seed, name, dose, generated, baseline, base_height, probe, projection, height)
                    if name != "baseline":
                        del generated
                rows.append(row)
                if args.phase == "pilot" and name != "baseline":
                    require(height["effective_dose"] != 0 and row["actual_edit_l2"] > 0,
                            "Engineering pilot must exercise a nonzero supported edit")
                require(tensor_hash(actions) == action_hash, "Future/source actions changed")
                print(json.dumps({"phase": args.phase, "records_complete": len(rows), "record_id": row["record_id"],
                                  "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
            del baseline
    verify(args, initial=False)
    for row in rows:
        checked(row["record_report"])
        require(sha256(Path(row["generated_artifact_path"])) == row["generated_sha256"], "Generated archive changed during phase")
    result = {"status": "passed_rollout_steering_pilot" if args.phase == "pilot" else "passed_rollout_generation",
        "registration_sha256": reg_hash, "records": rows, "n_records": len(rows), "match_count": len(sources),
        "seed_count": 1 if args.phase == "pilot" else 2, "model_loading": loading,
        "worker_index": args.worker_index, "worker_count": 1 if args.phase == "pilot" else 2,
        "physical_control_established": False, "independent_video_evaluation_completed": False,
        "source_targets_loaded": False, "generated_artifact_bytes": sum(row["generated_bytes"] for row in rows),
        "elapsed_seconds": time.monotonic()-started, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "all_context_pixels_unchanged": all(row["decoded_context_bitwise_equal_to_baseline"] for row in rows)}
    if args.phase == "pilot":
        result.update(controls=pilot_controls, full_inferences=5, engineering_only=True,
                      full_budget_requires_agent_resource_review=True, mean_saved_condition_bytes=result["generated_artifact_bytes"]/len(rows))
    write(output_path, result, exclusive=True)
    print(f"Completed rollout {args.phase}; independent generated-video evaluation remains separate.", flush=True)


def aggregate(args):
    require(args.phase in {"selection", "confirmation"} and args.worker_index is None,
            "Aggregate only completed two-worker research phases")
    registration, reg_hash, _, _ = verify(args)
    destination = args.output_dir / f"{args.phase}.json"
    require(not destination.exists(), "Completed aggregate manifests are immutable")
    sources = [row for row in registration["records"] if row["role"] == args.phase]
    records, workers = [], []
    for index in range(2):
        path = args.output_dir / f"{args.phase}_worker{index}.json"
        report = read(path)
        require(report["status"] == "passed_rollout_generation" and report["registration_sha256"] == reg_hash
                and report["worker_index"] == index and report["worker_count"] == 2,
                "A registered rollout worker is incomplete")
        expected = {record_id(source, seed, name, dose) for source in sources[index::2]
                    for seed in SEEDS for name, dose in conditions()}
        require(len(report["records"]) == len(expected) and {row["record_id"] for row in report["records"]} == expected,
                "Worker's complete frozen condition grid differs")
        for row in report["records"]:
            saved = checked(row["record_report"])
            require(saved == {key: value for key, value in row.items() if key != "record_report"}
                    and sha256(Path(row["generated_artifact_path"])) == row["generated_sha256"],
                    "Worker condition metadata/artifact changed")
        records.extend(report["records"])
        workers.append({**binding(path), "elapsed_seconds": report["elapsed_seconds"],
                        "peak_gpu_allocated_bytes": report["peak_gpu_allocated_bytes"]})
    require(len(records) == len({row["record_id"] for row in records}) == len(sources)*2*21,
            "Duplicate or missing rollout conditions")
    records.sort(key=lambda row: row["record_id"])
    result = {"status": "passed_rollout_generation", "registration_sha256": reg_hash,
              "records": records, "n_records": len(records), "match_count": len(sources), "seed_count": 2,
              "workers": workers, "generated_artifact_bytes": sum(row["generated_bytes"] for row in records),
              "all_context_pixels_unchanged": all(row["decoded_context_bitwise_equal_to_baseline"] for row in records),
              "physical_control_established": False, "independent_video_evaluation_completed": False,
              "source_targets_loaded": False, "independent_completion_audit_required": True}
    verify(args, initial=False)
    write(destination, result, exclusive=True)
    print(f"Aggregated{len(records)} frozen {args.phase} conditions; independent audit required.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("register", "pilot", "selection", "confirmation"))
    parser.add_argument("--worker-index", type=int, choices=(0, 1))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--input-manifest", type=Path, default=ROOT / "results/rollout_inputs_v1.json")
    parser.add_argument("--input-audit", type=Path, default=ROOT / "results/rollout_inputs_v1_audit.json")
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "results/development_probes_v3")
    parser.add_argument("--probe-audit", type=Path, default=ROOT / "results/development_probes_v3/probe_audit.json")
    parser.add_argument("--maps", type=Path, default=ROOT / "results/geometry_development_v1/height/geometry_maps.npz")
    parser.add_argument("--geometry-export", type=Path, default=ROOT / "results/geometry_development_v1/height/export.json")
    parser.add_argument("--fidelity-results", type=Path, default=ROOT / "results/sparse_causal_fidelity_v1/fidelity_results.json")
    parser.add_argument("--fidelity-audit", type=Path, default=ROOT / "results/sparse_causal_fidelity_v1/fidelity_audit.json")
    parser.add_argument("--selection-audit", type=Path, default=ROOT / "results/rollout_steering_v2/selection_audit.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rollout_steering_v2")
    parser.add_argument("--artifact-dir", type=Path, default=Path("/data2/ishaangp/mira-interp/rollout_steering_v2"))
    parser.add_argument("--assets", type=Path, default=Path("/data2/ishaangp/mira-interp/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(args)
    elif args.phase == "register":
        register(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
