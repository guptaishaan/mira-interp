#!/usr/bin/env python3
"""Capture actual Rocket Science observations for registered reconstruction probes.

All model inputs are frames and actions. Physical labels are joined afterward.
Rows are view-major then latent-time; this is not future-state prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mira/src")]
from mira_interp.model_loading import load_pretrained_four_player, sha256

APPROVED_PROTOCOL_HASHES = {
    1: "2437f915ceb14a10c33a6b003b7273ee6ba135e8495d4b15b03ff72f0edbd6a3",
    2: "e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def clip_seed(clip_id):
    return int.from_bytes(hashlib.sha256(f"mira-observation-v1\0{clip_id}".encode()).digest()[:8], "big") % (2**63 - 1)


def pool_views(residual, players=4):
    """[1,T,P*H,W,C] -> [P*T,C], accumulating spatial means in float32."""
    if residual.ndim != 5 or residual.shape[0] != 1 or residual.shape[2] % players:
        raise ValueError("Expected one four-view tiled residual")
    _, time_steps, tiled_height, width, channels = residual.shape
    value = residual.reshape(1, time_steps, players, tiled_height // players, width, channels)
    return value.float().mean(dim=(3, 4))[0].permute(1, 0, 2).reshape(players * time_steps, channels)


def capture_features(model, frames, actions_array, *, seed, clip_id, pilot_dir=None):
    """Run one teacher-forced call without accepting or consulting physical labels."""
    import numpy as np
    import torch
    from einops import rearrange
    from mira.data.batch import VideoActionBatch
    from mira.world_model.actions_config import ActionTensors
    from mira_interp.instrumentation import CaptureMetadata, ResidualTrace

    if frames.shape != (4, 16, 3, 288, 512) or frames.dtype != np.uint8:
        raise ValueError("Expected uint8 four-view16-frame video at288x512")
    if actions_array.shape != (4, 16, 9) or not np.isin(actions_array, [0, 1]).all():
        raise ValueError("Expected four source-rate binary keyboard streams")
    device = model.device
    swm = model.single_world_model
    if (model.temporal_downsampling, model.spatial_downsampling) != (2, 32):
        raise ValueError("Registered label alignment requires codec TD2/SD32")
    actions = ActionTensors(model.config.actions, batch_size=4)
    actions.key_presses = torch.from_numpy(actions_array.astype(np.int32, copy=False))
    actions.mouse_movements = torch.zeros(4, 16, 2)
    batch = VideoActionBatch(torch.from_numpy(frames), actions).to(device)
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.codec.preprocess_batch(batch)
        z_flat = swm.encode_video(batch)
        if z_flat.shape != (4, 8, 9, 16, 32):
            raise ValueError(f"Unexpected per-view codec shape: {z_flat.shape}")
        z = rearrange(z_flat, "p t h w c -> 1 t (p h) w c")
        off = swm.action_temporal_downsampling - 1
        count = (z.shape[1] - 1) * swm.action_temporal_downsampling
        action_features = model._combine_player_actions(swm.action_encoder(batch.actions.slice_time(off, off + count)))
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(z.shape, generator=generator, device=device, dtype=z.dtype)
        tau = torch.full((1, 8, 1, 1, 1), 0.5, device=device)
        z_t = 0.5 * z + 0.5 * noise
        clean_past = torch.cat([swm.bos[None, None], z[:, :-1]], dim=1)
        noise_hash = hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest()
        metadata = CaptureMetadata(sample_ids=(clip_id,), condition="teacher_forced_noisy_observed_target_no_intervention", noise_seed=seed, noise_id=noise_hash, rollout_step=0, diffusion_step=0, tau=(0.5,) * 8, tau_shape=(1, 8, 1, 1, 1), latent_frame_indices=tuple(range(8)), cache_mode="fresh")
        pooled, full_inputs, handles = {}, [], []

        def capture_input(_module, inputs):
            if 0 in pooled:
                raise RuntimeError("Repeated block-zero execution in one capture")
            pooled[0] = pool_views(inputs[0]).cpu()
            if pilot_dir is not None:
                full_inputs.append(inputs[0].detach().to("cpu", copy=True))

        def capture_output(site):
            def hook(_module, _inputs, output):
                if site in pooled:
                    raise RuntimeError("Repeated residual execution in one capture")
                value = output[0] if isinstance(output, tuple) else output
                pooled[site] = pool_views(value).cpu()
            return hook

        def forward():
            return model.world_model(z_t, action_features, tau, clean_past=clean_past, activation_checkpointing=False)

        controls = {}
        if pilot_dir is not None:
            baseline = forward()
        handles.append(model.world_model.transformer[0].register_forward_pre_hook(capture_input))
        for i, block in enumerate(model.world_model.transformer):
            handles.append(block.register_forward_hook(capture_output(i + 1)))
        try:
            if pilot_dir is not None:
                with ResidualTrace(model, metadata) as full_trace:
                    prediction = forward()
            else:
                prediction = forward()
        finally:
            for handle in handles:
                handle.remove()
        if set(pooled) != set(range(17)):
            raise RuntimeError("Incomplete17-site capture")
        X = torch.stack([pooled[i] for i in range(17)], dim=1)
        if X.shape != (32, 17, 2048) or not torch.isfinite(X).all() or not torch.isfinite(prediction).all():
            raise ValueError("Incorrect or nonfinite pooled capture")
        if pilot_dir is not None:
            repeated = forward()
            controls.update(noop_bitwise_equal=torch.equal(baseline, prediction), repeat_bitwise_equal=torch.equal(baseline, repeated), noop_max_abs_difference=(baseline.float() - prediction.float()).abs().max().item())
            if not controls["noop_bitwise_equal"] or not controls["repeat_bitwise_equal"]:
                raise ValueError("Pilot no-op or repeat control failed")
            pilot_dir.mkdir(parents=True, exist_ok=True)
            full = {0: full_inputs[0], **{i + 1: value for i, value in full_trace.activations.items()}}
            reference_checks = []
            for site, value in full.items():
                # Independent explicit per-view slice, not the reshape used by pool_views.
                reference = torch.cat([value[0, :, p * 9:(p + 1) * 9].float().mean(dim=(1, 2)) for p in range(4)], dim=0)
                error = (reference - pooled[site]).abs().max().item()
                if not torch.allclose(reference, pooled[site], atol=1e-5, rtol=1e-5):
                    raise ValueError(f"Pilot per-view pooling disagreement at site{site}")
                path = pilot_dir / f"site_{site:02d}.pt"
                torch.save(value, path)
                reference_checks.append({"site": site, "max_abs_difference": error, "path": str(path), "sha256": sha256(path), "shape": list(value.shape)})
            controls["full_tensor_pooling"] = reference_checks
            controls["full_tensor_pooling_passed"] = True
        codec_X = z_flat.float().mean(dim=(2, 3)).reshape(32, 32).cpu()
        rgb = torch.from_numpy(frames[:, 1::2].copy()).reshape(32, 3, 288, 512).float() / 255
        rgb_X = torch.nn.functional.adaptive_avg_pool2d(rgb, (8, 8)).flatten(1)
        X_half = X.to(torch.float16)
        quant_error = (X - X_half.float()).abs().max().item()
        arrays = {"X": X_half.numpy(), "codec_X": codec_X.to(torch.float16).numpy(), "RGB_X": rgb_X.to(torch.float16).numpy()}
    torch.cuda.synchronize(device)
    return arrays, {"noise_seed": seed, "noise_sha256": noise_hash, "tau": 0.5, "metadata": metadata.__dict__, "controls": controls, "capture_seconds": time.monotonic() - start, "storage_max_abs_quantization_error": quant_error, "prediction_shape": list(prediction.shape)}


def audit_output(path, expected_hash=None):
    import numpy as np
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"Existing capture hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as value:
        expected = {"X": (32, 17, 2048), "y": (32, 30), "codec_X": (32, 32), "RGB_X": (32, 192), "view_index": (32,), "latent_frame_index": (32,), "source_frame_index": (32,), "timestamps": (32,)}
        for key, shape in expected.items():
            if value[key].shape != shape or not np.isfinite(value[key]).all():
                raise ValueError(f"Invalid saved capture field{key}: {path}")
        if not np.array_equal(value["view_index"], np.repeat(np.arange(4), 8)) or not np.array_equal(value["latent_frame_index"], np.tile(np.arange(8), 4)):
            raise ValueError("Capture row order changed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-audit", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, default=ROOT / "data/split_manifest.json")
    ap.add_argument("--protocol", type=Path, default=ROOT / "configs/observational_probe_v1.json")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-report", type=Path)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        ap.error("Invalid worker assignment")
    if args.pilot and args.workers != 1:
        ap.error("Pilot must run alone")
    started = time.monotonic()
    report = {"status": "running", "stage": "prerequisite_audit", "pilot": args.pilot, "evidence_level": "observational_teacher_forced_reconstruction_decoding", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "worker_index": args.worker_index, "workers": args.workers, "clips": []}

    def stage(name):
        report.update(stage=name, elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(args.report, report)
        print(f"[{report['elapsed_seconds']:.1f}s] {name} ({len(report['clips'])} clips)", flush=True)

    stage("prerequisite_audit")
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        protocol = json.loads(args.protocol.read_text())
        protocol_hash = sha256(args.protocol)
        if protocol_hash != APPROVED_PROTOCOL_HASHES.get(protocol.get("version")):
            raise ValueError("Registered protocol hash mismatch")
        data_audit = json.loads(args.data_audit.read_text())
        if data_audit.get("status") != "passed":
            raise ValueError("Data prerequisite audit has not passed")
        if protocol["version"] > 1 and data_audit.get("protocol_sha256") != protocol_hash:
            raise ValueError("Qualified data audit is not tied to this registered protocol")
        manifest_digest = sha256(args.manifest)
        split_digest = sha256(args.split_manifest)
        if data_audit.get("clip_manifest_sha256") != manifest_digest or data_audit.get("split_manifest_sha256") != split_digest:
            raise ValueError("Passed data audit is not bound to these exact input manifests")
        script_hash = sha256(Path(__file__))
        code_hashes = {name: sha256(ROOT / name) for name in ["scripts/capture_observations.py", "src/mira_interp/model_loading.py", "src/mira_interp/pretrained.py", "src/mira_interp/instrumentation.py"]}
        report["code_sha256"] = code_hashes
        report.update(protocol_sha256=protocol_hash, data_audit_sha256=sha256(args.data_audit), manifest_sha256=manifest_digest, split_manifest_sha256=split_digest, script_sha256=script_hash, cohort_sha256=protocol["cohort"]["balanced_manifest_sha256"])
        if not args.pilot:
            if args.pilot_report is None:
                raise ValueError("A passed GPU pilot report is required before research capture")
            pilot = json.loads(args.pilot_report.read_text())
            if pilot.get("status") != "passed" or not pilot.get("pilot") or pilot.get("protocol_sha256") != protocol_hash or pilot.get("script_sha256") != script_hash or pilot.get("code_sha256") != code_hashes:
                raise ValueError("Pilot gate failed or pilot did not run this exact registered capture script")
            report["pilot_report_sha256"] = sha256(args.pilot_report)
        manifest = json.loads(args.manifest.read_text())
        splits = json.loads(args.split_manifest.read_text())
        cohort_digest = hashlib.sha256(json.dumps(splits["matches"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if any(value != report["cohort_sha256"] for value in [cohort_digest, splits["sha256"], manifest.get("cohort_sha256"), data_audit.get("cohort_sha256")]):
            raise ValueError("Frozen cohort hash mismatch")
        if manifest.get("status") != "passed" or manifest.get("split_manifest_sha256") != split_digest:
            raise ValueError("Clip manifest has not passed or uses another split manifest")
        if manifest.get("source_revision") != protocol["source_revision"]:
            raise ValueError("Dataset revision differs from registration")
        records = sorted(manifest["records"], key=lambda x: x["clip_id"])
        if len({record["clip_id"] for record in records}) != len(records):
            raise ValueError("Duplicate clip IDs")
        assigned = {row["match_id"]: row for row in splits["matches"]}
        relevant_roles = {"pilot"} if args.pilot else {"discovery", "selection", "confirmation"}
        expected_matches = {mid for mid, row in assigned.items() if row["split"] in relevant_roles}
        actual_counts = {}
        for record in records:
            mid = record["match_id"]
            if mid not in assigned or record["role"] != assigned[mid]["split"] or record["player_ids"] != assigned[mid]["player_ids"]:
                raise ValueError("Clip role/player identity disagrees with frozen match assignment")
            if record["role"] in relevant_roles:
                actual_counts[mid] = actual_counts.get(mid, 0) + 1
        if set(actual_counts) != expected_matches or any(count != 8 for count in actual_counts.values()):
            raise ValueError("Capture requires the full registered match cohort with8clips per match")
        eligible = [x for x in records if (x["role"] == "pilot") == args.pilot]
        if args.pilot:
            eligible = eligible[:1]
        eligible = [x for i, x in enumerate(eligible) if i % args.workers == args.worker_index]
        if not eligible:
            raise ValueError("No eligible clips for this worker")
        report["expected_clips"] = len(eligible)
        args.output_root.mkdir(parents=True, exist_ok=True)
        model = None
        for record in eligible:
            clip_id = record["clip_id"]
            safe_name = hashlib.sha256(clip_id.encode()).hexdigest()[:24]
            output_path = args.output_root / f"{safe_name}.npz"
            meta_path = args.output_root / f"{safe_name}.json"
            if output_path.exists() and meta_path.exists():
                previous = json.loads(meta_path.read_text())
                if previous.get("input_sha256") != record["sha256"] or previous.get("protocol_sha256") != protocol_hash or previous.get("script_sha256") != script_hash or previous.get("code_sha256") != code_hashes:
                    raise ValueError(f"Existing capture provenance differs for{clip_id}")
                audit_output(output_path, previous["output_sha256"])
                report["clips"].append(previous)
                stage("resume_verified_clip")
                continue
            source_path = Path(record["artifact_path"])
            if not source_path.is_absolute():
                source_path = args.manifest.parent / source_path
            if sha256(source_path) != record["sha256"]:
                raise ValueError(f"Prepared clip hash mismatch:{clip_id}")
            if model is None:
                torch.manual_seed(20260906)
                loading = {}
                report["model_loading"] = loading
                model, _ = load_pretrained_four_player(args.assets, report=loading, progress=stage)
                if not args.device.startswith("cuda") or not torch.cuda.is_available():
                    raise ValueError("Observation capture needs an allocated CUDA GPU")
                device = torch.device(args.device)
                torch.cuda.set_device(device)
                torch.cuda.reset_peak_memory_stats(device)
                model.to(device=device, dtype=torch.bfloat16)
                report["gpu_name"] = torch.cuda.get_device_name(device)
            # np.load is lazy. Only model inputs are extracted before model capture.
            with np.load(source_path, allow_pickle=False) as source:
                frames, actions = source["frames"], source["actions"]
            stage(f"capture_{clip_id}")
            arrays, details = capture_features(model, frames, actions, seed=clip_seed(clip_id), clip_id=clip_id, pilot_dir=(args.output_root / "pilot_full") if args.pilot else None)
            # Labels and row metadata are now joined to already-computed activations.
            with np.load(source_path, allow_pickle=False) as source:
                targets = source["targets"]
                if targets.shape != (4, 16, 30):
                    raise ValueError("Expected per-view physical labels; broadcasting is forbidden")
                target_names = source["target_names"].tolist()
                if target_names != protocol["targets"]:
                    raise ValueError("Target field order differs from registered protocol")
                indices = source["source_frame_indices"]
                if indices.shape == (16,):
                    indices = np.broadcast_to(indices, (4, 16))
                if indices.shape != (4, 16) or source["timestamps"].shape != (4, 16):
                    raise ValueError("Source frame/time metadata is not aligned per-view")
                arrays.update(y=targets[:, 1::2].reshape(32, 30), source_frame_index=indices[:, 1::2].reshape(32), timestamps=source["timestamps"][:, 1::2].reshape(32), view_index=np.repeat(np.arange(4), 8), latent_frame_index=np.tile(np.arange(8), 4), player_ids=source["player_ids"], target_names=np.asarray(target_names), sites=np.asarray(protocol["capture"]["sites"]), match_ids=np.asarray([record["match_id"]] * 32), split=np.asarray([record["role"]] * 32), clip_ids=np.asarray([clip_id] * 32))
            tmp = output_path.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            audit_output(tmp)
            tmp.replace(output_path)
            meta = {"clip_id": clip_id, "match_id": record["match_id"], "role": record["role"], "player_ids": record["player_ids"], "rows": 32, "output_path": str(output_path.resolve()), "output_sha256": sha256(output_path), "output_bytes": output_path.stat().st_size, "input_sha256": record["sha256"], "protocol_sha256": protocol_hash, "script_sha256": script_hash, "code_sha256": code_hashes, "model_revision": report["model_loading"]["model_revision"], "verified_checkpoint_assets": report["model_loading"]["verified_assets"], "site_names": protocol["capture"]["sites"], "row_order": "view_major_then_latent_time", "target_alignment": "own_view_source_frame2t+1", **details}
            write_json(meta_path, meta)
            report["clips"].append(meta)
            report["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(model.device)
            stage("clip_complete")
        if any(sha256(ROOT / name) != expected for name, expected in code_hashes.items()):
            raise ValueError("Capture code changed during the run")
        report.update(status="passed", completed_clips=len(report["clips"]), rows=32 * len(report["clips"]))
        stage("complete")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        stage(report["stage"])
        raise


if __name__ == "__main__":
    main()
