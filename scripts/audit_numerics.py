#!/usr/bin/env python3
"""Reserved-pilot comparison of capture precision against the released runtime.

This is a numerical engineering audit, not probe fitting or causal physics
validation. It uses the already-prepared uint8 pilot, leaving resize rounding
unchanged so model/interpolation precision is isolated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mira/src")]
from mira_interp.model_loading import load_pretrained_four_player, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_variant(model, frames, raw_actions, seed, *, fp32_mix):
    import numpy as np
    import torch
    from einops import rearrange
    from mira.data.batch import VideoActionBatch
    from mira.world_model.actions_config import ActionTensors

    swm, device = model.single_world_model, model.device
    actions = ActionTensors(model.config.actions, batch_size=4)
    actions.key_presses = torch.from_numpy(raw_actions.astype(np.int32, copy=False))
    actions.mouse_movements = torch.zeros(4, 16, 2)
    batch = VideoActionBatch(torch.from_numpy(frames), actions).to(device)
    pooled, handles = {}, []

    def pool(value):
        return value.reshape(1, 8, 4, 9, 16, 2048).float().mean((3, 4))[0].permute(1, 0, 2).reshape(32, 2048).cpu()

    def input_hook(_module, args):
        if 0 in pooled:
            raise RuntimeError("Repeated block-zero hook")
        pooled[0] = pool(args[0])

    def output_hook(site):
        def hook(_module, _args, output):
            if site in pooled:
                raise RuntimeError("Repeated output hook")
            pooled[site] = pool(output[0] if isinstance(output, tuple) else output)
        return hook

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.codec.preprocess_batch(batch)
        z_flat = swm.encode_video(batch)
        if z_flat.shape != (4, 8, 9, 16, 32) or z_flat.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 codec output for exact paired-noise comparison")
        z = rearrange(z_flat, "p t h w c -> 1 t (p h) w c")
        off = swm.action_temporal_downsampling - 1
        count = (z.shape[1] - 1) * swm.action_temporal_downsampling
        combined = model._combine_player_actions(swm.action_encoder(batch.actions.slice_time(off, off + count)))
        noise = torch.randn(z.shape, device=device, dtype=z.dtype, generator=torch.Generator(device=device).manual_seed(seed))
        tau = torch.full((1, 8, 1, 1, 1), .5, device=device, dtype=torch.float32)
        z_t = tau * z + (1 - tau) * noise if fp32_mix else .5 * z + .5 * noise
        clean_past = torch.cat([swm.bos[None, None], z[:, :-1]], dim=1)

        def forward():
            return model.world_model(z_t, combined, tau, clean_past=clean_past, activation_checkpointing=False)

        baseline = forward()
        handles.append(model.world_model.transformer[0].register_forward_pre_hook(input_hook))
        handles.extend(block.register_forward_hook(output_hook(i + 1)) for i, block in enumerate(model.world_model.transformer))
        try:
            captured = forward()
        finally:
            for handle in handles:
                handle.remove()
        repeated = forward()
        if set(pooled) != set(range(17)):
            raise RuntimeError("Incomplete 17-site capture")
        controls = {"noop_bitwise_equal": torch.equal(baseline, captured), "repeat_bitwise_equal": torch.equal(baseline, repeated)}
        if not all(controls.values()):
            raise ValueError("Numerics pilot no-op/repeat control failed")
        arrays = {"codec": z_flat.float().cpu().numpy(), "interpolant": z_t.float().cpu().numpy(), "prediction": captured.float().cpu().numpy(), "means": torch.stack([pooled[i] for i in range(17)], dim=1).numpy()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Nonfinite numerical audit output")
        details = {"controls": controls, "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}), "codec_dtype": str(z_flat.dtype), "interpolant_dtype": str(z_t.dtype), "clean_past_dtype": str(clean_past.dtype), "prediction_dtype": str(captured.dtype), "noise_sha256": hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest(), "fp32_tau_mixing": fp32_mix}
    torch.cuda.synchronize(device)
    return arrays, details


def errors(reference, candidate):
    import numpy as np
    a, b = reference.astype(np.float64, copy=False), candidate.astype(np.float64, copy=False)
    delta = b - a
    denominator = max(float(np.linalg.norm(a.ravel())), 1e-30)
    return {"relative_l2": float(np.linalg.norm(delta.ravel()) / denominator), "rmse": float(np.sqrt(np.mean(delta ** 2))), "max_abs_difference": float(np.max(np.abs(delta))), "bitwise_equal": bool(np.array_equal(reference, candidate))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, default=ROOT / "results/capture_pilot_v2.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/qualified_pilot_clip_manifest.json")
    parser.add_argument("--data-audit", type=Path, default=ROOT / "results/qualified_pilot_data_audit.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = {"status": "running", "scope": "reserved actual pilot numerical parity only", "resize_quantization_removed": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "numpy_madvise_hugepage": os.environ.get("NUMPY_MADVISE_HUGEPAGE"), "variants": {}}

    def stage(name):
        report.update(stage=name, elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(args.report, report)
        print(f"[{report['elapsed_seconds']:.1f}s] {name}", flush=True)

    stage("prerequisites")
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        if args.output.exists():
            raise ValueError("Refusing to overwrite a numerical audit archive")
        pilot, manifest, audit = [json.loads(p.read_text()) for p in (args.pilot_report, args.manifest, args.data_audit)]
        if pilot.get("status") != "passed" or pilot.get("pilot") is not True or audit.get("status") != "passed" or manifest.get("status") != "passed":
            raise ValueError("Reserved pilot gates did not pass")
        if pilot["manifest_sha256"] != sha256(args.manifest) or audit["clip_manifest_sha256"] != sha256(args.manifest) or pilot["data_audit_sha256"] != sha256(args.data_audit):
            raise ValueError("Reserved pilot data audit hash mismatch")
        record = sorted(manifest["records"], key=lambda row: row["clip_id"])[0]
        old_clip = pilot["clips"][0]
        if record["role"] != "pilot" or record["clip_id"] != old_clip["clip_id"]:
            raise ValueError("The audit must use the same reserved pilot clip")
        for name, expected in pilot["code_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError("Original capture/helper code changed")
        source = Path(record["artifact_path"])
        if sha256(source) != record["sha256"] or sha256(Path(old_clip["output_path"])) != old_clip["output_sha256"]:
            raise ValueError("Pilot source or saved capture hash mismatch")
        with np.load(source, allow_pickle=False) as archive:
            frames, actions = archive["frames"], archive["actions"]
        if frames.shape != (4, 16, 3, 288, 512) or frames.dtype != np.uint8 or actions.shape != (4, 16, 9):
            raise ValueError("Unexpected reserved-pilot model input shape")
        report.update(clip_id=record["clip_id"], match_id=record["match_id"], role="pilot", input_sha256=record["sha256"], pilot_report_sha256=sha256(args.pilot_report), manifest_sha256=sha256(args.manifest), data_audit_sha256=sha256(args.data_audit), source_protocol_sha256=pilot["protocol_sha256"], script_sha256=sha256(Path(__file__)), source_capture_code_sha256=pilot["code_sha256"], seed=old_clip["noise_seed"])
        data = {}
        model = None
        for name, whole_bf16, fp32_mix in [("original_full_bf16", True, False), ("fp32_model_legacy_mix", False, False), ("publisher_precision", False, True)]:
            if name != "publisher_precision":
                if model is not None:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                loading = {}
                torch.manual_seed(20260906)
                model, _ = load_pretrained_four_player(args.assets, report=loading, progress=stage)
                if {p.dtype for p in model.parameters()} != {torch.float32}:
                    raise ValueError("Expected original FP32 checkpoint parameters before precision conversion")
                report.setdefault("load_provenance", {})[name] = loading
                model.to(device="cuda:0", dtype=torch.bfloat16 if whole_bf16 else torch.float32)
                model.eval().requires_grad_(False)
            torch.cuda.reset_peak_memory_stats()
            stage(name)
            values, details = run_variant(model, frames, actions, old_clip["noise_seed"], fp32_mix=fp32_mix)
            details["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if details["noise_sha256"] != old_clip["noise_sha256"]:
                raise ValueError("Paired noise differs from the original reserved pilot")
            report["variants"][name], data[name] = details, values
            if name == "original_full_bf16":
                with np.load(old_clip["output_path"], allow_pickle=False) as saved:
                    replay_equal = np.array_equal(values["means"].astype(np.float16), saved["X"])
                report["original_pooled_capture_replay_bitwise_equal"] = bool(replay_equal)
                if not replay_equal:
                    raise ValueError("Original BF16 path does not exactly replay the saved pilot means")
            stage(name + "_complete")
        comparisons = {}
        for reference, candidate in [("original_full_bf16", "fp32_model_legacy_mix"), ("fp32_model_legacy_mix", "publisher_precision"), ("original_full_bf16", "publisher_precision")]:
            name = reference + "__to__" + candidate
            comparisons[name] = {key: errors(data[reference][key], data[candidate][key]) for key in ["codec", "interpolant", "prediction", "means"]}
            comparisons[name]["sites"] = [{"site": site, **errors(data[reference]["means"][:, i], data[candidate]["means"][:, i])} for i, site in enumerate(old_clip["site_names"])]
        report["comparisons"] = comparisons
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        arrays = {name + "__" + key: value for name, values in data.items() for key, value in values.items()}
        arrays["sites"] = np.asarray(old_clip["site_names"])
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(temporary, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                raise ValueError("Numerics archive readback mismatch")
        if sha256(Path(__file__)) != report["script_sha256"] or any(sha256(ROOT / name) != expected for name, expected in pilot["code_sha256"].items()):
            raise ValueError("Audit or capture/helper code changed during comparison")
        temporary.replace(args.output)
        report.update(status="passed", output_path=str(args.output.resolve()), output_sha256=sha256(args.output), output_bytes=args.output.stat().st_size, gpu_name=torch.cuda.get_device_name(), physical_claim="none; differences alone do not establish cause of weak probes")
        stage("complete")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        stage(report.get("stage", "failed"))
        raise


if __name__ == "__main__":
    main()
