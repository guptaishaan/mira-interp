#!/usr/bin/env python3
"""Bounded, strict pretrained MIRA Mini readiness check; no physical-state claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MIRA_REV = "3d739ec2d31daf83559d33eb01727cea48fe90f7"
DINO_REV = "6876159a11b4df116f30f667f8c9888617df0751"
MODEL_REV = "d58ee2f9bca27289554c1e652943dc0e539e8971"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(directory):
    revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "-C", str(directory), "diff", "--quiet", "HEAD", "--"]).returncode:
        raise RuntimeError(f"Tracked source changes at {directory}; a commit hash alone is insufficient")
    untracked = subprocess.check_output(["git", "-C", str(directory), "ls-files", "--others", "--exclude-standard", "-z"]).decode().split("\0")
    source_files = [name for name in untracked if name and "__pycache__" not in Path(name).parts and Path(name).suffix in {".py", ".pyi", ".so"}]
    if source_files:
        raise RuntimeError(f"Untracked source files at {directory}: {source_files}")
    return revision


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--diffusion-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--output", type=Path, default=ROOT / "results/pretrained_smoke.json")
    ap.add_argument("--activation-dir", type=Path, required=True)
    ap.add_argument("--figure", type=Path, default=ROOT / "figures/pretrained_smoke.png")
    ap.add_argument("--load-only", action="store_true", help="CPU strict loading audit; no inference")
    args = ap.parse_args()
    if args.frames < 4 or args.frames % 2 or args.frames > 8:
        ap.error("This bounded readiness script accepts 4, 6, or 8 frames")
    if not 1 <= args.diffusion_steps <= 10:
        ap.error("Use 1 through 10 diffusion steps")
    args.assets = args.assets.resolve()
    started = time.monotonic()
    result = {
        "status": "running", "evidence_level": "pretrained_inference_readiness_only",
        "model": "Alakazam MIRA Mini 4P 1B independent reproduction",
        "model_repo": "alakazamworld/mira-mini-4p", "model_revision": MODEL_REV,
        "seed": args.seed, "requested_device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_validation": False, "stage": "input_audit",
    }

    def stage(name, **fields):
        result.update(fields, stage=name, elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(args.output, result)
        print(f"[{result['elapsed_seconds']:.1f}s] {name}", flush=True)

    stage("input_audit")
    try:
        import numpy as np
        import torch
        from omegaconf import OmegaConf

        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external/mira/src")]
        from mira.codec.codec_model import VideoCodec, REMOVED_CONFIG_FIELDS
        from mira.codec.config import VideoCodecConfig
        from mira.data.batch import VideoActionBatch
        from mira.ml.config_loading import drop_removed_fields, strip_hydra_targets
        from mira.world_model.actions_config import ActionTensors
        from mira.world_model.config import WorldModelInferenceConfig
        from mira.world_model.latent_world_model import _config_dict_from_yaml
        from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModelConfig
        from mira_interp.instrumentation import CaptureMetadata, ResidualTrace
        from mira_interp.pretrained import AlakazamFourPlayerWorldModel

        for folder, expected in [(ROOT / "external/mira", MIRA_REV), (ROOT / "external/dinov3", DINO_REV)]:
            if git_revision(folder) != expected:
                raise RuntimeError(f"Source revision mismatch at {folder}")
        result["source_revisions"] = {"mira": MIRA_REV, "dinov3": DINO_REV}
        result["experiment_code_sha256"] = {name: sha256(ROOT / name) for name in ["scripts/smoke_pretrained.py", "src/mira_interp/pretrained.py", "src/mira_interp/instrumentation.py"]}
        result["compatibility_adapter"] = {"name": "AlakazamFourPlayerWorldModel", "published_runtime": "alakazam-mira-mini==0.1.11", "wheel_sha256": "0acea9fc285c44537785e2987a42181695500317371e6cc652cefcd38e982cfe", "difference": "per-player 2048-to-2048 projection then mean instead of upstream concatenate then 8192-to-2048 projection"}
        result["torch_version"] = torch.__version__
        manifest = json.loads((ROOT / "results/model_access_audit.json").read_text())
        entry = next(x for x in manifest["third_party_pretrained_alternatives"] if x["repo_id"] == result["model_repo"])
        if entry["revision"] != MODEL_REV:
            raise ValueError("Audit manifest revision differs from the smoke pin")
        needed = ["checkpoint-90000/checkpoint.pth", "codec/checkpoint-125000/checkpoint.pth", "context/default.npz"]
        file_audit = []
        for name in needed:
            spec = next(x for x in entry["files"] if x["name"] == name)
            path = args.assets / name
            if path.stat().st_size != spec["size_bytes"]:
                raise ValueError(f"File size mismatch: {name}")
            actual = sha256(path)
            if actual != spec["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {name}")
            file_audit.append({"name": name, "bytes": path.stat().st_size, "sha256": actual})
        result["verified_assets"] = file_audit
        cfg = OmegaConf.load(args.assets / "world_model_config.yaml")
        codec_cfg = OmegaConf.load(args.assets / "codec/codec_config.yaml")
        result["original_config_sha256"] = {
            name: sha256(args.assets / name) for name in ["world_model_config.yaml", "codec/codec_config.yaml"]
        }
        raw_codec = drop_removed_fields(strip_hydra_targets(OmegaConf.to_container(codec_cfg.model.architecture.config, resolve=True)), REMOVED_CONFIG_FIELDS)
        raw_codec["encoder"]["compile_dino"] = False
        raw_codec["decoder"]["activation_checkpointing"] = False
        raw_wm = _config_dict_from_yaml(cfg.model.architecture.config)
        raw_wm["wm_config"]["codec_checkpoint"] = str(args.assets / needed[1])
        raw_wm["wm_config"]["activation_checkpointing"] = False
        result["runtime_overrides"] = {"codec_checkpoint": "local verified codec path", "compile_dino": False, "activation_checkpointing": False, "inference_context_frames": args.frames - 2}
        stage("load_codec_weights_safely")
        codec_checkpoint = torch.load(args.assets / needed[1], map_location="cpu", weights_only=True, mmap=True)
        local_hub_load = torch.hub.load

        def pinned_dino_factory(**kwargs):
            if kwargs["repo_or_dir"] != "facebookresearch/dinov3" or kwargs.get("pretrained") is not False:
                raise ValueError("Only source-only DINO construction from the pinned local checkout is allowed")
            return local_hub_load(str(ROOT / "external/dinov3"), kwargs["model"], source="local", pretrained=False)

        with patch.object(torch.hub, "load", side_effect=pinned_dino_factory):
            codec = VideoCodec(VideoCodecConfig.model_validate(raw_codec), require_dino_weights=False)
        torch.nn.Module.load_state_dict(codec, codec_checkpoint["state_dict"], strict=True, assign=True)
        codec.info_from_checkpoint = {k: v for k, v in codec_checkpoint.items() if k != "state_dict"}
        result["codec_strict_load"] = {"passed": True, "state_tensors": len(codec_checkpoint["state_dict"])}
        del codec_checkpoint
        stage("construct_and_strict_load_world_model")
        # Supply the already safely loaded codec. Avoid upstream's torch.load and its
        # multiplayer warm-start override, which tolerates some random parameters.
        with patch.object(VideoCodec, "load_from_checkpoint", return_value=codec):
            model = AlakazamFourPlayerWorldModel(MultiWrapperWorldModelConfig.model_validate(raw_wm))
        checkpoint = torch.load(args.assets / needed[0], map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint["state_dict"]
        expected_keys = model.state_dict()
        missing = sorted(set(expected_keys) - set(state))
        unexpected = sorted(set(state) - set(expected_keys))
        shape_mismatch = [k for k in set(expected_keys) & set(state) if expected_keys[k].shape != state[k].shape]
        result["world_model_strict_load"] = {"passed": False, "missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatch": shape_mismatch, "state_tensors": len(state)}
        if missing or unexpected or shape_mismatch:
            stage("incompatible_checkpoint")
            raise RuntimeError("Checkpoint is incompatible with the pinned architecture; no random fallback is permitted")
        torch.nn.Module.load_state_dict(model, state, strict=True, assign=True)
        result["world_model_strict_load"]["passed"] = True
        result["parameter_count_including_codec"] = sum(p.numel() for p in model.parameters())
        result["codec_parameter_count"] = sum(p.numel() for p in codec.parameters())
        del expected_keys, state, checkpoint
        model.eval().requires_grad_(False)
        if args.load_only:
            result["status"] = "passed_load_only"
            stage("complete_load_only")
            return
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("Bounded inference requires an explicitly selected available CUDA device")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        result["gpu_name"] = torch.cuda.get_device_name(device)
        model.to(device=device, dtype=torch.bfloat16)
        model.set_inference_context(args.frames - 2)
        with np.load(args.assets / "context/default.npz", allow_pickle=False) as context:
            if set(context.files) != {"frames", "actions"}:
                raise ValueError("Unexpected example context fields")
            frames = context["frames"][:, :args.frames].copy()
            actions_array = context["actions"][:, :args.frames].copy()
        if frames.shape != (4, args.frames, 3, 288, 512) or actions_array.shape != (4, args.frames, 9):
            raise ValueError("Unexpected four-player context shape")
        result["input"] = {"frames_shape": list(frames.shape), "actions_shape": list(actions_array.shape), "physics_labels": False, "match_id": None, "context_frames": args.frames - 2, "generated_frames": 2}
        actions = ActionTensors(model.config.actions, batch_size=4)
        actions.key_presses = torch.from_numpy(actions_array)
        actions.mouse_movements = torch.zeros(4, args.frames, 2)
        batch = VideoActionBatch(torch.from_numpy(frames), actions).to(device)
        stage("encode_public_context")
        from einops import rearrange
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prepared = batch.clone()
            codec.preprocess_batch(prepared)
            z_flat = model.single_world_model.encode_video(prepared)
            z = rearrange(z_flat, "(b p) t h w c -> b t (p h) w c", p=4)
            swm = model.single_world_model
            off = swm.action_temporal_downsampling - 1
            n_actions = (z.shape[1] - 1) * swm.action_temporal_downsampling
            a_flat = swm.action_encoder(prepared.actions.slice_time(off, off + n_actions))
            a = model._combine_player_actions(a_flat)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            noise = torch.randn(z.shape, generator=generator, device=device, dtype=z.dtype)
            tau = torch.full((1, z.shape[1], 1, 1, 1), 0.5, device=device)
            z_t = tau * z + (1 - tau) * noise
            clean_past = torch.cat([swm.bos[None, None], z[:, :-1]], dim=1)
            metadata = CaptureMetadata(sample_ids=("public_default_context_unknown_match",), condition="teacher_forced_noisy_observed_target_no_intervention", noise_seed=args.seed, noise_id=hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest(), rollout_step=0, diffusion_step=0, tau=tuple(tau.flatten().cpu().tolist()), tau_shape=tuple(tau.shape), latent_frame_indices=tuple(range(z.shape[1])), cache_mode="fresh")
            stage("unhooked_baseline_forward")
            baseline = model.world_model(z_t, a, tau, clean_past=clean_past, activation_checkpointing=False)
            stage("capture_all_residual_layers")
            block_zero_inputs = []
            first_handle = model.world_model.transformer[0].register_forward_pre_hook(
                lambda _module, inputs: block_zero_inputs.append(inputs[0].detach().to("cpu", copy=True))
            )
            try:
                with ResidualTrace(model, metadata) as trace:
                    prediction = model.world_model(z_t, a, tau, clean_past=clean_past, activation_checkpointing=False)
            finally:
                first_handle.remove()
            repeated = model.world_model(z_t, a, tau, clean_past=clean_past, activation_checkpointing=False)
            result["forward_controls"] = {
                "noop_hooks_bitwise_equal": torch.equal(baseline, prediction),
                "repeated_forward_bitwise_equal": torch.equal(baseline, repeated),
                "noop_max_abs_error": (baseline.float() - prediction.float()).abs().max().item(),
                "repeat_max_abs_error": (baseline.float() - repeated.float()).abs().max().item(),
            }
            if not all(result["forward_controls"][key] for key in ["noop_hooks_bitwise_equal", "repeated_forward_bitwise_equal"]):
                raise ValueError("No-op hooks or repeated forward changed the trained model output")
            if len(block_zero_inputs) != 1 or not torch.isfinite(block_zero_inputs[0]).all():
                raise ValueError("Expected one finite block-zero input capture")
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite pretrained transformer prediction")
            args.activation_dir.mkdir(parents=True, exist_ok=True)
            first_input_path = args.activation_dir / "block_00_input.pt"
            torch.save(block_zero_inputs[0], first_input_path)
            result["block_zero_input"] = {"shape": list(block_zero_inputs[0].shape), "activation_file": str(first_input_path), "sha256": sha256(first_input_path), "finite": True}
            rows = []
            for layer, activation in sorted(trace.activations.items()):
                value = activation.float()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite layer {layer}")
                path = args.activation_dir / f"layer_{layer:02d}.pt"
                torch.save(activation, path)
                rows.append({"layer": layer, "shape": list(value.shape), "mean": value.mean().item(), "std": value.std().item(), "rms": value.square().mean().sqrt().item(), "finite": True, "activation_file": str(path), "sha256": sha256(path)})
            result["capture"] = trace.manifest()
            result["capture"]["layers"] = rows
            result["capture"]["prediction_shape"] = list(prediction.shape)
            result["capture"]["all_layers_finite"] = True
            stage("sample_one_future_latent")
            generation_batch = batch.clone()
            generation_batch.video[:, -2:] = 0
            inference_config = WorldModelInferenceConfig(n_diffusion_steps=args.diffusion_steps, schedule_type="linear", noise_level=0.0)
            torch.manual_seed(args.seed)
            output = model.inference(generation_batch, inference_config, progress_bar=False)
            if not torch.isfinite(output.output_video).all() or not torch.isfinite(output.z_t).all():
                raise ValueError("Nonfinite generated video or latent")
            stage("verify_future_placeholder_invariance")
            alternate_batch = batch.clone()
            alternate_batch.video[:, -2:] = 255
            torch.manual_seed(args.seed)
            alternate = model.inference(alternate_batch, inference_config, progress_bar=False)
            result["generation_controls"] = {"primary_future_placeholder": "all zero uint8", "alternate_future_placeholder": "all255 uint8", "actions_identical": True, "seed_identical": True, "generated_latents_bitwise_equal": torch.equal(output.z_t[:, -1:], alternate.z_t[:, -1:]), "generated_video_bitwise_equal": torch.equal(output.output_video[:, -2:], alternate.output_video[:, -2:]), "latent_max_abs_difference": (output.z_t[:, -1:].float() - alternate.z_t[:, -1:].float()).abs().max().item(), "video_max_abs_difference": (output.output_video[:, -2:].float() - alternate.output_video[:, -2:].float()).abs().max().item()}
            if not all(result["generation_controls"][key] for key in ["generated_latents_bitwise_equal", "generated_video_bitwise_equal"]):
                raise ValueError("Future placeholder change altered generated trajectories; generation gate failed")
            result["sample"] = {"output_shape": list(output.output_video.shape), "latent_shape": list(output.z_t.shape), "finite": True, "diffusion_steps": args.diffusion_steps, "schedule": "linear", "noise_level": 0.0, "value_min": output.output_video.min().item(), "value_max": output.output_video.max().item()}
            generated_video = rearrange(output.output_video[:, -2:], "b t c (p h) w -> (b p) t c h w", p=4)
            generated_frames_path = args.activation_dir / "generated_frames.npz"
            generated_latents_path = args.activation_dir / "generated_latents.pt"
            np.savez_compressed(
                generated_frames_path,
                frames=generated_video.float().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy(),
                relative_to_last_context_seconds=np.array([0.05, 0.10]),
                source_frame_indices=np.arange(args.frames - 2, args.frames),
            )
            torch.save(output.z_t[:, -1:].detach().cpu(), generated_latents_path)
            result["sample"]["saved_generated_frames"] = {"path": str(generated_frames_path), "sha256": sha256(generated_frames_path), "shape": list(generated_video.shape), "dtype": "uint8", "axes": ["player_view", "time", "channel", "height", "width"], "relative_to_last_context_seconds": [0.05, 0.10], "conversion": "clamp to [0,1], multiply by255, round to uint8 for display"}
            result["sample"]["saved_generated_latents"] = {"path": str(generated_latents_path), "sha256": sha256(generated_latents_path), "shape": list(output.z_t[:, -1:].shape), "axes": ["player_view", "latent_time", "latent_height", "latent_width", "channel"], "normalization": "world-model normalized codec coordinates", "dtype": str(output.z_t.dtype)}
            generated = output.output_video[0, -1].float().cpu().clamp(0, 1).numpy()
        stage("write_readiness_artifacts")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 2, figsize=(10, 12))
        for player in range(4):
            axes[player, 0].imshow(frames[player, args.frames - 3].transpose(1, 2, 0))
            axes[player, 1].imshow(generated[:, player * 288:(player + 1) * 288].transpose(1, 2, 0))
            axes[player, 0].set_ylabel(f"View {player}")
            for ax in axes[player]:
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_title("Last observed context frame")
        axes[0, 1].set_title("Generated final frame (+0.10 s)")
        fig.suptitle("MIRA Mini 4P pretrained readiness: one public context\nShort context; no physical-state validation", fontsize=13)
        fig.text(0.5, 0.015, "Independent Alakazam weights; Rocket League content: Psyonix / Epic Games. CC BY-NC-SA 4.0.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=120)
        plt.close(fig)
        csv_path = args.output.with_name("pretrained_layer_stats.csv")
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["layer", "mean", "std", "rms", "finite"])
            writer.writeheader()
            writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
        result["artifacts"] = {"figure": str(args.figure), "layer_statistics": str(csv_path), "activation_directory": str(args.activation_dir)}
        torch.cuda.synchronize(device)
        result["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["status"] = "passed"
        stage("complete")
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        stage(result["stage"])
        raise


if __name__ == "__main__":
    main()
