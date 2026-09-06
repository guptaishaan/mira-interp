#!/usr/bin/env python3
"""Exercise instrumentation on a tiny real upstream MIRA transformer, random weights.

ENGINEERING ONLY: no trained weights, real clips, physical labels, codec or rollout.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from mira.ml.image_config import ImageConfig
from mira.world_model.actions_config import ActionConfig
from mira.world_model.config import LatentWorldModelConfig
from mira.world_model.diffusion_transformer import DiffusionTransformer
from mira_interp.instrumentation import CaptureMetadata, Intervention, ResidualTrace, attribution_scores


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def max_error(left, right):
    return float((left.detach() - right.detach()).abs().max().cpu())


def caches_equal(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    return len(left) == len(right) and all(caches_equal(a, b) for a, b in zip(left, right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", type=Path, default=Path("results/instrumentation_smoke_cpu.json"))
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    config = LatentWorldModelConfig(
        actions=ActionConfig(valid_keys=["W", "A", "S", "D"], source_fps=20, target_fps=10),
        video=ImageConfig(height=64, width=64, timesteps=4, fps=10),
        hidden_dim=64, n_head=4, n_kv_head=2, n_layers=3,
        n_register_tokens=1, patch_size=2, time_attention_every=2,
        ada_attn_ln=True, attention_gating=True, activation_checkpointing=False,
    )
    model = DiffusionTransformer(config, latent_dim=8, temporal_downsampling=2, spatial_downsampling=16)
    model.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    clean_z = torch.randn((1, 2, 4, 4, 8), generator=generator).to(device)
    noise = torch.randn(clean_z.shape, generator=generator).to(device)
    corrupted_z = clean_z + 0.1 * noise
    actions = torch.randn((1, 2, 64), generator=generator).to(device)
    tau = torch.full((1, 2, 1, 1, 1), 0.5, device=device)
    metadata = CaptureMetadata(
        sample_ids=("random-init-engineering-only-0",), condition="clean-synthetic-latent",
        noise_seed=args.seed + 1, noise_id=digest(noise), rollout_step=0, diffusion_step=0,
        tau=tuple(tau.cpu().flatten().tolist()), tau_shape=tuple(tau.shape), latent_frame_indices=(0, 1),
    )
    corrupted_metadata = replace(metadata, condition="perturbed-synthetic-latent")

    with torch.no_grad():
        reference, reference_cache = model(clean_z, actions, tau, return_kv=True)
        with ResidualTrace(model, metadata) as clean_trace:
            captured, captured_cache = model(clean_z, actions, tau, return_kv=True)
    assert torch.equal(reference, captured), "Capture hooks changed the forward result"
    assert caches_equal(reference_cache, captured_cache), "Capture hooks changed KV output"
    expected_shape = [1, 3, 2, 2, 64]  # two latent frames plus one register slot
    assert all(list(value.shape) == expected_shape for value in clean_trace.activations.values())
    assert len(clean_trace.activations) == config.n_layers

    with ResidualTrace(model, corrupted_metadata, retain_grad=True) as corrupted_trace:
        corrupted = model(corrupted_z, actions, tau)
        objective = corrupted.mean()  # arbitrary linear readout, NOT a physical-state objective
        objective.backward()
    assert all(value.grad is not None for value in corrupted_trace.activations.values())

    with torch.no_grad():
        with ResidualTrace(model, corrupted_metadata, interventions={
            1: Intervention("replace", corrupted_trace.activations[1].detach()),
        }):
            noop = model(corrupted_z, actions, tau)
        assert torch.equal(noop, corrupted)

        # Modify one spatial token of the first actual latent frame, not register slot zero.
        mask = torch.zeros(expected_shape[:-1], dtype=torch.bool)
        mask[0, 1, 0, 0] = True
        with ResidualTrace(model, metadata, interventions={0: Intervention("ablate", mask=mask)}) as ablated_trace:
            ablated = model(clean_z, actions, tau)
        at_site = ablated_trace.activations[0]
        original_site = clean_trace.activations[0]
        expanded_mask = mask.unsqueeze(-1).expand_as(at_site)
        assert torch.equal(at_site[~expanded_mask], original_site[~expanded_mask])
        assert torch.count_nonzero(at_site[expanded_mask]) == 0
        assert max_error(ablated, reference) > 0

        sequence = (Intervention("ablate", mask=mask), Intervention("restore", original_site, mask))
        with ResidualTrace(model, metadata, interventions={0: sequence}):
            restored = model(clean_z, actions, tau)
        assert torch.equal(restored, reference)

        last = config.n_layers - 1
        gradient = corrupted_trace.activations[last].grad
        estimated = attribution_scores(clean_trace.activations[last], corrupted_trace.activations[last], gradient).sum()
        with ResidualTrace(model, corrupted_metadata, interventions={last: Intervention("replace", clean_trace.activations[last])}):
            exact_patch = model(corrupted_z, actions, tau)
        exact = exact_patch.mean() - objective.detach()
        assert torch.allclose(estimated, exact, atol=1e-6, rtol=1e-4)
    assert all(not block._forward_hooks for block in model.transformer)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    repo_root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(repo_root / "external/mira"), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "status": "PASS",
        "evidence_level": "engineering-only; tiny random-init real upstream architecture",
        "scientific_claims": [],
        "not_tested": ["pretrained weights", "video codec", "action encoder", "real data", "physical probes", "physical causal effects", "generated rollouts", "geometry", "SAEs or BSFs"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": commit,
        "device": str(device), "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": args.seed, "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_class": "mira.world_model.diffusion_transformer.DiffusionTransformer",
        "config": config.model_dump(),
        "inputs": {"latent_shape": list(clean_z.shape), "action_embedding_shape": list(actions.shape), "clean_sha256": digest(clean_z), "corrupted_sha256": digest(corrupted_z), "actions_sha256": digest(actions)},
        "manifest": clean_trace.manifest(),
        "checks": {
            "all_layers_captured": len(clean_trace.activations),
            "capture_noop_max_abs_error": max_error(reference, captured),
            "kv_preserved": caches_equal(reference_cache, captured_cache),
            "replacement_noop_max_abs_error": max_error(noop, corrupted),
            "masked_ablation_locality": True,
            "ablation_output_max_abs_effect": max_error(ablated, reference),
            "restoration_max_abs_error": max_error(restored, reference),
            "gradients_retained_all_layers": True,
            "linear_head_attribution_estimate": float(estimated.cpu()),
            "linear_head_exact_patch_effect": float(exact.cpu()),
            "linear_head_attribution_abs_error": float((estimated - exact).abs().cpu()),
            "all_hooks_removed": True,
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(args.output)
    print(json.dumps({"status": result["status"], "scope": result["evidence_level"], "output": str(args.output), "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
