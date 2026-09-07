"""Descriptor reconstruction lifts; these helpers make no physical-control claim."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from mira_interp.feature_geometry import TopKDictionary


def spatial_descriptor(value, projection):
    """Equal-area 3x4 bins, each containing 3x4 tokens, then channel projection."""
    if value.ndim != 3 or value.shape[:2] != (9, 16) or projection.ndim != 2 or value.shape[-1] != projection.shape[0]:
        raise ValueError("Expected [9,16,C] tile and [C,K] projection")
    if not torch.isfinite(value).all() or not torch.isfinite(projection).all():
        raise ValueError("Descriptor inputs must be finite")
    with torch.autocast(value.device.type, enabled=False):
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        return (value.to(dtype).reshape(3, 3, 4, 4, -1).mean((1, 3)) @ projection.to(dtype)).reshape(1, -1)


def spatial_lift(delta, projection):
    """Minimum-norm right inverse for orthonormal Q; NOT a gradient pullback.

    Each of a bin's 12 tokens gets Q @ delta_bin. There is no 1/12 factor:
    averaging identical offsets already returns the desired bin displacement.
    A gradient pullback, in contrast, distributes a derivative across 12 tokens.
    """
    if projection.ndim != 2 or delta.shape not in {(12 * projection.shape[1],), (1, 12 * projection.shape[1])}:
        raise ValueError("Descriptor delta must contain 12 projected bins")
    if not torch.isfinite(delta).all() or not torch.isfinite(projection).all():
        raise ValueError("Lift inputs must be finite")
    with torch.autocast(delta.device.type, enabled=False):
        dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
        bins = delta.to(dtype).reshape(3, 4, -1) @ projection.to(dtype).T
        return bins.repeat_interleave(3, 0).repeat_interleave(4, 1)


def reconstruct_tile(native, reconstructed_descriptor, projection):
    """Retain the native descriptor nullspace and add the reconstructed offset."""
    original = spatial_descriptor(native, projection)
    target = reconstructed_descriptor.to(device=native.device, dtype=original.dtype)
    if target.shape != original.shape:
        raise ValueError("Reconstruction must match the native descriptor shape")
    return native.to(original.dtype) + spatial_lift(target - original, projection)


def load_dictionary(path: Path, *, dimension=1536, width=3072):
    """Load the exact trained state and scalar normalizer, without refitting."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if set(saved) != {"state_dict", "normalizer", "report"}:
        raise ValueError("Unexpected dictionary checkpoint schema")
    report, normalizer = saved["report"], saved["normalizer"]
    if report["input_dimension"] != dimension or report["latent_scalar_width"] != width:
        raise ValueError("Dictionary dimensions differ from the registered descriptor")
    if report["selection_used_for_updates"] is not False or set(normalizer) != {"mean", "scale"}:
        raise ValueError("A frozen discovery-trained dictionary normalizer is required")
    mean, scale = normalizer["mean"], normalizer["scale"]
    if mean.dtype != torch.float64 or scale.dtype != torch.float64 or mean.shape != (dimension,) or scale.shape != mean.shape:
        raise ValueError("Normalizer must be the saved FP64 vectors")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not torch.all(scale == scale[0]) or scale[0] <= 0:
        raise ValueError("Normalizer must use one finite positive global RMS")
    digest = hashlib.sha256(mean.numpy().tobytes() + scale.numpy().tobytes()).hexdigest()
    if digest != report["normalizer_sha256"]:
        raise ValueError("Normalizer hash differs from training")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all()
           for value in saved["state_dict"].values()):
        raise ValueError("Saved dictionary state must be finite FP32 before loading")
    model = TopKDictionary(dimension, width=width, active=report["active_scalar_budget"],
                          kind=report["kind"], group_size=report["group_size"])
    model.load_state_dict(saved["state_dict"], strict=True)
    if any(parameter.dtype != torch.float32 or not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Dictionary weights must be finite FP32")
    model.eval().requires_grad_(False)
    return model, mean, scale, report


@torch.no_grad()
def reconstruct_descriptor(bundle, descriptor):
    """Use training's FP64 normalization -> FP32 dictionary -> raw descriptor."""
    model, mean, scale, _ = bundle
    descriptor = descriptor.detach().cpu().double()
    if descriptor.ndim != 2 or descriptor.shape[1] != len(mean) or not torch.isfinite(descriptor).all():
        raise ValueError("Invalid descriptor matrix")
    normalized = ((descriptor - mean) / scale).float()
    reconstruction, code = model(normalized)
    raw = reconstruction.double() * scale + mean
    if not torch.isfinite(raw).all() or not torch.isfinite(code).all():
        raise ValueError("Nonfinite sparse reconstruction")
    return raw.float(), code
