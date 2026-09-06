"""Small, eager-mode hooks for MIRA residuals; no claim of physical feature recovery.

One trace represents one transformer call, at one identified diffusion/rollout step.
All block outputs are captured, including register slots. Codec layers are not sites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CaptureMetadata:
    sample_ids: tuple[str, ...]
    condition: str
    noise_seed: int
    noise_id: str
    rollout_step: int
    diffusion_step: int
    tau: tuple[float, ...]
    tau_shape: tuple[int, ...]
    latent_frame_indices: tuple[int, ...]
    cache_mode: Literal["fresh", "cached"] = "fresh"

    def __post_init__(self):
        if not self.sample_ids or not self.noise_id or not self.condition:
            raise ValueError("Sample IDs, noise identity, and condition are required")
        if self.rollout_step < 0 or self.diffusion_step < 0:
            raise ValueError("Rollout and diffusion indices are zero-based and nonnegative")
        if not self.tau or prod(self.tau_shape) != len(self.tau):
            raise ValueError("tau_shape must describe every recorded tau value")
        if len(self.tau_shape) < 2 or self.tau_shape[:2] != (len(self.sample_ids), len(self.latent_frame_indices)):
            raise ValueError("tau_shape must match the sample and latent-frame metadata")
        if not self.latent_frame_indices or any(index < 0 for index in self.latent_frame_indices):
            raise ValueError("Nonnegative latent-frame indices are required")
        if any(not 0 <= value <= 1 for value in self.tau):
            raise ValueError("Flow-matching tau must lie in [0, 1]")
        if self.cache_mode not in ("fresh", "cached"):
            raise ValueError("Unknown cache mode")


@dataclass(frozen=True)
class ResidualSite:
    layer_index: int
    module_path: str
    temporal_attention: bool
    boundary: str = "block_output_after_mlp"
    axes: tuple[str, ...] = ("batch", "time_with_registers", "height", "width", "channel")


def residual_sites(model: nn.Module) -> list[ResidualSite]:
    """Resolve real MIRA's bare transformer, single-player model, or multiplayer wrapper."""
    for prefix in ("single_world_model.world_model", "world_model", ""):
        module = model
        try:
            for part in prefix.split(".") if prefix else ():
                module = getattr(module, part)
        except AttributeError:
            continue
        blocks = getattr(module, "transformer", None)
        if isinstance(blocks, nn.ModuleList) and len(blocks):
            root = f"{prefix}." if prefix else ""
            return [
                ResidualSite(i, f"{root}transformer.{i}", bool(getattr(block, "time_attention", False)))
                for i, block in enumerate(blocks)
            ]
    raise ValueError("Expected a MIRA model with a nonempty transformer ModuleList")


@dataclass(frozen=True)
class Intervention:
    """Exact donor replacement or zero ablation, optionally restricted to a boolean mask.

    A mask has either the full residual shape or its first four axes (whole tokens).
    Restoration is replacement by the corresponding pre-ablation residual, and is
    named separately to make the intended control visible in experiment records.
    """

    mode: Literal["replace", "ablate", "restore"]
    donor: Tensor | None = None
    mask: Tensor | None = None

    def apply(self, residual: Tensor) -> Tensor:
        if self.mode not in ("replace", "ablate", "restore"):
            raise ValueError(f"Unknown intervention mode: {self.mode}")
        if self.mode == "ablate":
            if self.donor is not None:
                raise ValueError("Zero ablation does not accept a donor")
            replacement = torch.zeros_like(residual)
        else:
            if self.donor is None or self.donor.shape != residual.shape:
                raise ValueError("Donor must have exactly the residual shape; broadcasting is forbidden")
            if not torch.isfinite(self.donor).all():
                raise ValueError("Donor must contain only finite values")
            replacement = self.donor.detach().to(device=residual.device, dtype=residual.dtype)
            if not torch.isfinite(replacement).all():
                raise ValueError("Donor must remain finite after conversion to the residual dtype")
        if self.mask is None:
            return replacement
        if self.mask.dtype != torch.bool:
            raise ValueError("Intervention mask must be boolean")
        mask = self.mask.to(residual.device)
        if mask.shape == residual.shape[:-1]:
            mask = mask.unsqueeze(-1)
        elif mask.shape != residual.shape:
            raise ValueError("Mask must have the full residual shape or the full token-grid shape")
        return torch.where(mask, replacement, residual)


class ResidualTrace:
    """Capture all residual layers for exactly one eager forward; remove hooks on exit.

    `retain_grad=True` keeps graph tensors on their original device for attribution.
    The default copies detached tensors to CPU. Contexts are deliberately single-use
    to reject ambiguous repeated diffusion calls and checkpoint recomputation.
    Interventions at a site are applied in the supplied order, before capture.
    """

    def __init__(
        self,
        model: nn.Module,
        metadata: CaptureMetadata,
        *,
        interventions: dict[int, Intervention | tuple[Intervention, ...]] | None = None,
        retain_grad: bool = False,
    ):
        self.model = model
        self.metadata = metadata
        self.sites = residual_sites(model)
        transformer_path = self.sites[0].module_path.rsplit("transformer.", 1)[0].rstrip(".")
        transformer = model.get_submodule(transformer_path) if transformer_path else model
        self.register_slots = int(getattr(transformer, "n_register_tokens", 0))
        self.interventions = interventions or {}
        unknown = set(self.interventions) - {site.layer_index for site in self.sites}
        if unknown:
            raise ValueError(f"Unknown zero-based residual layer indices: {sorted(unknown)}")
        self.retain_grad = retain_grad
        self.activations: dict[int, Tensor] = {}
        self.input_activation: Tensor | None = None
        self._handles = []
        self._used = False

    def _capture_input(self, module, inputs):
        if self.input_activation is not None:
            raise RuntimeError("A ResidualTrace permits one forward only; disable activation checkpointing")
        residual = inputs[0]
        if not isinstance(residual, Tensor) or residual.ndim != 5:
            raise ValueError("Expected MIRA block-0 input [batch, time, height, width, channel]")
        if self.retain_grad:
            if not residual.requires_grad:
                raise RuntimeError("Gradient capture requires a grad-enabled block-0 input")
            residual.retain_grad()
            self.input_activation = residual
        else:
            self.input_activation = residual.detach().to("cpu", copy=True)

    def _hook(self, site: ResidualSite):
        def capture(module, inputs, output):
            if site.layer_index in self.activations:
                raise RuntimeError("A ResidualTrace permits one forward only; disable activation checkpointing")
            residual = output[0] if isinstance(output, tuple) else output
            if not isinstance(residual, Tensor) or residual.ndim != 5:
                raise ValueError("Expected MIRA residual [batch, time, height, width, channel]")
            if residual.shape[0] != len(self.metadata.sample_ids):
                raise ValueError("Sample metadata does not match the residual batch")
            expected_time = len(self.metadata.latent_frame_indices)
            if self.metadata.cache_mode == "fresh":
                expected_time += self.register_slots
            if residual.shape[1] != expected_time:
                raise ValueError("Latent-frame metadata and register slots do not match residual time")
            patches = self.interventions.get(site.layer_index, ())
            if isinstance(patches, Intervention):
                patches = (patches,)
            for patch in patches:
                residual = patch.apply(residual)
            if self.retain_grad:
                if not residual.requires_grad:
                    raise RuntimeError("Gradient capture requires a grad-enabled residual")
                residual.retain_grad()
                self.activations[site.layer_index] = residual
            else:
                self.activations[site.layer_index] = residual.detach().to("cpu", copy=True)
            # The second tuple item is upstream's temporal KV cache, not a residual.
            return (residual, *output[1:]) if isinstance(output, tuple) else residual

        return capture

    def __enter__(self):
        if self._used:
            raise RuntimeError("ResidualTrace contexts are single-use")
        self._used = True
        try:
            first_block = self.model.get_submodule(self.sites[0].module_path)
            self._handles.append(first_block.register_forward_pre_hook(self._capture_input))
            for site in self.sites:
                module = self.model.get_submodule(site.module_path)
                self._handles.append(module.register_forward_hook(self._hook(site)))
        except BaseException:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self._remove_hooks()
        if exc_type is None and len(self.activations) != len(self.sites):
            raise RuntimeError("Incomplete all-layer capture; not every residual site executed")
        return False

    def manifest(self) -> dict:
        return {
            "metadata": asdict(self.metadata),
            "capture_boundary": "after each complete AdaSTBlock; tuple item zero",
            "indexing": "zero-based; layer 0 is after the first block",
            "register_slots_in_this_call": self.register_slots if self.metadata.cache_mode == "fresh" else 0,
            "cache_policy": "preserve each block's KV output unchanged; this is a residual-only intervention",
            "input_site": {
                "module_path": self.sites[0].module_path,
                "boundary": "block_0_input_after_latent_projection_and_register_insertion",
                "shape": list(self.input_activation.shape) if self.input_activation is not None else None,
            },
            "sites": [
                {
                    **asdict(site),
                    "shape": list(self.activations[site.layer_index].shape)
                    if site.layer_index in self.activations else None,
                }
                for site in self.sites
            ],
        }


def attribution_scores(clean: Tensor, corrupted: Tensor, gradient: Tensor) -> Tensor:
    """Per-sample first-order effect sum((clean-corrupted) * grad_corrupted).

    The objective's sign is the caller's responsibility. This is a local linear
    estimate; exact patching must validate rankings on separate selection data.
    """
    if gradient is None:
        raise ValueError("A retained objective gradient is required")
    if clean.shape != corrupted.shape or gradient.shape != corrupted.shape:
        raise ValueError("Clean, corrupted, and gradient shapes must agree exactly")
    if not all(torch.isfinite(value).all() for value in (clean, corrupted, gradient)):
        raise ValueError("Activations and gradient must contain only finite values")
    if clean.ndim < 2:
        raise ValueError("Expected a batch axis and at least one feature axis")
    delta = clean.detach().to(device=gradient.device, dtype=torch.float32) - corrupted.detach().to(
        device=gradient.device, dtype=torch.float32
    )
    return (delta * gradient.detach().float()).flatten(1).sum(1)


def norm_matched_random_delta(delta: Tensor, *, generator: torch.Generator) -> Tensor:
    """Random-direction control with the same whole-site L2 norm for each sample.

    Match intervention *deltas*, not clean activation norms. A masked intervention
    should pass its selected coordinates separately so the control has the same support.
    """
    if delta.ndim < 2:
        raise ValueError("Expected batch and feature axes")
    random = torch.randn(delta.shape, dtype=torch.float32, device=delta.device, generator=generator)
    target_norm = delta.float().flatten(1).norm(dim=1)
    random_norm = random.flatten(1).norm(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    shape = (-1,) + (1,) * (delta.ndim - 1)
    return (random * (target_norm / random_norm).reshape(shape)).to(delta.dtype)
