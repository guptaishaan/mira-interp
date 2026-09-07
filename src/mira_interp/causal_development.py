"""Development-only residual interventions and differentiable frozen readouts.

These helpers measure an internal output proxy. They do not recover a physical
trajectory or turn a natural donor into a one-variable counterfactual.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib

import numpy as np
import torch

SITES = ["block_0_input"] + [f"block_{i}_output" for i in range(16)]


def tensor_hash(value):
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def tile(value, *, view=0, latent=7):
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[2:4] != (36, 16):
        raise ValueError("Expected joint [1,T,36,16,C] layout")
    if not 0 <= view < 4 or not 0 <= latent < value.shape[1]:
        raise ValueError("Invalid view/latent")
    return value[0, latent, view * 9:(view + 1) * 9]


class ResidualHooks(AbstractContextManager):
    """Capture raw sites, or replace one exact tile; always remove every hook.

    Input is site0; block i's output is site i+1. Tuple auxiliaries and all
    unedited tokens are preserved. Full captures retain a differentiable tensor
    reference; detached captures store only the chosen tile to bound memory.
    """

    def __init__(self, blocks, *, capture=False, differentiable=False,
                 site=None, replacement=None, view=0, latent=7):
        self.blocks, self.capture, self.differentiable = list(blocks), capture, differentiable
        self.site, self.replacement, self.view, self.latent = site, replacement, view, latent
        self.values, self.handles = {}, []
        if (site is None) != (replacement is None):
            raise ValueError("A replacement and site must be specified together")
        if site is not None and not 0 <= site <= len(self.blocks):
            raise ValueError("Invalid residual site")

    def apply(self, index, value):
        if index in self.values:
            raise ValueError("A residual site executed twice")
        if index == self.site:
            old = tile(value, view=self.view, latent=self.latent)
            donor = self.replacement.to(device=old.device, dtype=old.dtype)
            if donor.shape != old.shape or not torch.isfinite(donor).all():
                raise ValueError("Replacement must have the finite exact tile shape")
            value = value.clone()
            value[0, self.latent, self.view * 9:(self.view + 1) * 9] = donor
        if self.capture:
            self.values[index] = value if self.differentiable else tile(value, view=self.view, latent=self.latent).detach().float().clone()
        return value

    def __enter__(self):
        try:
            def pre(_module, args):
                return (self.apply(0, args[0]), *args[1:])
            self.handles.append(self.blocks[0].register_forward_pre_hook(pre))
            for index, block in enumerate(self.blocks, start=1):
                def post(_module, _args, output, site=index):
                    value = output[0] if isinstance(output, tuple) else output
                    changed = self.apply(site, value)
                    return (changed, *output[1:]) if isinstance(output, tuple) else changed
                self.handles.append(block.register_forward_hook(post))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def torch_predict(model, features):
    """Use the frozen FP64 coefficients; do not autocast or refit a readout."""
    with torch.autocast(features.device.type, enabled=False):
        x = features.double()
        get = lambda name: torch.as_tensor(getattr(model, name), device=x.device, dtype=torch.float64)
        if x.shape[-1] != len(model.x_mean):
            raise ValueError("Probe input width differs")
        return ((x - get("x_mean")) / get("x_scale") @ get("coefficient")) * get("y_scale") + get("y_mean")


def endpoint(inputs, prediction):
    """Euler clean-endpoint estimate from upstream v=z_clean-z_noise training."""
    if prediction.shape != inputs["z_t"].shape:
        raise ValueError("Flow prediction shape differs from latent input")
    return inputs["z_t"].float() + (1 - inputs["tau"].float()) * prediction.float()


def output_objective(inputs, prediction, probe, donor_ball_z):
    clean_estimate = endpoint(inputs, prediction)
    values = torch_predict(probe, tile(clean_estimate).reshape(1, -1))[0]
    objective = -((values[2] - donor_ball_z) / float(probe.y_scale[2])) ** 2
    if not torch.isfinite(values).all() or not torch.isfinite(objective):
        raise ValueError("Nonfinite output objective")
    return objective, values, clean_estimate


def spatial_pullback(probe, projection, target_index):
    """Raw tile gradient of one saved3x4-bin/128channel linear prediction.

    A bin contains3x4=12 tokens; each token receives exactly1/12 of its
    projected bin's weight. Discovery x/y scales are part of the derivative.
    """
    if probe.coefficient.shape[0] != 1536 or projection.shape != (2048, 128):
        raise ValueError("Expected registered1536D spatial probe and2048x128 projection")
    coefficient = probe.coefficient[:, target_index] * probe.y_scale[target_index] / probe.x_scale
    coefficient = torch.as_tensor(coefficient.reshape(3, 4, 128), dtype=torch.float64, device=projection.device)
    with torch.autocast(projection.device.type, enabled=False):
        directions = coefficient @ projection.double().T / 12
    return directions.repeat_interleave(3, 0).repeat_interleave(4, 1).float()


def tile_features(value, projection):
    if value.shape != (9, 16, 2048):
        raise ValueError("Expected one9x16x2048 residual tile")
    with torch.autocast(value.device.type, enabled=False):
        return (value.float().reshape(3, 3, 4, 4, 2048).mean((1, 3)) @ projection.float()).reshape(1, 1536)


def intervention_tiles(recipient, donor, probe, projection, *, seed):
    """Fixed full transfer, component transfer, removal/restoration and controls."""
    recipient, donor = recipient.float(), donor.float()
    if recipient.shape != donor.shape or not torch.isfinite(donor).all():
        raise ValueError("Finite equal-shape donor and recipient required")
    direction = spatial_pullback(probe, projection, 2)
    wrong = spatial_pullback(probe, projection, 0)
    norm, wrong_norm = torch.linalg.vector_norm(direction), torch.linalg.vector_norm(wrong)
    if norm <= 0 or wrong_norm <= 0:
        raise ValueError("Probe direction has zero norm")
    unit, wrong_unit = direction / norm, wrong / wrong_norm
    dose = ((donor - recipient).double() * unit.double()).sum().float()
    delta = dose * unit
    prediction = torch_predict(probe, tile_features(recipient, projection))[0, 2]
    centered = float(prediction - probe.y_mean[2])
    ablated = recipient - (centered / norm) * unit
    # Restore from the exact cached recipient. Report this as an implementation
    # equality control, not as independent evidence for component sufficiency.
    restored = recipient.clone()
    generator = torch.Generator(device=recipient.device).manual_seed(seed)
    random = torch.randn(recipient.shape, device=recipient.device, generator=generator)
    random_unit = random / torch.linalg.vector_norm(random)
    random = random_unit * torch.linalg.vector_norm(delta)
    result = {"self": recipient, "exact_donor": donor, "component_transfer": recipient + delta,
              "ablate_to_discovery_mean": ablated, "restore": restored,
              "norm_matched_random": recipient + random,
              "norm_matched_random_full": recipient + random_unit * torch.linalg.vector_norm(donor-recipient),
              "norm_matched_random_ablation": recipient + random_unit * torch.linalg.vector_norm(ablated-recipient),
              "wrong_variable_ball_x": recipient + dose * wrong_unit}
    return result, delta, {"signed_component_dose": float(dose), "component_delta_l2": float(torch.linalg.vector_norm(delta)),
                          "ablation_delta_l2": float(torch.linalg.vector_norm(ablated-recipient)),
                          "height_direction_l2": float(norm), "height_x_direction_cosine": float((unit.double()*wrong_unit.double()).sum()),
                          "recipient_site_ball_z_proxy": float(prediction), "ablation_target": float(probe.y_mean[2])}


def matched_pairs(records, assignments):
    """First and last clip IDs per match, chosen without targets/effects."""
    roles = {"discovery", "selection"}
    if any(row["split"] not in roles for row in records):
        raise ValueError("Confirmation/pilot records are forbidden")
    if len({row["clip_id"] for row in records}) != len(records):
        raise ValueError("Duplicate clip IDs")
    grouped = {}
    for row in records:
        assignment = assignments.get(row["match_id"])
        if assignment is None or assignment["split"] != row["split"]:
            raise ValueError("Original match role differs")
        grouped.setdefault(row["match_id"], []).append(row)
    pairs = []
    for match, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: row["clip_id"])
        if len(group) != 8:
            raise ValueError("Exactly eight clips per match required")
        pairs.append({"match_id": match, "split": group[0]["split"], "recipient": group[0], "donor": group[-1]})
    return pairs
