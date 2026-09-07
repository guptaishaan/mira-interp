"""One-shot residual steering inside the original streaming inference sampler."""
from __future__ import annotations

from contextlib import AbstractContextManager
import numpy as np
import torch

from mira_interp.causal_development import tensor_hash
from mira_interp.sparse_causal_fidelity import spatial_descriptor, spatial_lift

HEIGHT_SUPPORT = (86.37890625, 1810.9219970703125)
DOSES = (-600., -300., 0., 300., 600.)
PATHS = ("probe_linear", "affine_forward", "quadratic_forward", "random_norm", "wrong_variable_ball_x")


def conditions():
    return [("baseline", 0.)] + [(name, dose) for name in PATHS for dose in DOSES if dose != 0]


def hidden_video(frames, *, placeholder=0., context=16):
    """Return a fresh tensor; never pass observed future pixels to inference."""
    if frames.ndim != 5 or frames.shape[:3] != (4, 24, 3) or frames.dtype != torch.float32:
        raise ValueError("Expected FP32 [4,24,3,H,W] input")
    if not torch.isfinite(frames).all() or float(frames.min()) < -1e-3 or float(frames.max()) > 255.001:
        raise ValueError("Input pixels must be finite in the original 0..255 scale")
    if context != 16 or placeholder not in (0., 255.):
        raise ValueError("Only fixed16 context and registered zero/255 placeholders supported")
    video = frames.clone()
    video[:, context:] = placeholder
    return video


def clipped_heights(base, dose):
    if not np.isfinite(base) or dose not in DOSES:
        raise ValueError("Finite base height and registered dose required")
    clipped_base = float(np.clip(base, *HEIGHT_SUPPORT))
    target = float(np.clip(clipped_base + dose, *HEIGHT_SUPPORT))
    return {"base_height_raw": float(base), "base_height_clipped": clipped_base, "requested_dose": float(dose),
            "target_height_clipped": target, "effective_dose": target-clipped_base,
            "base_clipped": clipped_base != base, "target_clipped": target != clipped_base+dose}


def forward_map(maps, kind, height):
    if kind not in {"affine", "quadratic"}:
        raise ValueError("Only frozen affine/quadratic forward maps supported")
    z = (float(height)-float(maps["physical_mean"])) / float(maps["physical_scale"])
    x = np.array([[z]] if kind == "affine" else [[z, z*z]], dtype=np.float64)
    get = lambda name: maps[f"{kind}__{name}"]
    return (((x-get("x_mean"))/get("x_scale")) @ get("coefficient"))*get("y_scale")+get("y_mean")


def raw_probe_direction(probe, target):
    direction = probe.coefficient[:, target]*probe.y_scale[target]/probe.x_scale
    if direction.shape != (1536,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("Invalid fixed spatial probe direction")
    return direction


def path_delta(name, base, dose, probe, maps, projection, *, seed):
    """All paths return descriptor offsets, never overwrite the native nullspace."""
    height = clipped_heights(base, dose)
    target, origin, effective = height["target_height_clipped"], height["base_height_clipped"], height["effective_dose"]
    if name not in PATHS and name != "baseline":
        raise ValueError("Unregistered steering path")
    if name == "baseline" or effective == 0:
        return torch.zeros(1, 1536, device=projection.device), height
    if name == "probe_linear":
        direction = raw_probe_direction(probe, 2)
        delta = effective * direction / np.dot(direction, direction)
    elif name in {"affine_forward", "quadratic_forward"}:
        kind = name.removesuffix("_forward")
        delta = (forward_map(maps, kind, target)-forward_map(maps, kind, origin)).ravel()
    else:
        quadratic = forward_map(maps, "quadratic", target)-forward_map(maps, "quadratic", origin)
        reference = torch.as_tensor(quadratic, dtype=torch.float32, device=projection.device)
        norm = torch.linalg.vector_norm(spatial_lift(reference, projection))
        direction = np.random.default_rng(seed).normal(size=1536) if name == "random_norm" else raw_probe_direction(probe, 0)
        candidate = torch.as_tensor(np.sign(effective)*direction[None], dtype=torch.float32, device=projection.device)
        delta = candidate * (norm/torch.linalg.vector_norm(spatial_lift(candidate, projection)))
        return delta, height
    return torch.as_tensor(delta.reshape(1, 1536), dtype=torch.float32, device=projection.device), height


class FirstGeneratedEdit(AbstractContextManager):
    """Observe every sampler call; edit exactly one true denoising forward.

    Registers at the world-model boundary to inspect actual tau/cache state,
    then at block15 output to preserve its tuple/cache auxiliary. Calls at tau1
    initialize or update the cache and are never edited. Each context manager
    starts a new rollout trace; the caller invokes original model.inference.
    """
    def __init__(self, world_model, *, schedule, step_index=8, descriptor_delta=None, projection=None):
        self.world_model, self.schedule, self.step_index = world_model, list(schedule), step_index
        self.target_tau = float(self.schedule[step_index])
        if len(self.schedule) != 11 or not 0 < self.target_tau < 1 or step_index != 8:
            raise ValueError("Expected registered10-step schedule and step index8")
        self.delta, self.projection = descriptor_delta, projection
        if descriptor_delta is not None and projection is None:
            raise ValueError("An edit requires the fixed projection")
        self.handles, self.trace = [], []
        self.active, self.latent, self.step, self.hits = False, -1, -1, 0
        self.native_tile, self.edited_tile, self.details = None, None, None

    def before(self, module, args, kwargs):
        get = lambda index, key, default=None: args[index] if len(args) > index else kwargs.get(key, default)
        value, actions, tau = get(0, "z_t"), get(1, "a"), get(2, "tau")
        cache, returning = get(5, "kv_caches"), get(4, "return_kv", False)
        if value.ndim != 5 or value.shape[0] != 1 or value.shape[2:4] != (36, 16):
            raise ValueError("Unexpected streaming latent layout")
        ts = tau.detach().float().flatten()
        if not torch.isfinite(ts).all() or not torch.all(ts == ts[0]):
            raise ValueError("Expected a uniform finite sampler tau")
        actual = float(ts[0])
        if tau.dtype not in (torch.float32, torch.bfloat16) or value.dtype != tau.dtype:
            raise ValueError("Sampler tau must retain the observed FP32/BF16 latent dtype")
        realized_schedule = torch.as_tensor(self.schedule, dtype=tau.dtype).float().tolist()
        self.active = False
        if not self.trace:
            if value.shape[1] != 8 or actual != 1 or cache is not None or returning is not True:
                raise ValueError("Each rollout must start from fresh8-latent context and no cache")
            phase = "context_cache_initialization"
        elif actual == 1:
            if value.shape[1] != 1 or cache is None or returning is not True or self.step != 9:
                raise ValueError("Unexpected cache update")
            phase = "generated_cache_update"
        else:
            if value.shape[1] != 1 or cache is None or returning:
                raise ValueError("Edit candidates must be true single-frame denoising forwards")
            if actual == 0:
                if self.trace[-1]["phase"] not in {"context_cache_initialization", "generated_cache_update"}:
                    raise ValueError("Denoising restarted without a cache boundary")
                self.latent += 1
                self.step = 0
            else:
                self.step += 1
            if self.latent not in range(4) or self.step not in range(10) or actual != realized_schedule[self.step]:
                raise ValueError("Actual sampler step differs from registered schedule")
            phase = "denoising"
            self.active = self.latent == 0 and self.step == self.step_index
        self.trace.append({"call_index": len(self.trace), "phase": phase, "generated_latent_index": self.latent,
                           "step_index": self.step, "tau": actual, "time_tokens": value.shape[1],
                           "tau_dtype": str(tau.dtype), "latent_dtype": str(value.dtype),
                           "nominal_schedule_tau": self.schedule[self.step] if phase == "denoising" else 1.,
                           "cache_present": cache is not None, "return_kv": bool(returning), "target_call": self.active,
                           "latent_input_sha256": tensor_hash(value), "action_features_sha256": tensor_hash(actions)})

    def after_block(self, module, args, output):
        if not self.active:
            return output
        self.hits += 1
        if self.hits != 1:
            raise ValueError("More than one residual intervention attempted")
        value = output[0] if isinstance(output, tuple) else output
        if value.shape != (1, 1, 36, 16, 2048):
            raise ValueError("Expected cached single-frame block15 output without register tokens")
        native = value[0, 0, :9].detach().float().clone()
        self.native_tile = native
        if self.delta is None:
            self.edited_tile = native.clone()
            return output
        replacement = native + spatial_lift(self.delta.to(native.device), self.projection)
        changed = value.clone()
        changed[0, 0, :9] = replacement.to(value.dtype)
        self.edited_tile = changed[0, 0, :9].detach().float().clone()
        self.details = {"requested_descriptor_delta_l2": float(torch.linalg.vector_norm(self.delta)),
                        "actual_raw_edit_l2": float(torch.linalg.vector_norm(self.edited_tile-native)),
                        "realized_descriptor_delta": (spatial_descriptor(self.edited_tile, self.projection)
                                                      - spatial_descriptor(native, self.projection)).cpu().numpy()}
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def validate(self):
        if self.hits != 1 or self.latent != 3 or len(self.trace) != 45 or self.trace[-1]["phase"] != "generated_cache_update":
            raise ValueError("Incomplete four-latent rollout or missed intervention site")

    def __enter__(self):
        try:
            self.handles.append(self.world_model.register_forward_pre_hook(self.before, with_kwargs=True))
            self.handles.append(self.world_model.transformer[15].register_forward_hook(self.after_block))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
