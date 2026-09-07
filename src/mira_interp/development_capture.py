"""Spatially structured development capture; never modifies the frozen v2 path."""
from __future__ import annotations

import hashlib
import io
import numpy as np
import torch
import torch.nn.functional as F

PROJECTION_SEED = 20260907


def channel_projection(device="cpu"):
    """Fixed data-independent orthonormal channel map, shared by all sites."""
    rng = np.random.default_rng(PROJECTION_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((2048, 128)))
    return torch.from_numpy(q.astype(np.float32)).to(device)


def descriptors(residual, projection):
    """Mean and 3x4 spatial bins. Both are linear in the original residual."""
    if residual.ndim != 5 or residual.shape[0] != 1 or residual.shape[2:] != (36, 16, 2048):
        raise ValueError("Expected [1,T,36,16,2048] four-view residual")
    t = residual.shape[1]
    views = residual[0].reshape(t, 4, 9, 16, 2048).permute(1, 0, 2, 3, 4).float()
    mean = views.mean(dim=(2, 3)).reshape(4*t, 2048)
    bins = views.reshape(4, t, 3, 3, 4, 4, 2048).mean(dim=(3, 5))
    # Disable autocast for the readout, so all spatial sums/projections are FP32.
    with torch.autocast("cuda", enabled=False):
        spatial = (bins @ projection).reshape(4*t, 1536)
    return mean, spatial


def decode_float_video(blob, selected, expected_frames=80):
    """Publisher-style fractional RGB resize, with source-frame/PTS validation."""
    import av
    arrays, pts, all_pts = [], [], []
    wanted = set(selected)
    with av.open(io.BytesIO(blob)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if not np.isclose(float(stream.average_rate), 20):
            raise ValueError("Expected 20 FPS")
        for i, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Missing PTS")
            stamp = float(frame.pts * frame.time_base)
            all_pts.append(stamp)
            if i in wanted:
                arrays.append(frame.to_ndarray(format="rgb24"))
                pts.append(stamp)
    if len(all_pts) != expected_frames or len(arrays) != len(selected):
        raise ValueError("Source frame counts disagree")
    if not np.allclose(np.diff(all_pts), .05, rtol=0, atol=1e-7):
        raise ValueError("Nonuniform PTS")
    pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float()
    resized = F.interpolate(pixels, (288, 512), mode="bilinear", align_corners=False, antialias=True)
    return resized.numpy(), np.asarray(pts)


def make_inputs(model, frames, actions_array, seed, tau_value=.5):
    """Inputs depend exclusively on video/actions/noise, not simulator labels."""
    from einops import rearrange
    from mira.data.batch import VideoActionBatch
    from mira.world_model.actions_config import ActionTensors
    if frames.shape != (4, 16, 3, 288, 512) or frames.dtype != np.float32:
        raise ValueError("Require unrounded resized float32 RGB, 4x16x3x288x512")
    # Bilinear FP32 accumulation can overshoot255 by a few ulps (pilot:1.53e-5).
    # Preserve publisher-style fractional values; reject material range errors.
    if not np.isfinite(frames).all() or frames.min() < -1e-3 or frames.max() > 255+1e-3:
        raise ValueError("Invalid RGB range")
    if actions_array.shape != (4, 16, 9) or not np.isin(actions_array, [0, 1]).all():
        raise ValueError("Invalid actions")
    actions = ActionTensors(model.config.actions, batch_size=4)
    actions.key_presses = torch.from_numpy(actions_array.astype(np.int32, copy=False))
    actions.mouse_movements = torch.zeros(4, 16, 2)
    batch = VideoActionBatch(torch.from_numpy(frames), actions).to(model.device)
    swm = model.single_world_model
    model.codec.preprocess_batch(batch)
    z_flat = swm.encode_video(batch)
    if z_flat.shape != (4, 8, 9, 16, 32):
        raise ValueError("Unexpected latent layout")
    z = rearrange(z_flat, "p t h w c -> 1 t (p h) w c")
    off = swm.action_temporal_downsampling - 1
    count = (z.shape[1]-1)*swm.action_temporal_downsampling
    af = model._combine_player_actions(swm.action_encoder(batch.actions.slice_time(off, off+count)))
    generator = torch.Generator(device=model.device).manual_seed(seed)
    noise = torch.randn(z.shape, generator=generator, device=model.device, dtype=z.dtype)
    tau = torch.full((1, 8, 1, 1, 1), tau_value, device=model.device, dtype=torch.float32)
    z_t = tau*z + (1-tau)*noise
    past = torch.cat([swm.bos[None, None], z[:, :-1]], dim=1)
    return dict(z_t=z_t, action_features=af, tau=tau, clean_past=past), z_flat, noise


def forward(model, inputs):
    return model.world_model(inputs["z_t"], inputs["action_features"], inputs["tau"],
                             clean_past=inputs["clean_past"], activation_checkpointing=False)


def capture(model, frames, actions, *, seed, projection, pilot=False):
    """Capture 17 sites with a same-input no-op and repeat check on reserved pilot."""
    stored, handles, controls = {}, [], {}
    def save(site, residual):
        if site in stored:
            raise ValueError("Repeated transformer execution")
        a, b = descriptors(residual, projection)
        stored[site] = (a.cpu(), b.cpu())
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inputs, z_flat, noise = make_inputs(model, frames, actions, seed)
        baseline = forward(model, inputs) if pilot else None
        handles.append(model.world_model.transformer[0].register_forward_pre_hook(lambda _m, args: save(0, args[0])))
        for i, block in enumerate(model.world_model.transformer):
            handles.append(block.register_forward_hook(lambda _m, _a, out, site=i+1: save(site, out[0] if isinstance(out, tuple) else out)))
        try:
            pred = forward(model, inputs)
        finally:
            for h in handles:
                h.remove()
        if pilot:
            repeated = forward(model, inputs)
            controls = {"noop_bitwise_equal": torch.equal(pred, baseline), "repeat_bitwise_equal": torch.equal(pred, repeated)}
            if not all(controls.values()):
                raise ValueError("No-op/repeat check failed")
    if set(stored) != set(range(17)) or not torch.isfinite(pred).all():
        raise ValueError("Incomplete/nonfinite capture")
    arrays = {
        "mean_X": torch.stack([stored[s][0] for s in range(17)], 1).half().numpy(),
        "spatial_X": torch.stack([stored[s][1] for s in range(17)], 1).half().numpy(),
        "codec_spatial_X": z_flat.float().reshape(32, -1).cpu().half().numpy(),
        "codec_mean_X": z_flat.float().mean((2, 3)).reshape(32, 32).cpu().half().numpy(),
        "RGB_X": F.adaptive_avg_pool2d(torch.from_numpy(frames[:, 1::2].copy()).reshape(32, 3, 288, 512)/255, (16, 32)).flatten(1).half().numpy(),
    }
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite stored features")
    return arrays, {"controls": controls, "no_labels_passed_to_model": True,
                    "noise_sha256": hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest(),
                    "prediction_sha256": hashlib.sha256(pred.float().cpu().numpy().tobytes()).hexdigest(),
                    "z_t_dtype": str(inputs["z_t"].dtype), "parameter_dtype": str(next(model.parameters()).dtype)}
