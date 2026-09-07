"""Independent frozen VideoMAE pixel encoder and explicit state/codec calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import TARGET_NAMES
from .development import ROLE_TARGET_NAMES, role_targets
from .model_loading import sha256

VIDEOMAE_REVISION = 'dc740ceda42fce44faed2ea03c6d447db72f6af9'
VIDEOMAE_WEIGHT_SHA = 'bc053ca2840a038b1068269a4eec06ca569689e9a1ed9376a5b2b8a111be5290'
CONFIG_HASHES = {'config.json': '5fbedfea3706bd9fe6c0dc706beeb501ae72e4954703e03bebe3406f046f8212',
                 'preprocessor_config.json': 'c3aa722f22a7ff0d234d407862025fe47672f74f46f92c91f757f7a7354107c2'}


def convert_legacy_biases(state):
    """Exact Transformers4.22 Q/V biases ->5.x Linear biases; old K bias is zero."""
    converted = dict(state); count = 0
    for key in list(state):
        if key.endswith('.q_bias'):
            prefix = key[:-len('q_bias')]
            q, v = converted.pop(key), converted.pop(prefix+'v_bias')
            if any(prefix+name+'.bias' in converted for name in ('query', 'key', 'value')):
                raise ValueError('Ambiguous old and new VideoMAE bias keys')
            converted.update({prefix+'query.bias': q, prefix+'value.bias': v, prefix+'key.bias': torch.zeros_like(q)})
            count += 1
    return converted, count


def load_encoder(assets):
    """Strictly load every official pretraining tensor on CPU, then retain encoder."""
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig, VideoMAEForPreTraining
    import transformers

    assets = Path(assets)
    if sha256(assets/'model.safetensors') != VIDEOMAE_WEIGHT_SHA:
        raise ValueError('VideoMAE checkpoint checksum mismatch')
    if any(sha256(assets/name) != digest for name, digest in CONFIG_HASHES.items()):
        raise ValueError('VideoMAE config or preprocessing hash mismatch')
    config = VideoMAEConfig.from_json_file(assets/'config.json')
    signature = tuple(getattr(config, key) for key in ('num_frames', 'tubelet_size', 'image_size', 'patch_size', 'hidden_size'))
    if signature != (16, 2, 224, 16, 768):
        raise ValueError('Unexpected VideoMAE token layout')
    model = VideoMAEForPreTraining(config)
    original = load_file(str(assets/'model.safetensors'), device='cpu')
    state, converted = convert_legacy_biases(original)
    if converted != 16:
        raise ValueError('Expected12 encoder and4 pretraining-decoder bias conversions')
    torch.nn.Module.load_state_dict(model, state, strict=True, assign=True)
    max_bias_error = 0.
    with torch.inference_mode():
        for key in original:
            if key.endswith('.q_bias'):
                prefix = key[:-len('q_bias')]
                attention = model.get_submodule(prefix.rstrip('.'))
                example = torch.linspace(-1, 1, attention.query.in_features).reshape(1, 1, -1)
                for name, bias in [('query', original[key]), ('key', torch.zeros_like(original[key])),
                                   ('value', original[prefix+'v_bias'])]:
                    reference = F.linear(example, original[prefix+name+'.weight'], bias)
                    actual = getattr(attention, name)(example)
                    max_bias_error = max(max_bias_error, float((reference-actual).abs().max()))
    if max_bias_error != 0.:
        raise ValueError('Legacy bias mapping failed exact projection equivalence')
    encoder = model.videomae.eval().requires_grad_(False)
    preprocess = json.loads((assets/'preprocessor_config.json').read_text())
    report = {'model': 'MCG-NJU/videomae-base', 'revision': VIDEOMAE_REVISION,
              'weight_sha256': VIDEOMAE_WEIGHT_SHA, 'strict_loaded_tensors': len(state),
              'original_checkpoint_tensors': len(original), 'converted_attention_modules': converted,
              'bias_conversion_max_abs_error': max_bias_error,
              'bias_conversion_source': 'https://github.com/huggingface/transformers/blob/v4.22.0/src/transformers/models/videomae/modeling_videomae.py',
              'encoder_parameters': sum(p.numel() for p in encoder.parameters()),
              'transformers_version': transformers.__version__, 'torch_version': torch.__version__,
              'config_sha256': sha256(assets/'config.json'),
              'processor_config_sha256': sha256(assets/'preprocessor_config.json'),
              'frozen_encoder': True, 'has_cls_token': False, 'tubelets': 8, 'spatial_grid': [14, 14],
              'readout': 'last tubelet,3x4 spatial bins,all768 channels;9216D',
              'preprocessing': 'full-frame224x224 FP32 bilinear antialias; no center crop; published mean/std',
              'domain_limit': 'codec reconstructions do not establish reliability on generated or intervened videos'}
    return encoder, preprocess, report


def preprocess_frames(frames, preprocess, device):
    """Preserve the whole camera image; record the deliberate aspect-ratio change."""
    frames = np.asarray(frames)
    if frames.ndim != 5 or frames.shape[1:3] != (16, 3) or not np.isfinite(frames).all():
        raise ValueError('Require finite video[B,16,3,H,W]')
    if frames.min() < -1e-3 or frames.max() > 255 + 1e-3:
        raise ValueError('Video input must be floating/uint8 source values in0..255')
    video = torch.as_tensor(frames, dtype=torch.float32, device=device)
    b, t, c, h, w = video.shape
    video = F.interpolate(video.reshape(b*t, c, h, w), size=(224, 224), mode='bilinear', antialias=True)
    mean = video.new_tensor(preprocess['image_mean']).view(1, 3, 1, 1)
    std = video.new_tensor(preprocess['image_std']).view(1, 3, 1, 1)
    return ((video / 255 - mean) / std).reshape(b, t, c, 224, 224)


def pool_last_tubelet(hidden):
    if hidden.ndim != 3 or hidden.shape[1:] != (8*14*14, 768):
        raise ValueError('Expected VideoMAE patch tokens[B,1568,768], with no CLS token')
    last = hidden[:, -14*14:].reshape(-1, 14, 14, 768).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(last.float(), (3, 4)).flatten(1)


def encode_frames(encoder, frames, preprocess, *, device='cpu'):
    outputs = []
    with torch.inference_mode():
        for view in range(len(frames)):
            video = preprocess_frames(frames[view:view+1], preprocess, device)
            hidden = encoder(pixel_values=video, bool_masked_pos=None).last_hidden_state
            feature = pool_last_tubelet(hidden)
            if not torch.isfinite(feature).all():
                raise ValueError('Nonfinite VideoMAE features')
            outputs.append(feature.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def endpoint_labels(source):
    if source['targets'].shape != (4, 16, 30) or list(source['target_names']) != list(TARGET_NAMES):
        raise ValueError('Expected canonical four-view state labels')
    if source['timestamps'].shape != (4, 16) or source['player_ids'].shape != (4,):
        raise ValueError('Missing per-view timing or player identity')
    if len(set(source['player_ids'].tolist())) != 4:
        raise ValueError('Duplicate players')
    absolute = source['targets'][:, -1].astype(np.float64)
    labels = role_targets(absolute, np.arange(4))
    timestamps = source['timestamps'][:, -1]
    indices = source['source_frame_indices']
    if indices.shape == (16,):
        indices = np.repeat(indices[-1], 4)
    elif indices.shape == (4, 16):
        indices = indices[:, -1]
    else:
        raise ValueError('Invalid source-frame alignment')
    if not np.isfinite(labels).all() or not np.isfinite(timestamps).all():
        raise ValueError('Invalid endpoint labels or timestamps')
    return {'y': labels, 'target_names': np.asarray(ROLE_TARGET_NAMES), 'timestamps': timestamps,
            'source_frame_index': indices, 'view_index': np.arange(4), 'absolute_ball6': absolute[:, :6]}


def load_roundtrip_codec(assets):
    """Reuse the existing fully strict loader, retaining only its verified codec."""
    from .model_loading import load_pretrained_four_player
    model, report = load_pretrained_four_player(Path(assets))
    codec = model.codec
    del model
    return codec, report


def codec_roundtrip(codec, frames, *, device='cpu'):
    """Encode/decode each view; convert normalized decoder output to clipped RGB."""
    frames = np.asarray(frames)
    if frames.shape != (4, 16, 3, 288, 512) or not np.isfinite(frames).all():
        raise ValueError('Require all four16-frame native-size views')
    outputs, fractions = [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
        for view in range(4):
            video = torch.as_tensor(frames[view:view+1], dtype=torch.float32, device=device) / 255
            _, encoded = codec.encode(video, trim_video=False)
            decoded = codec.decode(encoded.z).float()
            if decoded.shape != video.shape or not torch.isfinite(decoded).all():
                raise ValueError('Codec reconstruction alignment/values invalid')
            fractions.append(float(((decoded < -1) | (decoded > 1)).float().mean()))
            outputs.append(((decoded.clamp(-1, 1) + 1) * 127.5).cpu().numpy())
    reconstruction = np.concatenate(outputs)
    return reconstruction, {'clipped_fraction_by_view': fractions,
                            'rgb_mse_0_255': float(np.mean((reconstruction - frames) ** 2)),
                            'label_assumption': 'same source endpoint; codec reconstruction may change visible state'}
