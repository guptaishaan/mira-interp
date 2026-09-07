"""Bounded evaluator heads; labels are supervised targets, never encoder inputs."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SpatialStateHead(nn.Module):
    """Discovery-normalized spatial descriptor -> standardized twelve state targets."""
    def __init__(self, x_mean, x_scale, y_mean, y_scale, *, hidden=256, dropout=.1):
        super().__init__()
        arrays = [np.asarray(value, dtype=np.float32) for value in (x_mean, x_scale, y_mean, y_scale)]
        if arrays[0].ndim != 1 or arrays[0].shape != arrays[1].shape or arrays[2].ndim != 1 or arrays[2].shape != arrays[3].shape:
            raise ValueError('Invalid normalizer dimensions')
        if not all(np.isfinite(value).all() for value in arrays) or np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0):
            raise ValueError('Normalizers must be finite with positive scales')
        if hidden != 256 or not 0 <= dropout < 1:
            raise ValueError('This bounded head uses256 hidden units and valid dropout')
        for name, value in zip(('x_mean', 'x_scale', 'y_mean', 'y_scale'), arrays):
            self.register_buffer(name, torch.from_numpy(value.copy()))
        self.layers = nn.Sequential(nn.Linear(len(arrays[0]), hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, len(arrays[2])))

    def forward(self, features):
        return self.layers((features.float()-self.x_mean)/self.x_scale)

    def predict(self, features):
        return self(features)*self.y_scale+self.y_mean


def enable_last_blocks(encoder, count=2):
    """Freeze encoder, then enable exactly its last two blocks and final norm."""
    if count != 2 or len(encoder.encoder.layer) != 12:
        raise ValueError('Expected twelve VideoMAE blocks, with last two enabled')
    encoder.requires_grad_(False)
    for block in encoder.encoder.layer[-count:]:
        block.requires_grad_(True)
    encoder.layernorm.requires_grad_(True)
    allowed = ('encoder.layer.10.', 'encoder.layer.11.', 'layernorm.')
    if any(parameter.requires_grad != name.startswith(allowed) for name, parameter in encoder.named_parameters()):
        raise ValueError('Unexpected trainable encoder parameters')
    return {'trainable': sum(p.numel() for p in encoder.parameters() if p.requires_grad),
            'frozen': sum(p.numel() for p in encoder.parameters() if not p.requires_grad),
            'trainable_blocks': [10, 11], 'final_layernorm': True}
