"""Exact frozen-prefix split for partial VideoMAE adaptation, with explicit FP32 weights."""
import hashlib
import math

import torch
from torch import nn


def deterministic_pool(hidden):
    """Same adaptive3x4 bins as frozen readout, via deterministic linear algebra."""
    if hidden.ndim != 3 or hidden.shape[1:] != (1568, 768):
        raise ValueError('Expected complete VideoMAE token grid')
    weights = hidden.new_zeros((12, 14, 14), dtype=torch.float32)
    for row in range(3):
        for col in range(4):
            top, bottom = math.floor(row*14/3), math.ceil((row+1)*14/3)
            left, right = math.floor(col*14/4), math.ceil((col+1)*14/4)
            weights[row*4+col, top:bottom, left:right] = 1/((bottom-top)*(right-left))
    # Pooling in the frozen capture explicitly uses FP32. Disable autocast
    # here so a matmul implementation does not silently introduce BF16 pooling.
    with torch.autocast('cuda', enabled=False):
        last = hidden[:, -196:].float().transpose(1, 2)
        return (last @ weights.flatten(1).T).flatten(1)


@torch.no_grad()
def frozen_prefix(encoder, video):
    hidden = encoder.embeddings(video, bool_masked_pos=None)
    for layer in encoder.encoder.layer[:10]:
        hidden = layer(hidden)
    if hidden.shape[1:] != (1568, 768) or not torch.isfinite(hidden).all():
        raise ValueError('Invalid frozen VideoMAE prefix output')
    if torch.is_autocast_enabled('cuda') and hidden.dtype != torch.bfloat16:
        raise ValueError('Expected native BF16 residuals under the registered CUDA autocast')
    return hidden.detach()


class VideoMAETail(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        if len(encoder.encoder.layer) != 12 or encoder.layernorm is None:
            raise ValueError('Expected twelve blocks and final layernorm')
        self.blocks = nn.ModuleList(encoder.encoder.layer[-2:])
        self.norm = encoder.layernorm
        self.head = head
        self.blocks.requires_grad_(True); self.norm.requires_grad_(True)
        # Original Transformers4.22 has no learned key bias: the adapter added
        # known zeros for strict loading into current Linear modules.
        for block in self.blocks:
            block.attention.attention.key.bias.requires_grad_(False)

    def descriptor(self, prefix):
        # Saving BF16 values losslessly in an FP32 array does not preserve
        # residual-addition arithmetic unless the native dtype is restored.
        hidden = prefix.to(torch.bfloat16) if torch.is_autocast_enabled('cuda') else prefix
        for layer in self.blocks:
            hidden = layer(hidden)
        return deterministic_pool(self.norm(hidden))

    def forward(self, prefix):
        return self.head(self.descriptor(prefix))

    def predict(self, prefix):
        return self(prefix)*self.head.y_scale+self.head.y_mean


def parameter_digest(named_parameters):
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(tail):
    return torch.optim.AdamW([
        {'params': list(tail.blocks.parameters())+list(tail.norm.parameters()), 'lr': 1e-5},
        {'params': tail.head.parameters(), 'lr': 1e-3}], weight_decay=.01)
