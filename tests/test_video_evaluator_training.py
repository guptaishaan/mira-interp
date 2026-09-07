"""Engineering checks for downstream normalization and frozen-prefix adaptation."""
import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from mira_interp.video_evaluator_training import SpatialStateHead, enable_last_blocks


def test_head_normalizers_are_fixed_buffers_and_predictions_use_source_units():
    torch.manual_seed(2)
    head = SpatialStateHead(np.ones(8), np.ones(8)*2, np.arange(12.), np.ones(12)*3, dropout=0)
    x = torch.randn(4, 8)
    expected = head.layers((x-1)/2)
    torch.testing.assert_close(head(x), expected)
    torch.testing.assert_close(head.predict(x), expected*3+torch.arange(12.))
    assert not any(name in dict(head.named_parameters()) for name in ('x_mean','x_scale','y_mean','y_scale'))
    saved = head.x_mean.clone(); head(x).square().mean().backward()
    assert head.x_mean.grad is None and torch.equal(saved, head.x_mean)
    with pytest.raises(ValueError, match='positive'):
        SpatialStateHead(np.ones(8), np.zeros(8), np.ones(12), np.ones(12))


def test_only_last_two_blocks_and_norm_can_train():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__(); self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([nn.Linear(2,2) for _ in range(12)])
            self.layernorm = nn.LayerNorm(2); self.embeddings = nn.Linear(2,2)
    encoder = Encoder(); report = enable_last_blocks(encoder)
    assert report['trainable'] == 16 and report['frozen'] == 66
    for name, parameter in encoder.named_parameters():
        assert parameter.requires_grad == name.startswith(('encoder.layer.10.', 'encoder.layer.11.', 'layernorm.'))
    with pytest.raises(ValueError, match='last two'):
        enable_last_blocks(encoder, 1)


def test_deterministic_pool_matches_original_bins_and_linear_gradient():
    from mira_interp.video_evaluator import pool_last_tubelet
    from mira_interp.video_evaluator_finetune import deterministic_pool
    torch.manual_seed(0)
    hidden = torch.randn(1, 1568, 768, requires_grad=True)
    actual = deterministic_pool(hidden)
    expected = pool_last_tubelet(hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    assert torch.count_nonzero(hidden.grad[:, :-196]) == 0
    np.testing.assert_allclose(hidden.grad[:, -196:].sum(1).numpy(), np.full((1,768),12), rtol=1e-6)
    # A linear readout must give the identical derivative independently of values.
    alternative = (hidden.detach()*7).requires_grad_(True)
    deterministic_pool(alternative).sum().backward()
    torch.testing.assert_close(hidden.grad, alternative.grad, rtol=0, atol=0)
