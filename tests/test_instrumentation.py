"""Numerical contracts for causal hooks; toy networks are engineering evidence only."""

import pytest
import torch
from torch import nn

from mira_interp.instrumentation import (
    CaptureMetadata,
    Intervention,
    ResidualTrace,
    attribution_scores,
    norm_matched_random_delta,
    residual_sites,
)


class TupleBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.cache = (torch.tensor([7.0]), torch.tensor([8.0]))
        self.time_attention = True

    def forward(self, value):
        return value * self.scale, self.cache


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleList([TupleBlock(2), TupleBlock(3)])
        self.caches = []

    def forward(self, value):
        self.caches = []
        for block in self.transformer:
            value, cache = block(value)
            self.caches.append(cache)
        return value


def metadata():
    return CaptureMetadata(
        sample_ids=("engineering-only-0", "engineering-only-1"),
        condition="unit-test", noise_seed=17, noise_id="fixed-unit-input",
        rollout_step=0, diffusion_step=0, tau=(0.5, 0.5), tau_shape=(2, 1),
        latent_frame_indices=(0,),
    )


def values():
    return torch.arange(16.0).reshape(2, 1, 2, 2, 2).requires_grad_()


def test_all_layers_noop_correct_boundary_and_cache_preservation():
    model, x = LinearModel(), values()
    expected = model(x)
    with ResidualTrace(model, metadata()) as trace:
        actual = model(x)
    assert torch.equal(expected, actual)
    assert set(trace.activations) == {0, 1}
    assert torch.equal(trace.activations[0], 2 * x)
    assert torch.equal(trace.activations[1], 6 * x)
    assert torch.equal(trace.input_activation, x)
    assert all(cache is block.cache for cache, block in zip(model.caches, model.transformer))
    assert all(not block._forward_hooks for block in model.transformer)
    assert [site.module_path for site in residual_sites(model)] == ["transformer.0", "transformer.1"]


def test_mask_locality_exact_downstream_effect_and_no_input_mutation():
    model, x = LinearModel(), values()
    before = x.detach().clone()
    mask = torch.zeros(x.shape[:-1], dtype=torch.bool)
    mask[0, 0, 1, 0] = True
    donor = 2 * x.detach() + 10
    with ResidualTrace(model, metadata(), interventions={0: Intervention("replace", donor, mask)}) as trace:
        actual = model(x)
    expanded = mask.unsqueeze(-1).expand_as(x)
    expected_at_site = torch.where(expanded, donor, 2 * x)
    assert torch.equal(trace.activations[0], expected_at_site)
    assert torch.equal(actual, 3 * expected_at_site)
    assert torch.equal(actual[~expanded], (6 * x)[~expanded])
    assert torch.equal(x, before)


def test_ablation_then_restoration_recovers_original_output():
    model, x = LinearModel(), values()
    expected = model(x)
    with ResidualTrace(model, metadata(), interventions={0: Intervention("ablate")}) as ablated:
        zeroed = model(x)
    assert torch.count_nonzero(ablated.activations[0]) == 0
    assert torch.count_nonzero(zeroed) == 0
    sequence = (Intervention("ablate"), Intervention("restore", 2 * x.detach()))
    with ResidualTrace(model, metadata(), interventions={0: sequence}):
        restored = model(x)
    assert torch.equal(expected, restored)


def test_attribution_matches_exact_effect_for_linear_downstream_network():
    model, clean = LinearModel(), values()
    corrupted = (clean.detach() + 0.25).requires_grad_()
    with ResidualTrace(model, metadata()) as clean_trace:
        model(clean)
    with ResidualTrace(model, metadata(), retain_grad=True) as corrupted_trace:
        baseline = model(corrupted)
        objective = baseline.flatten(1).sum(1)
        objective.sum().backward()
    for layer in (0, 1):
        activation = corrupted_trace.activations[layer]
        assert activation.grad is not None
        estimated = attribution_scores(clean_trace.activations[layer], activation, activation.grad)
        with ResidualTrace(
            model, metadata(), interventions={layer: Intervention("replace", clean_trace.activations[layer])}
        ):
            patched = model(corrupted)
        exact = patched.flatten(1).sum(1) - objective.detach()
        assert torch.allclose(estimated, exact, atol=1e-6, rtol=1e-6)
        assert (estimated < 0).all()  # verifies clean-minus-corrupted direction


def test_cleanup_on_forward_exception_and_repeated_call_rejected():
    model = LinearModel()
    with pytest.raises(RuntimeError, match="one forward"):
        with ResidualTrace(model, metadata()):
            model(values())
            model(values())
    assert all(not block._forward_hooks for block in model.transformer)
    with pytest.raises(ValueError, match="exactly"):
        with ResidualTrace(model, metadata(), interventions={0: Intervention("replace", torch.ones(1))}):
            model(values())
    assert all(not block._forward_hooks for block in model.transformer)


def test_trace_rejects_missing_layers_bad_metadata_and_unknown_sites():
    model = LinearModel()
    with pytest.raises(RuntimeError, match="Incomplete"):
        with ResidualTrace(model, metadata()):
            pass
    with pytest.raises(ValueError, match="Unknown zero-based"):
        ResidualTrace(model, metadata(), interventions={2: Intervention("ablate")})
    with pytest.raises(ValueError, match="tau_shape"):
        CaptureMetadata(("x",), "test", 0, "noise", 0, 0, (0.5,), (2,), (0,))
    with pytest.raises(ValueError, match="boolean"):
        Intervention("ablate", mask=torch.ones(2, 1, 2, 2)).apply(values())
    with pytest.raises(ValueError, match="token-grid"):
        Intervention("ablate", mask=torch.ones(2, 1, dtype=torch.bool)).apply(values())


def test_random_control_matches_each_sample_delta_norm_reproducibly():
    delta = values().detach()
    delta[0] = 0
    first = norm_matched_random_delta(delta, generator=torch.Generator().manual_seed(6))
    second = norm_matched_random_delta(delta, generator=torch.Generator().manual_seed(6))
    assert torch.equal(first, second)
    assert torch.allclose(first.flatten(1).norm(dim=1), delta.flatten(1).norm(dim=1))
    assert torch.count_nonzero(first[0]) == 0


def test_nonfinite_values_rejected_before_patch_or_attribution():
    x = values()
    donor = x.detach().clone()
    donor[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        Intervention("replace", donor).apply(x)
    with pytest.raises(ValueError, match="finite"):
        attribution_scores(x, x, donor)


def test_register_slots_have_separate_time_accounting():
    model = LinearModel()
    model.n_register_tokens = 1
    meta = CaptureMetadata(("x",), "test", 0, "noise", 0, 0, (0.5,), (1, 1), (0,))
    with ResidualTrace(model, meta) as trace:
        model(torch.ones(1, 2, 2, 2, 2))
    assert trace.manifest()["register_slots_in_this_call"] == 1
    with pytest.raises(ValueError, match="register slots"):
        with ResidualTrace(model, meta):
            model(torch.ones(1, 1, 2, 2, 2))
