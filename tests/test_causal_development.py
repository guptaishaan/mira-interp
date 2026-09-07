"""Mathematical and hook-locality tests; no research observations or GPU use."""
import numpy as np
import pytest
import torch

from mira_interp.causal_development import (ResidualHooks, endpoint, intervention_tiles,
    matched_pairs, output_objective, spatial_pullback, tile, tile_features, torch_predict)
from mira_interp.probes import RidgeModel


class LinearBlock(torch.nn.Module):
    def __init__(self, factor, tuple_output=False):
        super().__init__()
        self.factor, self.tuple_output = factor, tuple_output

    def forward(self, x):
        y = self.factor * x
        return (y, "unchanged auxiliary") if self.tuple_output else y


def run(blocks, value):
    for block in blocks:
        result = block(value)
        if isinstance(result, tuple):
            assert result[1] == "unchanged auxiliary"
            result = result[0]
        value = result
    return value


def test_input_output_hook_indices_locality_tuple_and_removal():
    blocks = [LinearBlock(2, True), LinearBlock(3)]
    x = torch.arange(1*8*36*16*2, dtype=torch.float32).reshape(1, 8, 36, 16, 2)
    baseline = run(blocks, x)
    with ResidualHooks(blocks, capture=True) as trace:
        assert torch.equal(run(blocks, x), baseline)
    assert torch.equal(trace.values[0], tile(x))
    assert torch.equal(trace.values[1], tile(x)*2)
    assert torch.equal(trace.values[2], tile(x)*6)
    for index, remaining_factor in [(0, 6), (1, 3), (2, 1)]:
        replacement = torch.full_like(tile(x), 5)
        with ResidualHooks(blocks, site=index, replacement=replacement):
            edited = run(blocks, x)
        expected = baseline.clone()
        expected[0, 7, :9] = replacement * remaining_factor
        assert torch.equal(edited, expected)
    with ResidualHooks(blocks, site=1, replacement=trace.values[1]):
        assert torch.equal(run(blocks, x), baseline)
    with pytest.raises(ValueError, match="finite exact"):
        with ResidualHooks(blocks, site=1, replacement=torch.full_like(tile(x), float("nan"))):
            run(blocks, x)
    assert not any(block._forward_hooks or block._forward_pre_hooks for block in blocks)
    assert torch.equal(run(blocks, x), baseline)


def test_linear_attribution_matches_exact_output_change_all_sites():
    blocks = [LinearBlock(2), LinearBlock(3)]
    x = torch.randn(1, 8, 36, 16, 2, requires_grad=True)
    with ResidualHooks(blocks, capture=True, differentiable=True) as trace:
        output = run(blocks, x)
    objective = tile(output).sum()
    grads = torch.autograd.grad(objective, list(trace.values.values()))
    for site in range(3):
        original = tile(trace.values[site]).detach()
        donor = original + .25
        predicted_effect = ((donor-original)*tile(grads[site])).sum()
        with ResidualHooks(blocks, site=site, replacement=donor):
            edited = run(blocks, x.detach())
        assert float(tile(edited).sum()-objective.detach()) == pytest.approx(float(predicted_effect), abs=.01)


def probe(width, outputs=15, seed=8):
    rng = np.random.default_rng(seed)
    return RidgeModel(.1, rng.normal(size=width), rng.uniform(.5, 2, width),
                      rng.normal(size=outputs), rng.uniform(.5, 2, outputs), rng.normal(size=(width, outputs))*.01)


def test_output_metric_has_correct_flow_endpoint_and_donor_direction():
    output_probe = probe(4608)
    latent = torch.randn(1, 8, 36, 16, 32)
    noise = torch.randn_like(latent)
    inputs = {"z_t": .5*latent+.5*noise, "tau": torch.full((1, 8, 1, 1, 1), .5)}
    prediction = (latent-noise).requires_grad_()
    estimate = endpoint(inputs, prediction)
    torch.testing.assert_close(estimate, latent)
    raw = tile(estimate).reshape(1, -1)
    actual = torch_predict(output_probe, raw)
    np.testing.assert_allclose(actual.detach().numpy(), output_probe.predict(raw.detach().numpy()), atol=1e-12, rtol=1e-12)
    target = float(actual[0, 2].detach())+10
    objective, _, _ = output_objective(inputs, prediction, output_probe, target)
    grad, = torch.autograd.grad(objective, prediction)
    assert torch.count_nonzero(grad[0, :7]) == 0
    assert torch.count_nonzero(grad[0, 7, 9:]) == 0
    step = grad / torch.linalg.vector_norm(grad) * .001
    assert float(output_objective(inputs, prediction+step, output_probe, target)[0].detach()) > float(objective.detach())


def test_spatial_pullback_controls_ablation_and_paired_randomness():
    torch.set_num_threads(4)
    rng = np.random.default_rng(17)
    projection = torch.from_numpy(rng.normal(size=(2048, 128)).astype(np.float32) / np.sqrt(2048))
    spatial_probe = probe(1536)
    recipient = torch.from_numpy(rng.normal(size=(9, 16, 2048)).astype(np.float32)).requires_grad_()
    donor = recipient.detach()+torch.from_numpy(rng.normal(size=recipient.shape).astype(np.float32))
    value = torch_predict(spatial_probe, tile_features(recipient, projection))[0, 2]
    automatic, = torch.autograd.grad(value, recipient)
    manual = spatial_pullback(spatial_probe, projection, 2)
    torch.testing.assert_close(manual, automatic, atol=2e-9, rtol=2e-5)
    edits, delta, details = intervention_tiles(recipient.detach(), donor, spatial_probe, projection, seed=72)
    repeated, _, _ = intervention_tiles(recipient.detach(), donor, spatial_probe, projection, seed=72)
    assert torch.equal(edits["norm_matched_random"], repeated["norm_matched_random"])
    assert torch.equal(edits["restore"], recipient)
    assert torch.equal(edits["exact_donor"], donor)
    ablated_value = float(torch_predict(spatial_probe, tile_features(edits["ablate_to_discovery_mean"], projection))[0, 2])
    assert ablated_value == pytest.approx(spatial_probe.y_mean[2], abs=1e-5)
    delta_norm = float(torch.linalg.vector_norm(delta))
    for condition in ["norm_matched_random", "wrong_variable_ball_x"]:
        assert float(torch.linalg.vector_norm(edits[condition]-recipient.detach())) == pytest.approx(delta_norm, rel=2e-5, abs=1e-5)
    for control, original in [("norm_matched_random_full", "exact_donor"), ("norm_matched_random_ablation", "ablate_to_discovery_mean")]:
        assert float(torch.linalg.vector_norm(edits[control]-recipient.detach())) == pytest.approx(float(torch.linalg.vector_norm(edits[original]-recipient.detach())), rel=2e-5, abs=1e-5)
    assert np.isfinite(details["height_x_direction_cosine"])


def test_pairs_are_metadata_only_and_refuse_confirmation_before_data_access():
    records = [{"clip_id": f"clip{i:02}", "match_id": "match", "split": "discovery", "artifact_path": "does-not-exist"} for i in reversed(range(8))]
    result = matched_pairs(records, {"match": {"split": "discovery"}})
    assert result[0]["recipient"]["clip_id"] == "clip00"
    assert result[0]["donor"]["clip_id"] == "clip07"
    records[-1]["split"] = "confirmation"
    with pytest.raises(ValueError, match="Confirmation"):
        matched_pairs(records, {"match": {"split": "discovery"}})
