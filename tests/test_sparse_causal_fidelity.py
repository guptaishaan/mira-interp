"""Analytic fixtures, no research observations or GPU access."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mira_interp.feature_geometry import TopKDictionary
from mira_interp.sparse_causal_fidelity import (load_dictionary, reconstruct_descriptor,
    reconstruct_tile, spatial_descriptor, spatial_lift)


def fixture():
    torch.manual_seed(811)
    q, _ = torch.linalg.qr(torch.randn(16, 5, dtype=torch.float64))
    native = torch.randn(9, 16, 16, dtype=torch.float64)
    return native, q


def test_lift_roundtrip_has_no_bin_area_division_and_exact_noop():
    native, q = fixture()
    delta = torch.randn(1, 60, dtype=torch.float64)
    lifted = spatial_lift(delta, q)
    torch.testing.assert_close(spatial_descriptor(lifted, q), delta, atol=2e-15, rtol=2e-15)
    assert torch.equal(reconstruct_tile(native, spatial_descriptor(native, q), q), native)
    changed = reconstruct_tile(native, spatial_descriptor(native, q) + delta, q)
    torch.testing.assert_close(spatial_descriptor(changed, q), spatial_descriptor(native, q) + delta)


def test_lift_preserves_nullspace_and_is_minimum_norm():
    native, q = fixture()
    nullspace = native - spatial_lift(spatial_descriptor(native, q), q)
    torch.testing.assert_close(spatial_descriptor(nullspace, q), torch.zeros(1, 60, dtype=torch.float64), atol=1e-15, rtol=0)
    target = torch.randn(1, 60, dtype=torch.float64)
    reconstructed = reconstruct_tile(native, target, q)
    torch.testing.assert_close(reconstructed - spatial_lift(spatial_descriptor(reconstructed, q), q), nullspace)
    lift = spatial_lift(target, q)
    assert abs(float((lift * nullspace).sum())) < 1e-13
    assert float((lift + nullspace).square().sum()) > float(lift.square().sum())
    torch.testing.assert_close(lift.square().sum(), 12 * target.square().sum())


def test_discovery_mean_controls_remove_descriptor_difference_but_retain_native_nullspaces():
    recipient, q = fixture()
    donor = recipient + torch.randn_like(recipient)
    mean = torch.randn(1, 60, dtype=torch.float64)
    controlled_recipient = reconstruct_tile(recipient, mean, q)
    controlled_donor = reconstruct_tile(donor, mean, q)
    torch.testing.assert_close(spatial_descriptor(controlled_recipient, q), mean)
    torch.testing.assert_close(spatial_descriptor(controlled_donor, q), mean)
    native_difference = donor-recipient
    nullspace_difference = native_difference-spatial_lift(spatial_descriptor(native_difference, q), q)
    torch.testing.assert_close(controlled_donor-controlled_recipient, nullspace_difference)
    assert float(torch.linalg.vector_norm(controlled_donor-controlled_recipient)) > 0


def test_explicit_spatial_slices_and_invalid_shapes():
    native, q = fixture()
    expected = torch.cat([native[y*3:(y+1)*3, x*4:(x+1)*4].mean((0, 1)) @ q
                          for y in range(3) for x in range(4)])[None]
    torch.testing.assert_close(spatial_descriptor(native, q), expected)
    with pytest.raises(ValueError, match="12 projected bins"):
        spatial_lift(torch.zeros(59), q)
    with pytest.raises(ValueError, match="finite"):
        spatial_descriptor(native * float("nan"), q)


def save_dictionary(path, *, malformed=None):
    model = TopKDictionary(12, width=24, active=8, kind="block", group_size=8)
    mean, scale = torch.arange(12, dtype=torch.float64), torch.full((12,), 3., dtype=torch.float64)
    if malformed == "whitening":
        scale[1] = 4
    report = {"input_dimension": 12, "latent_scalar_width": 24, "active_scalar_budget": 8,
              "kind": "block", "group_size": 8, "selection_used_for_updates": False,
              "normalizer_sha256": hashlib.sha256(mean.numpy().tobytes() + scale.numpy().tobytes()).hexdigest()}
    state = model.state_dict()
    if malformed == "missing_state":
        del state["encoder.bias"]
    if malformed == "wrong_dtype":
        state["encoder.weight"] = state["encoder.weight"].double()
    torch.save({"state_dict": state, "normalizer": {"mean": mean, "scale": scale}, "report": report}, path)
    return model, mean, scale


def test_checkpoint_strict_load_and_training_normalization(tmp_path):
    path = tmp_path / "dictionary.pt"
    original, mean, scale = save_dictionary(path)
    bundle = load_dictionary(path, dimension=12, width=24)
    descriptor = torch.randn(2, 12)
    with torch.no_grad():
        expected, code = original(((descriptor.double() - mean) / scale).float())
    actual, actual_code = reconstruct_descriptor(bundle, descriptor)
    assert torch.equal(actual_code, code)
    assert torch.equal(actual, (expected.double() * scale + mean).float())
    save_dictionary(path, malformed="whitening")
    with pytest.raises(ValueError, match="global RMS"):
        load_dictionary(path, dimension=12, width=24)
    save_dictionary(path, malformed="missing_state")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_dictionary(path, dimension=12, width=24)
    save_dictionary(path, malformed="wrong_dtype")
    with pytest.raises(ValueError, match="before loading"):
        load_dictionary(path, dimension=12, width=24)


def evaluator():
    path = Path(__file__).resolve().parents[1] / "scripts/sparse_causal_fidelity.py"
    spec = importlib.util.spec_from_file_location("sparse_fidelity_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_report_rejected_before_any_checkpoint_access(tmp_path, monkeypatch):
    runner = evaluator()
    paths = []
    for seed in range(3):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps({"status": "passed_development_feature_analysis", "scope": "development_only",
            "descriptor_identity": "spatial/block_15_output", "dimension": 1536, "temporal_readout": "current",
            "confirmation_rows_loaded": True}))
        paths.append(path)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "passed", "all_reconstruction_metrics_recomputed": True,
                               "feature_reports": [runner.binding(path) for path in paths]}))
    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint should not be opened after the report scope fails")
    monkeypatch.setattr(runner, "load_dictionary", forbidden)
    with pytest.raises(ValueError, match="scope or descriptor"):
        runner.feature_inputs(SimpleNamespace(feature_report=paths, feature_audit=audit))


def test_uncertainty_averages_seeds_before_resampling_independent_matches():
    runner = evaluator()
    metrics = ["standardized_error_gain", "gain_difference_from_native_reference", "flow_distance_from_native_reference_l2",
               "flow_mse_from_native_reference", "proxy_distance_from_native_reference_standardized_rms",
               "descriptor_reconstruction_rmse", "endpoint_recipient_source_mse"]
    rows = [{"match_id": f"match{match:02}", "conditions": [{"dictionary_id": "fixed", "condition": "donor_reconstruction",
             **{metric: float(match + seed * 100) for metric in metrics}}]} for match in range(11) for seed in range(2)]
    result = runner.matched_summary(rows, "fixed", "donor_reconstruction")
    expected = np.arange(11) + 50
    draws = np.random.default_rng(20260907).integers(0, 11, (500, 11))
    for metric in metrics:
        assert result["metrics"][metric]["mean"] == 55
        np.testing.assert_array_equal(list(result["metrics"][metric]["match_values"].values()), expected)
        np.testing.assert_allclose(result["metrics"][metric]["match_bootstrap_ci95"], np.quantile(expected[draws].mean(1), [.025, .975]))
    with pytest.raises(ValueError, match="Incomplete fidelity pairs"):
        runner.matched_summary(rows[:-1], "fixed", "donor_reconstruction")


def test_mean_comparison_distinguishes_objective_improvement_from_native_fidelity():
    runner = evaluator()
    rows = [{"match_id": f"match{match:02}", "conditions": [
        {"dictionary_id": "fixed", "condition": "donor_reconstruction", "standardized_error_gain": 6.5,
         "gain_difference_from_native_reference": 1.5},
        {"dictionary_id": "discovery_mean", "condition": "donor_reconstruction", "standardized_error_gain": 4.,
         "gain_difference_from_native_reference": -1.}]} for match in range(11) for _ in range(2)]
    result = runner.paired_mean_summary(rows, "fixed", "donor_reconstruction")["metrics"]
    assert result["donor_objective_gain_above_mean_control"]["mean"] == 2.5
    assert result["absolute_native_gain_error_improvement_over_mean_control"]["mean"] == -.5
