"""Temporary mathematical fixtures, never developmental research results."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mira_interp.development import (causal_row_indices, fit_readout, group_indices,
    residual_winners, role_targets, target_definitions, temporal_features)
from mira_interp.probes import load_selected_models, save_selected_models


def test_roles_select_the_observing_player_and_subtract_in_world_axes():
    entities = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    views = np.arange(4)
    y = entities.reshape(4, 30)
    result = role_targets(y, views)
    for row, view in enumerate(views):
        np.testing.assert_array_equal(result[row, :6], entities[row, view + 1])
        np.testing.assert_array_equal(result[row, 6:], entities[row, 0] - entities[row, view + 1])
    # Jointly relabeling canonical players and the observing slot changes nothing.
    permutation = np.array([2, 0, 3, 1])
    renamed = np.concatenate((entities[:, :1], entities[:, permutation + 1]), axis=1)
    renamed_views = np.argsort(permutation)[views]
    np.testing.assert_array_equal(role_targets(renamed.reshape(4, 30), renamed_views), result)
    with pytest.raises(ValueError, match="canonical view"):
        role_targets(y, np.array([0, 1, 2, 4]))


def temporal_fixture():
    clip = np.repeat(["clip-a", "clip-b"], 4 * 8)
    view = np.tile(np.repeat(np.arange(4), 8), 2)
    time = np.tile(np.arange(8), 8)
    x = np.column_stack((np.repeat([100., 1000.], 32) + view * 10 + time, time ** 2))
    return clip, view, time, x


def test_temporal_difference_is_causal_and_never_crosses_view_or_clip():
    clips, views, times, x = temporal_fixture()
    # Exercise identity lookup rather than relying on array adjacency.
    order = np.random.default_rng(6).permutation(len(clips))
    clips, views, times, x = (a[order] for a in (clips, views, times, x))
    current, previous = causal_row_indices(clips, views, times)
    assert len(current) == 2 * 4 * 6
    assert np.all(times[current] >= 2)
    np.testing.assert_array_equal(clips[current], clips[previous])
    np.testing.assert_array_equal(views[current], views[previous])
    np.testing.assert_array_equal(times[current] - times[previous], 1)
    result = temporal_features(x, current, previous)
    np.testing.assert_array_equal(result[:, :2], x[current])
    np.testing.assert_array_equal(result[:, 2], 1)
    np.testing.assert_array_equal(result[:, 3], 2 * times[current] - 1)
    chosen = int(np.flatnonzero((clips[current] == "clip-a") & (views[current] == 0) & (times[current] == 2))[0])
    changed = x.copy()
    changed[(clips == "clip-a") & (views == 0) & (times > 2)] += 999
    np.testing.assert_array_equal(temporal_features(changed, current, previous)[chosen], result[chosen])
    with pytest.raises(ValueError, match="unique view/time"):
        causal_row_indices(clips[:-1], views[:-1], times[:-1])


def fit_fixture():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(24, 5))
    ids = np.repeat(["a", "b", "c", "d", "e", "f"], 4)
    roles = np.repeat(["discovery"] * 4 + ["selection"] * 2, 4)
    absolute = rng.normal(size=(24, 30)) + x[:, :1] * 2
    views = np.tile(np.arange(4), 6)
    targets = np.concatenate((absolute, role_targets(absolute, views)), axis=1)
    definitions = target_definitions([f"state{i}" for i in range(30)])
    return x, targets, ids, roles, definitions


def test_development_fit_rejects_confirmation_before_processing_target_values():
    x, targets, ids, roles, definitions = fit_fixture()
    roles = roles.astype("U20")
    roles[-4:] = "confirmation"
    targets[-4:] = np.nan
    with pytest.raises(ValueError, match="confirmation is forbidden"):
        fit_readout(x, targets, ids, roles, definitions, name="test", kind="residual", feature_spec={})


def test_grouped_models_use_discovery_normalization_and_preserve_frozen_coefficients(tmp_path):
    x, targets, ids, roles, definitions = fit_fixture()
    fit = fit_readout(x, targets, ids, roles, definitions, name="mean/input", kind="residual", feature_spec={"temporal": "current"}, alphas=(0.1, 1.))
    assert len(fit.models) == 6
    assert set(fit.report["targets"]) == {"absolute30", "role12"}
    for record, model in zip(fit.model_records, fit.models):
        indices = record["joined_target_indices"]
        np.testing.assert_allclose(model.main.y_mean, targets[roles == "discovery"][:, indices].mean(0))
        np.testing.assert_allclose(model.main.x_mean, x[roles == "discovery"].mean(0))
        curve = record["selection_curve"]
        assert model.main.alpha == min(curve, key=lambda row: row["main_normalized_mse"])["alpha"]
        assert len(record["target_names"]) == len(indices)
    assert len(group_indices(12)["velocity"]) == 6
    assert len(group_indices(30)["position"]) == 15
    path = tmp_path / "temporary_coefficients.npz"
    save_selected_models(fit.models, path)
    reloaded = load_selected_models(path, {"models": fit.model_records})
    assert [m.main.fingerprint() for m in reloaded] == [m.main.fingerprint() for m in fit.models]
    # With alpha fixed, changing selection inputs cannot alter discovery coefficients.
    changed = targets.copy()
    changed[roles == "selection"] += 10000
    first = fit_readout(x, targets, ids, roles, definitions, name="test", kind="residual", feature_spec={}, alphas=(1.,))
    second = fit_readout(x, changed, ids, roles, definitions, name="test", kind="residual", feature_spec={}, alphas=(1.,))
    assert [m.main.fingerprint() for m in first.models] == [m.main.fingerprint() for m in second.models]
    winners = residual_winners([fit.report, {**fit.report, "name": "identical-second"}])
    assert winners["role12"]["all"] == "mean/input"  # deterministic first-on-tie


def test_cli_manifest_rejects_confirmation_before_opening_any_npz(tmp_path):
    import json
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_development.py"
    spec = importlib.util.spec_from_file_location("development_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registration = tmp_path / "protocol.json"
    registration.write_text(json.dumps({"analysis": {"ridge_alphas": list(module.ALPHAS), "scored_latents": list(range(2, 8))}}))
    split = tmp_path / "split.json"
    assignments = [{"match_id": f"d{i}", "split": "discovery"} for i in range(31)] + [{"match_id": f"s{i}", "split": "selection"} for i in range(11)]
    split.write_text(json.dumps({"matches": assignments}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "passed", "confirmation_used": False, "registration_sha256": module.sha(registration),
         "records": [{"match_id": "forbidden", "clip_id": "forbidden", "split": "confirmation", "path": "not-opened.npz"}]}))
    with pytest.raises(ValueError, match="Confirmation/pilot capture is forbidden"):
        module.load_captures(manifest, manifest, registration, split)
