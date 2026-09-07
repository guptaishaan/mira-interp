"""Synthetic mathematical tests only; no toy probe curves are research results."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mira_interp.probes import (
    load_selected_models, match_bootstrap_metrics, match_weights, paired_mse_gain, ridge_path,
    save_selected_models, select_models, shuffle_match_labels, validate_dataset,
)


@pytest.mark.parametrize("shape", [(20, 3), (5, 9)])
def test_weighted_ridge_matches_direct_linear_system_in_primal_and_dual(shape):
    rng = np.random.default_rng(7)
    x = rng.normal(size=shape)
    x[:, -1] = 6.0  # constant feature must not produce a spurious coefficient
    y = np.column_stack((x[:, 0] * 2.0 + 4.0, np.full(len(x), 11.0)))
    weights = np.full(len(x), 1.0 / len(x))
    model = ridge_path(x, y, weights, alphas=(0.1,))[0]
    standardized_x = (x - model.x_mean) / model.x_scale
    standardized_y = (y - model.y_mean) / model.y_scale
    expected = np.linalg.solve(standardized_x.T @ (weights[:, None] * standardized_x) + 0.1 * np.eye(x.shape[1]),
                               standardized_x.T @ (weights[:, None] * standardized_y))
    np.testing.assert_allclose(model.coefficient, expected, atol=1e-12)
    np.testing.assert_allclose(model.predict(x)[:, 1], 11.0, atol=1e-12)
    assert model.x_scale[-1] == 1.0
    np.testing.assert_allclose(model.coefficient[-1], 0.0, atol=1e-12)


def test_match_weighting_is_invariant_to_duplicating_repeated_rows_within_one_match():
    x = np.array([[0.], [1.], [4.], [5.]])
    y = np.array([[1.], [3.], [6.], [9.]])
    matches = np.array(["a", "a", "b", "b"])
    original = ridge_path(x, y, match_weights(matches), alphas=(1.,))[0]
    rows = [0, 1, 0, 1, 2, 3]
    duplicated = ridge_path(x[rows], y[rows], match_weights(matches[rows]), alphas=(1.,))[0]
    np.testing.assert_allclose(original.predict(x), duplicated.predict(x), atol=1e-12)


def small_dataset():
    rng = np.random.default_rng(81)
    ids = np.repeat([f"match-{i}" for i in range(8)], 4)
    split = np.repeat(["discovery"] * 4 + ["selection"] * 2 + ["confirmation"] * 2, 4)
    x = rng.normal(size=(32, 2, 3))
    y = np.column_stack((x[:, 0, 0] * 2 + 1, x[:, 0, 1] * -3))
    return {"X": x, "y": y, "match_ids": ids, "split": split,
            "sites": np.array(["input", "output"]), "target_names": np.array(["position", "velocity"]),
            "codec_X": rng.normal(size=(32, 2))}


def test_confirmation_values_cannot_affect_normalization_model_selection_or_hashes(tmp_path):
    original = small_dataset()
    changed = copy.deepcopy(original)
    confirm = changed["split"] == "confirmation"
    changed["X"][confirm] += 100000
    changed["y"][confirm] -= 99999
    changed["codec_X"][confirm] *= -100
    models, frozen = select_models(original)
    changed_models, changed_frozen = select_models(changed)
    assert frozen == changed_frozen
    for site, other in zip(models, changed_models):
        assert site.main.fingerprint() == other.main.fingerprint()
        np.testing.assert_allclose(site.main.y_mean, original["y"][original["split"] == "discovery"].mean(0))
    destination = tmp_path / "models.npz"
    save_selected_models(models, destination)
    reloaded = load_selected_models(destination, frozen)
    for site, loaded in zip(models, reloaded):
        assert site.main.fingerprint() == loaded.main.fingerprint()
    frozen["models"][0]["model_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="fingerprint"):
        load_selected_models(destination, frozen)


def test_whole_match_shuffle_deranges_blocks_and_preserves_order_and_label_marginals():
    ids = np.repeat(["a", "b", "c", "d"], 3)
    y = np.arange(24).reshape(12, 2)
    shuffled, mapping = shuffle_match_labels(y, ids, seed=4)
    assert all(recipient != donor for recipient, donor in mapping.items())
    for recipient, donor in mapping.items():
        np.testing.assert_equal(shuffled[ids == recipient], y[ids == donor])
    np.testing.assert_equal(np.sort(shuffled, axis=0), np.sort(y, axis=0))
    with pytest.raises(ValueError, match="equal discovery rows"):
        shuffle_match_labels(y[:-1], ids[:-1], seed=4)


def test_match_bootstrap_has_raw_unit_metrics_constant_r2_guard_and_not_frame_count():
    y = np.array([[0., 4.], [2., 4.], [8., 4.], [10., 4.]])
    prediction = y + np.array([1., 2.])
    report = match_bootstrap_metrics(y, prediction, ["a", "a", "b", "b"], [1., 1.], replicates=30)
    np.testing.assert_allclose(report["mae"], [1., 2.])
    np.testing.assert_allclose(report["rmse"], [1., 2.])
    assert report["r2"][0] == pytest.approx(1 - 1 / 17)
    assert report["r2"][1] is None
    assert report["r2_bootstrap_valid_replicates"][1] == 0
    assert report["n_matches"] == 2 and report["n_examples"] == 4
    assert report["normalized_mse_ci95"] == [2.5, 2.5]
    comparison = paired_mse_gain(y, prediction, y + [2., 4.], ["a", "a", "b", "b"], [1., 1.], replicates=30)
    assert comparison["normalized_mse_gain"] == 7.5
    assert comparison["ci95"] == [7.5, 7.5]


def test_validation_rejects_match_leakage_and_missing_baselines():
    data = small_dataset()
    assert validate_dataset(data, expected_sites=2, expected_targets=2) == {"discovery": 4, "selection": 2, "confirmation": 2}
    data["match_ids"][-1] = data["match_ids"][0]
    with pytest.raises(ValueError, match="split boundaries"):
        validate_dataset(data, expected_sites=2, expected_targets=2)
    del data["codec_X"]
    with pytest.raises(ValueError, match="Required arrays"):
        validate_dataset(data, expected_sites=2, expected_targets=2)


@pytest.mark.parametrize("version", [1, 2])
def test_audit_gate_binds_each_approved_protocol_to_its_actual_capture(version):
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_probes.py"
    spec = importlib.util.spec_from_file_location("probe_analysis_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol_hash = next(key for key, value in module.APPROVED_REGISTRATIONS.items() if value == version)
    other_hash = next(key for key, value in module.APPROVED_REGISTRATIONS.items() if value != version)
    audit = {"status": "passed", "analysis_npz_sha256": "data-digest", "registration_sha256": protocol_hash,
             "real_data": True, "video_alignment_checked": True, "physics_alignment_checked": True,
             "all_layers_complete": True, "match_splits_disjoint": True, "checkpoint_integrity_checked": True}
    module.validate_audit(audit, "data-digest", protocol_hash)
    with pytest.raises(ValueError, match="selected frozen"):
        module.validate_audit(audit, "data-digest", other_hash)
    with pytest.raises(ValueError, match="exact NPZ"):
        module.validate_audit(audit, "changed-data", protocol_hash)
    with pytest.raises(ValueError, match="Unknown or changed"):
        module.validate_audit(audit, "data-digest", "unknown-protocol")


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_audit_stream_skips_excluded_confirmation_values(tmp_path):
    module = load_script("audit_probe_selection.py")
    keep = np.array([False, True, True, False, True, False])
    targets = np.arange(180, dtype=np.float64).reshape(6, 30)
    targets[~keep] = np.nan
    path = tmp_path / "temporary_engineering_fixture.npz"
    np.savez_compressed(path, y=targets)
    np.testing.assert_array_equal(module.selected_target_rows(path, keep), targets[keep])


def test_selection_audit_rejects_wrong_discovery_solution_and_normalization():
    module = load_script("audit_probe_selection.py")
    rng = np.random.default_rng(42)
    features = rng.normal(size=(18, 4))
    targets = rng.normal(size=(18, 30))
    weights = np.ones(18) / 18
    fitted = ridge_path(features, targets, weights, alphas=[0.1])[0]
    model = {field: np.asarray(getattr(fitted, field)) for field in ("alpha", *module.FIELDS)}
    valid = module.check_model(model, features, targets, features, targets, weights, np.repeat(["a", "b", "c"], 6))
    assert valid["normal_equation_max_abs_residual"] < 1e-10
    assert module.fingerprint(model) == fitted.fingerprint()
    wrong = copy.deepcopy(model)
    wrong["coefficient"][0, 0] += 0.01
    with pytest.raises(ValueError, match="solve discovery ridge"):
        module.check_model(wrong, features, targets, features, targets, weights, np.repeat(["a", "b", "c"], 6))
    wrong = copy.deepcopy(model)
    wrong["y_mean"][0] += 0.01
    with pytest.raises(ValueError, match="discovery-only"):
        module.check_model(wrong, features, targets, features, targets, weights, np.repeat(["a", "b", "c"], 6))


def test_confirmation_gate_requires_matching_independent_selection_audit():
    module = load_script("analyze_probes.py")
    frozen = {key: key + "-digest" for key in ("input_npz_sha256", "registration_sha256", "capture_audit_sha256", "model_archive_sha256")}
    frozen.update(analysis_code_sha256={"code": "digest"}, match_counts={"discovery": 31, "selection": 11, "confirmation": 11}, winner="block_0_input")
    audit = {**frozen, "status": "passed", "frozen_selection_sha256": "frozen-digest", "audit_script_sha256": "helper-digest",
             "frozen_winner": frozen["winner"], "confirmation_target_values_used": False,
             "models_refitted": False, "ridge_normal_equations_checked": True}
    module.validate_selection_audit(audit, frozen, "frozen-digest", "helper-digest")
    for key, changed in (("status", "running"), ("model_archive_sha256", "changed"), ("models_refitted", True),
                         ("confirmation_target_values_used", True), ("audit_script_sha256", "changed")):
        with pytest.raises(ValueError):
            module.validate_selection_audit({**audit, key: changed}, frozen, "frozen-digest", "helper-digest")
