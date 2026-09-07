"""Observational, match-weighted ridge probes. Decoding is not causal evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SPLITS = ("discovery", "selection", "confirmation")
MINIMUM_MATCHES = {"discovery": 30, "selection": 10, "confirmation": 10}


def match_weights(match_ids):
    """Weights sum to one, with equal total weight per independent match."""
    ids = np.asarray(match_ids).astype(str)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    if len(unique) == 0:
        raise ValueError("At least one independent match is required")
    return 1.0 / (len(unique) * counts[inverse])


def validate_dataset(data, *, expected_sites=17, expected_targets=30):
    required = ("X", "y", "match_ids", "split", "target_names", "sites", "codec_X")
    if any(name not in data for name in required):
        raise ValueError(f"Required arrays: {required}")
    x, y = data["X"], data["y"]
    if x.ndim != 3 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("Require X[N,sites,features], y[N,targets], and at least one row")
    if x.shape[1] != expected_sites or y.shape[1] != expected_targets:
        raise ValueError("Unexpected number of sites or targets")
    for name, length in (("match_ids", len(x)), ("split", len(x)),
                         ("target_names", expected_targets), ("sites", expected_sites)):
        if data[name].ndim != 1 or len(data[name]) != length:
            raise ValueError(f"Wrong shape for {name}")
    if len(set(map(str, data["sites"]))) != expected_sites or len(set(map(str, data["target_names"]))) != expected_targets:
        raise ValueError("Site and target names must be unique")
    for name in ("X", "y", "codec_X", "RGB_X"):
        if name in data:
            array = data[name]
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite and numeric")
            if name in ("codec_X", "RGB_X") and (array.ndim != 2 or len(array) != len(x) or not array.shape[1]):
                raise ValueError(f"{name} must be [N,features] on the identical example cohort")
    ids, splits = data["match_ids"].astype(str), data["split"].astype(str)
    if set(splits) - set(SPLITS):
        raise ValueError("Only discovery, selection, and confirmation rows are permitted")
    if any(not identifier for identifier in ids):
        raise ValueError("Match identifiers must not be empty")
    if any(len(set(splits[ids == identifier])) != 1 for identifier in np.unique(ids)):
        raise ValueError("A match cannot cross split boundaries")
    counts = {split: len(np.unique(ids[splits == split])) for split in SPLITS}
    return counts


def shuffle_match_labels(y, match_ids, *, seed):
    """Derange entire equally sized match label blocks; retain within-match order.

    A random ordering followed by a cyclic shift has no self-pairs. Label values,
    marginal distribution, and within-block temporal/view structure are preserved.
    """
    ids = np.asarray(match_ids).astype(str)
    groups = np.unique(ids)
    if len(groups) < 2:
        raise ValueError("Match-label control needs at least two discovery matches")
    rows = [np.flatnonzero(ids == group) for group in groups]
    if len(set(map(len, rows))) != 1:
        raise ValueError("Whole-match label shuffle requires equal discovery rows per match")
    order = np.random.default_rng(seed).permutation(len(groups))
    shuffled = np.empty_like(y)
    mapping = {}
    for recipient, donor in zip(order, np.roll(order, -1)):
        shuffled[rows[recipient]] = y[rows[donor]]
        mapping[str(groups[recipient])] = str(groups[donor])
    return shuffled, mapping


@dataclass
class RidgeModel:
    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coefficient: np.ndarray

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.x_mean) or not np.isfinite(x).all():
            raise ValueError("Prediction features have incompatible shape or nonfinite values")
        return (((x - self.x_mean) / self.x_scale) @ self.coefficient) * self.y_scale + self.y_mean

    def outputs(self, indices):
        return RidgeModel(self.alpha, self.x_mean, self.x_scale, self.y_mean[indices],
                          self.y_scale[indices], self.coefficient[:, indices])

    def fingerprint(self):
        digest = hashlib.sha256(str(self.alpha).encode())
        for value in (self.x_mean, self.x_scale, self.y_mean, self.y_scale, self.coefficient):
            array = np.ascontiguousarray(value, dtype="<f8")
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def ridge_path(x, y, weights, *, alphas=ALPHAS):
    """Fit discovery-only normalization and all alphas using one eigendecomposition.

    Minimize sum_i w_i ||standardized_y_i - standardized_X_i B||^2 + alpha ||B||^2.
    The intercept is unpenalized; weights must sum to one. Constant columns use scale1.
    """
    x, y, weights = (np.asarray(a, dtype=np.float64) for a in (x, y, weights))
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or weights.shape != (len(x),):
        raise ValueError("Invalid discovery feature, target, or weight dimensions")
    if not all(np.isfinite(a).all() for a in (x, y, weights)) or not (weights > 0).all():
        raise ValueError("Discovery inputs must be finite and weights positive")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Ridge weights must sum to one")
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge alphas must be positive and finite")
    x_mean, y_mean = weights @ x, weights @ y
    x_centered, y_centered = x - x_mean, y - y_mean
    x_scale = np.sqrt(weights @ (x_centered * x_centered))
    y_scale = np.sqrt(weights @ (y_centered * y_centered))
    # Weighted sums can leave roundoff-sized variance even for exactly constant
    # columns. Do not turn that roundoff into a unit-sized feature or target.
    tolerance = np.finfo(float).eps * 16
    x_scale = np.where(x_scale > tolerance * np.maximum(1, np.max(np.abs(x), axis=0)), x_scale, 1.0)
    y_scale = np.where(y_scale > tolerance * np.maximum(1, np.max(np.abs(y), axis=0)), y_scale, 1.0)
    sqrt_w = np.sqrt(weights[:, None])
    xw, yw = (x_centered / x_scale) * sqrt_w, (y_centered / y_scale) * sqrt_w
    if x.shape[1] <= x.shape[0]:
        eigenvalues, basis = np.linalg.eigh(xw.T @ xw)
        projected = basis.T @ (xw.T @ yw)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(xw @ xw.T)
        basis = xw.T @ eigenvectors
        projected = eigenvectors.T @ yw
    eigenvalues = np.maximum(eigenvalues, 0)
    return [RidgeModel(float(alpha), x_mean, x_scale, y_mean, y_scale,
                       basis @ (projected / (eigenvalues[:, None] + alpha))) for alpha in alphas]


def normalized_mse(y, prediction, match_ids, discovery_scale):
    error = ((prediction - y) / discovery_scale) ** 2
    return float(match_weights(match_ids) @ error.mean(axis=1))


@dataclass
class SelectedSite:
    name: str
    kind: str
    main: RidgeModel
    shuffled: RidgeModel
    curve: list[dict]


def select_models(data, *, seed=20260906, progress=None):
    """Fit on discovery and choose alphas/sites on selection. Never index confirmation."""
    split = data["split"].astype(str)
    discovery, selection = split == "discovery", split == "selection"
    ids = data["match_ids"].astype(str)
    y_train, y_select = data["y"][discovery], data["y"][selection]
    y_control, permutation = shuffle_match_labels(y_train, ids[discovery], seed=seed)
    n_targets = y_train.shape[1]
    joined_targets = np.concatenate((y_train, y_control), axis=1)
    weights = match_weights(ids[discovery])
    selected = []
    inputs = [(str(name), "residual", index) for index, name in enumerate(data["sites"])]
    inputs += [(name, "baseline", name) for name in ("codec_X", "RGB_X") if name in data]
    for name, kind, key in inputs:
        if progress:
            progress(name)
        features = data["X"][:, key, :] if kind == "residual" else data[key]
        models = ridge_path(features[discovery], joined_targets, weights)
        curve = []
        for model in models:
            prediction = model.predict(features[selection])
            curve.append({"alpha": model.alpha,
                          "main_normalized_mse": normalized_mse(y_select, prediction[:, :n_targets], ids[selection], model.y_scale[:n_targets]),
                          "shuffle_normalized_mse": normalized_mse(y_select, prediction[:, n_targets:], ids[selection], model.y_scale[n_targets:])})
        main_index = min(range(len(models)), key=lambda index: curve[index]["main_normalized_mse"])
        shuffled_index = min(range(len(models)), key=lambda index: curve[index]["shuffle_normalized_mse"])
        selected.append(SelectedSite(name, kind, models[main_index].outputs(slice(0, n_targets)),
                                     models[shuffled_index].outputs(slice(n_targets, None)), curve))
    residual = [site for site in selected if site.kind == "residual"]
    winner = min(residual, key=lambda site: min(row["main_normalized_mse"] for row in site.curve)).name
    frozen = {
        "selection_rule": "minimum equal-match normalized selection MSE averaged over all targets; ties use smaller alpha then site order",
        "alpha_grid": list(ALPHAS), "winner": winner,
        "label_shuffle_seed": seed, "label_shuffle_donor_by_recipient_match": permutation,
        "models": [{"name": site.name, "kind": site.kind, "selected_alpha": site.main.alpha,
                    "shuffle_alpha": site.shuffled.alpha, "model_sha256": site.main.fingerprint(),
                    "shuffle_model_sha256": site.shuffled.fingerprint(), "selection_curve": site.curve} for site in selected],
    }
    return selected, frozen


def match_bootstrap_metrics(y, prediction, match_ids, discovery_scale, *, seed=20260906, replicates=500):
    """Equal-match MAE/R2 and match-resampling percentile CIs; rows are not independent."""
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    ids = np.asarray(match_ids).astype(str)
    if y.shape != prediction.shape or y.ndim != 2 or len(ids) != len(y):
        raise ValueError("Metric arrays do not align")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric arrays must be finite")
    groups = np.unique(ids)
    per_match = []
    for group in groups:
        actual, predicted = y[ids == group], prediction[ids == group]
        per_match.append(np.stack(((actual - predicted).__abs__().mean(0),
                                   ((actual - predicted) ** 2).mean(0), actual.mean(0), (actual ** 2).mean(0))))
    statistics = np.asarray(per_match)

    def aggregate(stats):
        mean = stats.mean(axis=-3)
        mae, mse, target_mean, target_second = (mean[..., index, :] for index in range(4))
        variance = np.maximum(0, target_second - target_mean ** 2)
        tolerance = np.finfo(float).eps * np.maximum(1, target_second) * 16
        r2 = np.full_like(mse, np.nan)
        np.divide(mse, variance, out=r2, where=variance > tolerance)
        r2 = 1 - r2
        nmse = (mse / np.asarray(discovery_scale) ** 2).mean(-1)
        return mae, np.sqrt(mse), r2, nmse

    mae, rmse, r2, nmse = aggregate(statistics)
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(replicates, len(groups)))
    bootstrap_mae, bootstrap_rmse, bootstrap_r2, bootstrap_nmse = aggregate(statistics[draws])

    def interval(values):
        values = np.asarray(values)
        if values.ndim == 1:
            finite = values[np.isfinite(values)]
            return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None]
        return [interval(values[:, index]) for index in range(values.shape[1])]

    return {"n_matches": len(groups), "n_examples": len(y), "bootstrap_unit": "whole_match",
            "bootstrap_replicates": replicates, "bootstrap_seed": seed,
            "normalized_mse": float(nmse), "normalized_mse_ci95": interval(bootstrap_nmse),
            "mae": mae.tolist(), "mae_ci95": interval(bootstrap_mae),
            "rmse": rmse.tolist(), "rmse_ci95": interval(bootstrap_rmse),
            "r2": [float(value) if np.isfinite(value) else None for value in r2],
            "r2_ci95": interval(bootstrap_r2),
            "r2_bootstrap_valid_replicates": np.isfinite(bootstrap_r2).sum(0).tolist(),
            "uncertainty_note": "Intervals resample observed confirmation matches; no model retraining, seed uncertainty, or population guarantee."}


def save_selected_models(selected, path):
    arrays = {}
    for index, site in enumerate(selected):
        for kind, model in (("main", site.main), ("shuffled", site.shuffled)):
            for name in ("alpha", "x_mean", "x_scale", "y_mean", "y_scale", "coefficient"):
                arrays[f"site{index}_{kind}_{name}"] = np.asarray(getattr(model, name))
    np.savez_compressed(path, **arrays)


def paired_mse_gain(y, prediction, baseline, match_ids, discovery_scale, *, seed=20260906, replicates=500):
    """Baseline minus model standardized MSE, bootstrapping paired complete matches."""
    y, prediction, baseline = (np.asarray(value, dtype=np.float64) for value in (y, prediction, baseline))
    if y.shape != prediction.shape or y.shape != baseline.shape:
        raise ValueError("Paired comparison arrays must have identical shapes")
    ids = np.asarray(match_ids).astype(str)
    row_gain = (((baseline - y) / discovery_scale) ** 2 - ((prediction - y) / discovery_scale) ** 2).mean(1)
    gain = np.array([row_gain[ids == match].mean() for match in np.unique(ids)])
    draws = np.random.default_rng(seed).integers(0, len(gain), size=(replicates, len(gain)))
    return {"normalized_mse_gain": float(gain.mean()),
            "ci95": np.quantile(gain[draws].mean(1), [0.025, 0.975]).tolist(),
            "positive_means": "lower error than baseline", "n_matches": len(gain),
            "bootstrap_unit": "paired_whole_match", "bootstrap_replicates": replicates,
            "bootstrap_seed": seed}


def load_selected_models(path, frozen):
    selected = []
    with np.load(path, allow_pickle=False) as arrays:
        for index, entry in enumerate(frozen["models"]):
            models = []
            for kind, hash_key in (("main", "model_sha256"), ("shuffled", "shuffle_model_sha256")):
                values = {name: arrays[f"site{index}_{kind}_{name}"] for name in
                          ("alpha", "x_mean", "x_scale", "y_mean", "y_scale", "coefficient")}
                values["alpha"] = float(values["alpha"])
                model = RidgeModel(**values)
                if model.fingerprint() != entry[hash_key]:
                    raise ValueError("Reloaded frozen model does not match its selection fingerprint")
                models.append(model)
            selected.append(SelectedSite(entry["name"], entry["kind"], *models, entry["selection_curve"]))
    return selected
