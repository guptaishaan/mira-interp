"""Development-only probe comparisons; never consume a confirmation partition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .probes import RidgeModel, SelectedSite, match_weights, ridge_path, shuffle_match_labels

ALPHAS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
SEED = 20260907
ROLE_TARGET_NAMES = tuple(f"{entity}.{field}.{axis}" for entity in ("ego", "ball_minus_ego")
                          for field in ("location", "velocity") for axis in "xyz")


def check_development_roles(match_ids, split):
    ids, roles = np.asarray(match_ids).astype(str), np.asarray(split).astype(str)
    if ids.ndim != 1 or ids.shape != roles.shape or not len(ids):
        raise ValueError("Invalid development row metadata")
    if set(roles) != {"discovery", "selection"}:
        raise ValueError("Only discovery and selection are permitted; confirmation is forbidden")
    if any(len(set(roles[ids == match])) != 1 for match in np.unique(ids)):
        raise ValueError("A match cannot cross development partitions")
    return ids, roles


def role_targets(y, view_index):
    """Ego state and ball-minus-ego state, in relative WORLD axes, not camera axes."""
    y, views = np.asarray(y, dtype=np.float64), np.asarray(view_index)
    if y.ndim != 2 or y.shape[1] != 30 or views.shape != (len(y),):
        raise ValueError("Expected y[N,30] and view_index[N]")
    if views.dtype.kind not in "iu" or np.any((views < 0) | (views > 3)) or not np.isfinite(y).all():
        raise ValueError("Invalid canonical view indices or targets")
    entities = y.reshape(-1, 5, 6)
    ego = entities[np.arange(len(y)), views + 1]
    return np.concatenate((ego, entities[:, 0] - ego), axis=1)


def causal_row_indices(clip_ids, view_index, latent_frame_index, *, n_latents=8, first=2):
    """Locate current and previous rows without crossing a clip or camera view."""
    clips, views, times = map(np.asarray, (clip_ids, view_index, latent_frame_index))
    if clips.ndim != 1 or clips.shape != views.shape or clips.shape != times.shape:
        raise ValueError("Temporal row metadata does not align")
    if views.dtype.kind not in "iu" or times.dtype.kind not in "iu" or not 1 <= first < n_latents:
        raise ValueError("Invalid temporal index convention")
    keys = [(str(c), int(v), int(t)) for c, v, t in zip(clips, views, times)]
    lookup = {key: i for i, key in enumerate(keys)}
    expected = {(str(c), v, t) for c in np.unique(clips) for v in range(4) for t in range(n_latents)}
    if len(lookup) != len(keys) or set(lookup) != expected:
        raise ValueError("Each clip needs every unique view/time row exactly once")
    current = np.flatnonzero(times >= first)
    previous = np.array([lookup[(str(clips[i]), int(views[i]), int(times[i]) - 1)] for i in current])
    return current, previous


def temporal_features(features, current, previous):
    """Concatenate h_t and h_t-h_(t-1); no future row enters the descriptor."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Temporal features must be a finite matrix")
    return np.concatenate((features[current], features[current] - features[previous]), axis=1)


def target_definitions(absolute_names):
    if len(absolute_names) != 30:
        raise ValueError("Thirty canonical absolute targets required")
    return {"absolute30": {"indices": np.arange(30), "names": list(absolute_names)},
            "role12": {"indices": np.arange(30, 42), "names": list(ROLE_TARGET_NAMES)}}


def group_indices(n_targets):
    if n_targets % 6:
        raise ValueError("Each entity needs XYZ location then XYZ velocity")
    indices = np.arange(n_targets)
    return {"all": indices, "position": indices[indices % 6 < 3], "velocity": indices[indices % 6 >= 3]}


def metrics(y, prediction, ids, scale):
    """Equal-match raw-unit errors and per-target normalized errors, no inferential CI."""
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    weights = match_weights(ids)
    error = prediction - y
    mse = weights @ (error * error)
    mean = weights @ y
    variance = weights @ ((y - mean) ** 2)
    r2 = np.full(y.shape[1], np.nan)
    valid = variance > np.finfo(float).eps * np.maximum(1, weights @ (y * y)) * 16
    np.divide(mse, variance, out=r2, where=valid)
    r2 = 1 - r2
    normalized = mse / np.asarray(scale) ** 2
    return {"normalized_mse": float(normalized.mean()), "normalized_mse_by_target": normalized.tolist(),
            "mae": (weights @ np.abs(error)).tolist(), "rmse": np.sqrt(mse).tolist(),
            "r2": [float(v) if np.isfinite(v) else None for v in r2],
            "n_matches": len(np.unique(ids)), "n_rows": len(y)}


@dataclass
class DevelopmentFit:
    report: dict
    models: list[SelectedSite]
    model_records: list[dict]


def fit_readout(features, targets, match_ids, split, definitions, *, name, kind, feature_spec,
                alphas=ALPHAS, seed=SEED):
    """One decomposition for both target definitions and the matched shuffle."""
    ids, roles = check_development_roles(match_ids, split)
    x, y = np.asarray(features), np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(ids), 42) or len(x) != len(ids):
        raise ValueError("Expected feature rows and absolute30+role12 targets")
    train, selection = roles == "discovery", roles == "selection"
    shuffled, mapping = shuffle_match_labels(y[train], ids[train], seed=seed)
    path = ridge_path(x[train], np.concatenate((y[train], shuffled), axis=1), match_weights(ids[train]), alphas=alphas)
    predictions = [model.predict(x[selection]) for model in path]
    report = {"name": name, "kind": kind, "feature_spec": feature_spec, "feature_width": x.shape[1],
              "label_shuffle_donor_by_recipient_match": mapping, "targets": {}}
    selected, records = [], []
    train_predictions = {}
    for definition, details in definitions.items():
        indices, names = np.asarray(details["indices"]), list(details["names"])
        curve = []
        for model, prediction in zip(path, predictions):
            curve.append({"alpha": model.alpha,
                          "main": metrics(y[selection][:, indices], prediction[:, indices], ids[selection], model.y_scale[indices]),
                          "shuffled": metrics(y[selection][:, indices], prediction[:, indices + 42], ids[selection], model.y_scale[indices + 42])})
        target_report = {"target_names": names, "selection_curve": curve, "groups": {}}
        for group, within in group_indices(len(indices)).items():
            scores = {variant: [float(np.asarray(row[variant]["normalized_mse_by_target"])[within].mean()) for row in curve]
                      for variant in ("main", "shuffled")}
            choices = {variant: int(np.argmin(values)) for variant, values in scores.items()}
            group_models, group_metrics = {}, {}
            for variant, offset in (("main", 0), ("shuffled", 42)):
                chosen = choices[variant]
                model = path[chosen].outputs(indices[within] + offset)
                group_models[variant] = model
                if chosen not in train_predictions:
                    train_predictions[chosen] = path[chosen].predict(x[train])
                group_metrics[variant] = {
                    "alpha": model.alpha,
                    "selection": metrics(y[selection][:, indices[within]], predictions[chosen][:, indices[within] + offset], ids[selection], model.y_scale),
                    "discovery_true_labels": metrics(y[train][:, indices[within]], train_predictions[chosen][:, indices[within] + offset], ids[train], model.y_scale),
                    "coefficient_norm": float(np.linalg.norm(model.coefficient)),
                    "alpha_at_boundary": chosen in (0, len(path) - 1),
                }
            group_names = [names[i] for i in within]
            model_name = f"{name}/{definition}/{group}"
            model_curve = [{"alpha": alpha, "main_normalized_mse": scores["main"][i], "shuffle_normalized_mse": scores["shuffled"][i]}
                           for i, alpha in enumerate(alphas)]
            item = SelectedSite(model_name, kind, group_models["main"], group_models["shuffled"], model_curve)
            selected.append(item)
            record = {"name": model_name, "readout": name, "kind": kind, "feature_spec": feature_spec,
                      "target_definition": definition, "target_group": group, "target_names": group_names,
                      "joined_target_indices": indices[within].tolist(),
                      "selected_alpha": item.main.alpha, "shuffle_alpha": item.shuffled.alpha,
                      "model_sha256": item.main.fingerprint(), "shuffle_model_sha256": item.shuffled.fingerprint(),
                      "selection_curve": model_curve}
            records.append(record)
            mean_prediction = np.broadcast_to(item.main.y_mean, (int(selection.sum()), len(within)))
            target_report["groups"][group] = {"target_names": group_names, **group_metrics,
                "discovery_mean_baseline": metrics(y[selection][:, indices[within]], mean_prediction, ids[selection], item.main.y_scale)}
        report["targets"][definition] = target_report
    return DevelopmentFit(report, selected, records)


def residual_winners(reports):
    """Select only among residual readouts, preserving representation/site order on ties."""
    residual = [row for row in reports if row["kind"] == "residual"]
    if not residual:
        raise ValueError("No residual readouts to compare")
    return {definition: {group: min(residual, key=lambda row: row["targets"][definition]["groups"][group]["main"]["selection"]["normalized_mse"])["name"]
                         for group in ("all", "position", "velocity")}
            for definition in residual[0]["targets"]}
