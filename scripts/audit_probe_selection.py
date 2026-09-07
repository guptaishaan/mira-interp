#!/usr/bin/env python3
"""Independently verify frozen discovery fits and selection choices before confirmation.

This audit never decodes confirmation target rows. It checks saved ridge solutions
through their normal equations, without fitting another model or choosing again.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "8"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APPROVED = {
    "2437f915ceb14a10c33a6b003b7273ee6ba135e8495d4b15b03ff72f0edbd6a3": 1,
    "e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2": 2,
}
FIELDS = ("x_mean", "x_scale", "y_mean", "y_scale", "coefficient")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_target_rows(path, keep):
    """Skip excluded .npy row bytes; never create arrays for excluded labels.

    ZIP may decompress skipped bytes internally. File hashing also reads bytes;
    neither operation interprets them as target values or uses them in statistics.
    """
    with zipfile.ZipFile(path) as archive, archive.open("y.npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "Unsupported target NPY header")
        header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = header(stream)
        require(shape == (len(keep), 30) and not fortran and dtype.kind == "f",
                "Target array must be row-major floating [N,30]")
        result = np.empty((int(keep.sum()), 30), dtype=dtype)
        row_bytes, output_row = dtype.itemsize * 30, 0
        starts = np.r_[0, np.flatnonzero(keep[1:] != keep[:-1]) + 1, len(keep)]
        for start, stop in zip(starts[:-1], starts[1:]):
            count = int(stop - start)
            if keep[start]:
                raw = stream.read(count * row_bytes)
                require(len(raw) == count * row_bytes, "Truncated target array")
                result[output_row:output_row + count] = np.frombuffer(raw, dtype=dtype).reshape(count, 30)
                output_row += count
            else:
                stream.seek(count * row_bytes, 1)
        require(np.isfinite(result).all(), "Nonfinite discovery/selection labels")
        return result


def statistics(values, weights):
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
    tolerance = np.finfo(float).eps * 16 * np.maximum(1, np.abs(values).max(0))
    return mean, np.where(scale > tolerance, scale, 1.0)


def fingerprint(model):
    result = hashlib.sha256(str(float(model["alpha"])).encode())
    for field in FIELDS:
        array = np.ascontiguousarray(model[field], dtype="<f8")
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def check_model(model, features, targets, selection_features, selection_targets,
                train_weights, selection_ids):
    """Check the unique ridge optimum and selected loss using independent equations."""
    x_mean, x_scale = statistics(features, train_weights)
    y_mean, y_scale = statistics(targets, train_weights)
    for name, expected in (("x_mean", x_mean), ("x_scale", x_scale),
                           ("y_mean", y_mean), ("y_scale", y_scale)):
        require(model[name].shape == expected.shape and np.isfinite(model[name]).all(), f"Invalid {name}")
        require(np.allclose(model[name], expected, rtol=1e-10, atol=1e-10), f"Not discovery-only {name}")
    coefficient = model["coefficient"]
    require(coefficient.shape == (features.shape[1], 30) and np.isfinite(coefficient).all(), "Invalid coefficient")
    z = (features - model["x_mean"]) / model["x_scale"]
    residual = z @ coefficient - (targets - model["y_mean"]) / model["y_scale"]
    gradient = z.T @ (train_weights[:, None] * residual) + float(model["alpha"]) * coefficient
    gradient_error = float(np.max(np.abs(gradient)))
    require(gradient_error < 1e-7, f"Stored coefficients do not solve discovery ridge: {gradient_error}")
    prediction = ((selection_features - model["x_mean"]) / model["x_scale"]) @ coefficient
    error = (prediction - (selection_targets - model["y_mean"]) / model["y_scale"]) ** 2
    loss = float(np.mean([error[selection_ids == match].mean() for match in np.unique(selection_ids)]))
    return {"normal_equation_max_abs_residual": gradient_error, "selection_normalized_mse": loss,
            "discovery_normalization_checked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--registration", type=Path, default=ROOT / "configs/observational_probe_v2.json")
    args = parser.parse_args()
    lock_path = args.selection_dir / "frozen_selection.json"
    models_path = args.selection_dir / "frozen_models.npz"
    output_path = args.selection_dir / "selection_audit.json"
    require(not output_path.exists(), "Selection audit exists; it must not be overwritten")
    require(not (args.selection_dir / "confirmation.json").exists(), "Selection audit must precede confirmation")
    frozen = json.loads(lock_path.read_text())
    frozen_hash = digest(lock_path)
    protocol = json.loads(args.registration.read_text())
    audit = json.loads(args.audit.read_text())
    hashes = {"input_npz_sha256": digest(args.data), "registration_sha256": digest(args.registration),
              "capture_audit_sha256": digest(args.audit), "model_archive_sha256": digest(models_path)}
    require(hashes["registration_sha256"] in APPROVED, "Unapproved registration")
    require(all(frozen.get(key) == value for key, value in hashes.items()), "Selection provenance mismatch")
    code_hashes = {name: digest(ROOT / name) for name in ("scripts/analyze_probes.py", "src/mira_interp/probes.py")}
    require(frozen.get("analysis_code_sha256") == code_hashes, "Analysis code changed after selection")
    require(audit.get("status") == "passed" and audit.get("analysis_npz_sha256") == hashes["input_npz_sha256"]
            and audit.get("registration_sha256") == hashes["registration_sha256"], "Capture gate mismatch")
    for name in ("real_data", "video_alignment_checked", "physics_alignment_checked", "all_layers_complete",
                 "match_splits_disjoint", "checkpoint_integrity_checked"):
        require(audit.get(name) is True, f"Missing passed capture prerequisite: {name}")
    require(frozen.get("confirmation_used_for_selection") is False and frozen.get("refit_on_selection") is False,
            "Selection lock claims forbidden data use")
    with np.load(args.data, allow_pickle=False) as archive:
        ids, roles = archive["match_ids"].astype(str), archive["split"].astype(str)
        sites, targets = archive["sites"].tolist(), archive["target_names"].tolist()
        require(ids.shape == roles.shape and ids.ndim == 1 and len(ids) > 0, "Row metadata mismatch")
        require(set(roles) == {"discovery", "selection", "confirmation"}, "Unexpected split labels")
        require(all(len(set(roles[ids == match])) == 1 and np.sum(ids == match) == 256 for match in np.unique(ids)), "Match split/count mismatch")
        counts = {role: len(np.unique(ids[roles == role])) for role in np.unique(roles)}
        require(counts == frozen["match_counts"] == {role: protocol["cohort"][role] for role in counts}, "Registered match counts differ")
        require(sites == protocol["capture"]["sites"] and targets == protocol["targets"] == frozen["target_names"], "Column order differs")
        keep = roles != "confirmation"
        x = archive["X"][keep]
        baselines = {name: archive[name][keep] for name in ("codec_X", "RGB_X") if name in archive.files}
    y = selected_target_rows(args.data, keep).astype(np.float64)
    ids, roles = ids[keep], roles[keep]
    train, selection = roles == "discovery", roles == "selection"
    train_ids, select_ids = ids[train], ids[selection]
    unique = np.unique(train_ids)
    weights = np.array([1.0 / (len(unique) * np.sum(train_ids == match)) for match in train_ids])
    donors = frozen["label_shuffle_donor_by_recipient_match"]
    require(set(donors) == set(unique) == set(donors.values()) and all(k != v for k, v in donors.items()), "Invalid whole-match derangement")
    order = np.random.default_rng(20260906).permutation(len(unique))
    expected_donors = {str(unique[a]): str(unique[b]) for a, b in zip(order, np.roll(order, -1))}
    require(frozen["label_shuffle_seed"] == 20260906 and donors == expected_donors, "Registered shuffle seed/mapping differs")
    control = np.empty_like(y[train])
    for recipient, donor in donors.items():
        control[train_ids == recipient] = y[train][train_ids == donor]
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    require(frozen["alpha_grid"] == protocol["probes"]["alphas"] == alphas, "Alpha grid differs")
    expected_models = [(site, "residual") for site in sites] + [(name, "baseline") for name in baselines]
    require([(r["name"], r["kind"]) for r in frozen["models"]] == expected_models, "Model/site ordering differs")
    checks = []
    with np.load(models_path, allow_pickle=False) as models:
        for index, row in enumerate(frozen["models"]):
            print(f"Independent selection audit: {row['name']}", flush=True)
            features = np.asarray(x[:, index, :] if row["kind"] == "residual" else baselines[row["name"]], dtype=np.float64)
            require(np.isfinite(features).all(), "Nonfinite discovery/selection features")
            curve = row["selection_curve"]
            require([point["alpha"] for point in curve] == alphas, "Incomplete selection curve")
            checked = {"name": row["name"], "kind": row["kind"]}
            for kind, label, alpha_key, hash_key, loss_key in (
                ("main", y[train], "selected_alpha", "model_sha256", "main_normalized_mse"),
                ("shuffled", control, "shuffle_alpha", "shuffle_model_sha256", "shuffle_normalized_mse")):
                model = {field: models[f"site{index}_{kind}_{field}"] for field in ("alpha", *FIELDS)}
                losses = [point[loss_key] for point in curve]
                require(np.isfinite(losses).all() and min(losses) >= 0, "Invalid recorded selection errors")
                chosen = int(np.argmin(losses))
                require(float(model["alpha"]) == row[alpha_key] == alphas[chosen], "Alpha differs from recorded selection minimum")
                require(fingerprint(model) == row[hash_key], "Coefficient/normalization fingerprint mismatch")
                checked[kind] = check_model(model, features[train], label, features[selection], y[selection], weights, select_ids)
                require(np.isclose(checked[kind]["selection_normalized_mse"], losses[chosen], rtol=1e-10, atol=1e-10), "Recomputed selected loss differs")
            checks.append(checked)
    winner = min((row for row in checks if row["kind"] == "residual"), key=lambda row: row["main"]["selection_normalized_mse"])["name"]
    require(winner == frozen["winner"], "Frozen global site is not the selection minimum")
    require(digest(lock_path) == frozen_hash and digest(models_path) == hashes["model_archive_sha256"], "Frozen artifacts changed during audit")
    require({name: digest(ROOT / name) for name in code_hashes} == code_hashes, "Analysis code changed during audit")
    report = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), **hashes,
              "frozen_selection_sha256": frozen_hash, "analysis_code_sha256": code_hashes,
              "audit_script_sha256": digest(Path(__file__)), "match_counts": counts, "frozen_winner": winner,
              "confirmation_target_rows_materialized": 0, "confirmation_target_values_used": False,
              "models_refitted": False, "ridge_normal_equations_checked": True,
              "unselected_alpha_models_independently_refitted": False,
              "limits": "Chosen models and losses independently recomputed; unselected-alpha curves checked for completeness and choice consistency, not refitted.",
              "models": checks}
    with output_path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Passed selection audit: {output_path}", flush=True)


if __name__ == "__main__":
    main()
