#!/usr/bin/env python3
"""Independently verify frozen fresh-match predictions and metrics without fitting."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCK = "2750c3a9df2478898aaef6f5bf470d5241eb4315257a2e75f59daddd6a7476b9"
SEED, REPLICATES = 20260907, 500
FEATURES = ["mean_X", "spatial_X", "codec_spatial_X", "codec_mean_X", "RGB_X"]
SHAPES = [(32, 17, 2048), (32, 17, 1536), (32, 4608), (32, 32), (32, 1536)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_numbers(actual, expected, name):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True), f"Metric differs: {name}")


def model_fingerprint(model):
    h = hashlib.sha256(str(float(model["alpha"])).encode())
    for key in ["x_mean", "x_scale", "y_mean", "y_scale", "coefficient"]:
        a = np.ascontiguousarray(model[key], dtype="<f8")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def ci(values):
    a = np.asarray(values)
    if a.ndim == 1:
        valid = a[np.isfinite(a)]
        return np.quantile(valid, [.025, .975]).tolist() if len(valid) else [None, None]
    return [ci(a[:, index]) for index in range(a.shape[1])]


def independent_metrics(actual, prediction, scale, match_rows, draws):
    # Separate per-match moments, then combine equal-weight matches. No row is
    # treated as an independent bootstrap sample.
    per_mse = np.stack([np.square(actual[rows] - prediction[rows]).mean(0) for rows in match_rows])
    per_mae = np.stack([np.abs(actual[rows] - prediction[rows]).mean(0) for rows in match_rows])
    per_mean = np.stack([actual[rows].mean(0) for rows in match_rows])
    per_second = np.stack([np.square(actual[rows]).mean(0) for rows in match_rows])
    mse, mean, second = per_mse.mean(0), per_mean.mean(0), per_second.mean(0)
    # Point R2 uses explicitly centered labels; bootstrap R2 uses weighted moments.
    variance = np.stack([np.square(actual[rows] - mean).mean(0) for rows in match_rows]).mean(0)
    valid = variance > np.finfo(float).eps * np.maximum(1., second) * 16
    r2 = np.full(len(scale), np.nan)
    r2[valid] = 1 - mse[valid] / variance[valid]
    bmse = per_mse[draws].mean(1)
    bmean, bsecond = per_mean[draws].mean(1), per_second[draws].mean(1)
    bvariance = np.maximum(0, bsecond - bmean ** 2)
    bvalid = bvariance > np.finfo(float).eps * np.maximum(1., bsecond) * 16
    br2 = np.full_like(bmse, np.nan)
    br2[bvalid] = 1 - bmse[bvalid] / bvariance[bvalid]
    return {"normalized_mse": float((mse / scale ** 2).mean()),
            "normalized_mse_ci95": ci((bmse / scale ** 2).mean(1)),
            "mae": per_mae.mean(0).tolist(), "mae_ci95": ci(per_mae[draws].mean(1)),
            "rmse": np.sqrt(mse).tolist(), "rmse_ci95": ci(np.sqrt(bmse)),
            "r2": [float(x) if np.isfinite(x) else None for x in r2], "r2_ci95": ci(br2),
            "r2_bootstrap_valid_replicates": np.isfinite(br2).sum(0).tolist()}


def independent_gain(actual, prediction, baseline, scale, match_rows, draws):
    differences = (np.square((baseline - actual) / scale) - np.square((prediction - actual) / scale)).mean(1)
    per_match = np.asarray([differences[rows].mean() for rows in match_rows])
    return {"normalized_mse_gain": float(per_match.mean()), "ci95": ci(per_match[draws].mean(1))}


def check_metric(observed, expected, name):
    require(expected["n_matches"] == 23 and expected["n_examples"] == 4416
            and expected["bootstrap_replicates"] == REPLICATES and expected["bootstrap_seed"] == SEED
            and expected["bootstrap_unit"] == "whole_match", "Metric resampling convention changed")
    for key, value in observed.items():
        equal_numbers(value, expected[key], f"{name}/{key}")


def check_gain(observed, expected, name):
    require(expected["n_matches"] == 23 and expected["bootstrap_replicates"] == REPLICATES
            and expected["bootstrap_seed"] == SEED and expected["bootstrap_unit"] == "paired_whole_match",
            "Gain resampling convention changed")
    for key, value in observed.items():
        equal_numbers(value, expected[key], f"{name}/{key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results/fresh_confirmation_v3/confirmation_audit.json")
    args = parser.parse_args()
    require(not args.report.exists(), "Prior independent audit exists; do not overwrite")
    result_path = ROOT / "results/fresh_confirmation_v3/confirmation.json"
    require(result_path.exists() and read(result_path)["status"] == "passed_fresh_observational_confirmation", "Fresh evaluation must finish first")
    started = time.monotonic()
    audit = {"status": "running", "scope": "independent frozen fresh-match metrics audit",
             "models_refitted": False, "new_hypotheses_or_choices": False, "clips_checked": 0, "models_checked": 0}
    def progress(stage):
        audit.update(stage=stage, elapsed_seconds=round(time.monotonic() - started, 3))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps({k: audit[k] for k in ["stage", "elapsed_seconds", "clips_checked", "models_checked"]}), flush=True)
    progress("prerequisites")
    try:
        lock_path = ROOT / "configs/fresh_evaluation_v3.json"
        qpath = ROOT / "data/qualified_fresh_confirmation_clip_manifest.json"
        prep_path = ROOT / "results/fresh_confirmation_float.json"
        candidate_path = ROOT / "data/fresh_confirmation_split_manifest.json"
        cap_audit_path = ROOT / "results/fresh_confirmation_v3/capture_audit.json"
        old_path = ROOT / "data/split_manifest.json"
        directory = ROOT / "results/development_probes_v3"
        selection_path, models_path = directory / "development_selection.json", directory / "development_models.npz"
        report_path, prior_audit_path = directory / "development_report.json", directory / "probe_audit.json"
        lock, result, qualified, prep, previous = (read(p) for p in [lock_path, result_path, qpath, prep_path, prior_audit_path])
        selection, development, capture_audit = read(selection_path), read(report_path), read(cap_audit_path)
        require(sha(lock_path) == LOCK == result["registered_choices_sha256"] == capture_audit["registered_choice_sha256"], "Frozen choice registration changed")
        require(lock["status"] == "frozen_before_fresh_capture" and lock["model_refit"] is False
                and lock["no_fresh_activation_or_prediction_inspected"] is True, "Invalid fresh choice scope")
        require(previous["status"] == capture_audit["status"] == qualified["status"] == prep["status"] == "passed", "Prior audits incomplete")
        bindings = {selection_path: lock["development_selection_sha256"], models_path: lock["model_archive_sha256"],
                    report_path: lock["development_report_sha256"], prior_audit_path: lock["probe_audit_sha256"],
                    candidate_path: lock["candidate_manifest_sha256"], cap_audit_path: result["capture_audit_sha256"]}
        require(previous["development_selection_sha256"] == result["development_selection_sha256"] == lock["development_selection_sha256"]
                and previous["model_archive_sha256"] == result["model_archive_sha256"] == lock["model_archive_sha256"]
                and previous["development_report_sha256"] == lock["development_report_sha256"], "Audited development inputs changed")
        for group in ["capture_code_sha256", "evaluation_code_sha256"]:
            bindings.update({ROOT / p: expected for p, expected in lock[group].items()})
        require(all(sha(p) == h for p, h in bindings.items()), "Frozen input/code bytes changed")
        bindings.update({p: sha(p) for p in [Path(__file__), lock_path, result_path, qpath, prep_path, old_path]})
        require(result["models_refitted"] is result["new_choices_from_confirmation"] is result["physical_control_established"] is False,
                "Unexpected fresh-result claims")
        require(result["n_matches"] == qualified["n_matches"] == 23 and result["n_rows"] == 4416
                and capture_audit["n_clips"] == 184 and capture_audit["rows"] == 4416, "Fresh coverage differs")
        require(qualified["registered_manifest_sha256"] == lock["candidate_manifest_sha256"]
                and prep["qualified_manifest_sha256"] == sha(qpath) == capture_audit["qualified_manifest_sha256"], "Fresh source lineage changed")
        # Reconstruct choices from the prior development report only, before any
        # fresh arrays are read. No confirmation score enters this step.
        expected_choices = {}
        for definition in ["absolute30", "role12"]:
            expected_choices[definition] = {}
            for group in ["all", "position", "velocity"]:
                def prior_winner(rows):
                    best = min(rows, key=lambda r: r["targets"][definition]["groups"][group]["main"]["selection"]["normalized_mse"])
                    return f"{best['name']}/{definition}/{group}"
                expected_choices[definition][group] = {
                    "primary": prior_winner([r for r in development["stage1"] + development["stage2"] if r["kind"] == "residual"]),
                    "mean_current": prior_winner([r for r in development["stage1"] if r["name"].startswith("mean/")]),
                    "spatial_current": prior_winner([r for r in development["stage1"] if r["name"].startswith("spatial/")]),
                    **{label: f"{array}/{definition}/{group}" for label, array in [("codec_spatial", "codec_spatial_X"), ("codec_mean", "codec_mean_X"), ("RGB", "RGB_X")]}}
        require(lock["choices"] == expected_choices, "Choices differ from frozen development-only rule")
        expected_matches = set(qualified["qualified_matches"])
        require(len(expected_matches) == 23 and not expected_matches & {r["match_id"] for r in read(old_path)["matches"]}, "Fresh matches overlap original cohort")
        original = {r["clip_id"]: r for r in qualified["records"]}
        prepared = {r["clip_id"]: r for r in prep["records"]}
        require(len(original) == len(qualified["records"]) == len(prepared) == len(prep["records"]) == 184
                and set(original) == set(prepared) and set(Counter(r["match_id"] for r in original.values()).values()) == {8}
                and {r["match_id"] for r in original.values()} == expected_matches, "Incomplete qualified/prepared cohort")
        workers = []
        for path in sorted((ROOT / "results").glob("fresh_capture_worker*.json")):
            worker = read(path)
            require(worker["status"] == "passed" and not worker["errors"] and worker["registration_sha256"] == LOCK
                    and worker["preparation_sha256"] == sha(prep_path) and worker["code_sha256"] == lock["capture_code_sha256"], "Worker incomplete or changed")
            require(datetime.fromisoformat(worker["started_utc"]) > datetime.fromisoformat(lock["registered_utc"]), "Capture began before choice freeze")
            workers.extend(worker["records"])
            bindings[path] = sha(path)
        require(len(workers) == 184 and {r["clip_id"] for r in workers} == set(original), "Worker record coverage differs")
        lookup = {r["clip_id"]: r for r in workers}
        pieces = defaultdict(list)
        source_hashes, clip_checks = {}, []
        for index, cid in enumerate(sorted(original, key=lambda c: (original[c]["match_id"], c))):
            raw_record, prepared_record, side = original[cid], prepared[cid], lookup[cid]
            require(raw_record["role"] == prepared_record["split"] == side["split"] == "confirmation"
                    and raw_record["match_id"] == prepared_record["match_id"] == side["match_id"], "Role or match mismatch before array access")
            raw_path, input_path, captured_path = Path(raw_record["artifact_path"]), Path(prepared_record["artifact_path"]), Path(side["path"])
            for path, expected in [(raw_path, raw_record["sha256"]), (input_path, prepared_record["sha256"]), (captured_path, side["sha256"])]:
                require(sha(path) == expected, f"Source/capture hash changed: {cid}")
                source_hashes[path] = expected
            require(prepared_record["parent_sha256"] == raw_record["sha256"] and side["input_sha256"] == prepared_record["sha256"], "Input parent mismatch")
            side_path = captured_path.with_suffix(".json")
            require(read(side_path) == side, "Worker/sidecar record differs")
            bindings[side_path] = sha(side_path)
            with np.load(raw_path, allow_pickle=False) as raw, np.load(input_path, allow_pickle=False) as resized, np.load(captured_path, allow_pickle=False) as saved:
                for key in ["targets", "source_frame_indices", "timestamps", "player_ids", "target_names", "actions"]:
                    require(np.array_equal(raw[key], resized[key]), "Prepared source metadata/targets differ")
                exact = {"y": raw["targets"][:, 1::2].reshape(32, 30), "timestamps": raw["timestamps"][:, 1::2].reshape(32),
                         "source_frame_index": np.tile(raw["source_frame_indices"][1::2], 4),
                         "canonical_player_ids": np.tile(raw["player_ids"], (32, 1)), "target_names": raw["target_names"],
                         "view_index": np.repeat(np.arange(4), 8), "latent_frame_index": np.tile(np.arange(8), 4)}
                require(exact["y"].dtype == np.float64 and np.isfinite(exact["y"]).all()
                        and np.isfinite(exact["timestamps"]).all(), "Invalid source labels/timestamps")
                require(all(np.array_equal(saved[key], value) for key, value in exact.items()), "Original source label/time/identity join differs")
                require(saved["clip_ids"].tolist() == [cid] * 32 and saved["match_ids"].tolist() == [side["match_id"]] * 32
                        and saved["split"].tolist() == ["confirmation"] * 32 and saved["sites"].tolist() == ["block_0_input"] + [f"block_{i}_output" for i in range(16)], "Row/site identity differs")
                for key, shape in zip(FEATURES, SHAPES):
                    value = saved[key]
                    require(value.shape == shape and value.dtype == np.float16 and np.isfinite(value).all(), "Invalid feature array")
                    pieces[key].append(value)
                pieces["y"].append(exact["y"])
                pieces["ids"].append(saved["match_ids"])
            clip_checks.append({"clip_id": cid, "sha256": side["sha256"], "source_sha256": raw_record["sha256"], "label_join_exact": True})
            audit["clips_checked"] = index + 1
            if index % 32 == 0:
                progress("read_source_labels_and_features")
        data = {key: np.concatenate(value) for key, value in pieces.items()}
        current = np.asarray([base + view * 8 + t for base in range(0, 184 * 32, 32) for view in range(4) for t in range(2, 8)])
        previous_rows = current - 1
        # Explicit per-view canonical player slice; no role-target helper reused.
        views = np.tile(np.repeat(np.arange(4), 8), 184)
        ego = np.stack([data["y"][row, (view + 1) * 6:(view + 2) * 6] for row, view in enumerate(views)])
        joined = np.column_stack([data["y"], ego, data["y"][:, :6] - ego])[current]
        ids = data["ids"][current]
        require(len(ids) == 4416 and Counter(ids).keys() == expected_matches and set(Counter(ids).values()) == {192}, "Scored row coverage differs")
        match_rows = [np.flatnonzero(ids == match) for match in sorted(expected_matches)]
        draws = np.random.default_rng(SEED).integers(23, size=(REPLICATES, 23))
        indexed = {entry["name"]: (index, entry) for index, entry in enumerate(selection["models"])}
        chosen = {name for groups in lock["choices"].values() for choices in groups.values() for name in choices.values()}
        require(set(result["metrics"]) == chosen, "Reported models differ from frozen choices")
        predictions, fitted, metrics = {}, {}, {}
        with np.load(models_path, allow_pickle=False) as archive:
            for name in sorted(chosen):
                index, entry = indexed[name]
                spec = entry["feature_spec"]
                features = data[spec["array"]]
                if spec["site_index"] is not None:
                    features = features[:, spec["site_index"]]
                features = features.astype(np.float64)
                x = features[current]
                if spec["temporal"] == "current_and_delta":
                    x = np.column_stack([x, x - features[previous_rows]])
                else:
                    require(spec["temporal"] == "current", "Unknown temporal feature")
                actual = joined[:, entry["joined_target_indices"]]
                values = {}
                for variant, expected in [("main", entry["model_sha256"]), ("shuffled", entry["shuffle_model_sha256"])]:
                    model = {key: archive[f"site{index}_{variant}_{key}"] for key in ["alpha", "x_mean", "x_scale", "y_mean", "y_scale", "coefficient"]}
                    require(model_fingerprint(model) == expected, "Saved model coefficients changed")
                    values[variant] = (((x - model["x_mean"]) / model["x_scale"]) @ model["coefficient"]) * model["y_scale"] + model["y_mean"]
                    if variant == "main":
                        fitted[name] = model
                main = fitted[name]
                values["mean_baseline"] = np.broadcast_to(main["y_mean"], actual.shape)
                metrics[name] = {variant: independent_metrics(actual, prediction, main["y_scale"], match_rows, draws) for variant, prediction in values.items()}
                for variant in values:
                    check_metric(metrics[name][variant], result["metrics"][name][variant], f"{name}/{variant}")
                require(result["metrics"][name]["feature_spec"] == spec and result["metrics"][name]["target_names"] == entry["target_names"], "Metric column names/spec changed")
                check_gain(independent_gain(actual, values["main"], values["shuffled"], main["y_scale"], match_rows, draws), result["metrics"][name]["gain_vs_shuffled"], name + "/shuffle_gain")
                predictions[name] = values["main"]
                audit["models_checked"] += 1
        primary = {}
        for definition, groups in lock["choices"].items():
            primary[definition] = {}
            for group, choices in groups.items():
                name = choices["primary"]
                actual = joined[:, indexed[name][1]["joined_target_indices"]]
                model = fitted[name]
                reported = result["comparisons"][definition][group]
                require(reported["choices"] == choices, "Published comparisons differ from frozen choices")
                comparisons = {label: (predictions[name], predictions[other]) for label, other in choices.items() if label != "primary"}
                comparisons["mean_baseline"] = (predictions[name], np.broadcast_to(model["y_mean"], actual.shape))
                comparisons["spatial_current_vs_mean_current"] = (predictions[choices["spatial_current"]], predictions[choices["mean_current"]])
                require(set(reported["paired_gains"]) == set(comparisons), "Missing or extra fixed comparison")
                gains = {}
                for label, (prediction, baseline) in comparisons.items():
                    gains[label] = independent_gain(actual, prediction, baseline, model["y_scale"], match_rows, draws)
                    check_gain(gains[label], reported["paired_gains"][label], f"{definition}/{group}/{label}")
                primary[definition][group] = {"name": name, "target_names": indexed[name][1]["target_names"],
                                             "main": metrics[name]["main"], "mean_baseline": metrics[name]["mean_baseline"], "paired_gains": gains}
        require(all(sha(path) == h for path, h in {**bindings, **source_hashes}.items()), "Inputs or code changed during audit")
        audit.update(status="passed", confirmation_sha256=sha(result_path), registered_choices_sha256=LOCK,
                     model_archive_sha256=sha(models_path), development_selection_sha256=sha(selection_path),
                     prior_probe_audit_sha256=sha(prior_audit_path), source_manifest_sha256=sha(qpath),
                     capture_audit_sha256=sha(cap_audit_path), n_matches=23, n_clips=184, scored_rows=4416,
                     all_source_label_joins_exact=True, frozen_choices_recomputed_from_development=True,
                     all_published_metrics_and_intervals_recomputed=True, paired_whole_match_bootstrap_checked=True,
                     bootstrap_replicates=REPLICATES, bootstrap_seed=SEED, primary=primary, clips=clip_checks,
                     audit_script_sha256=sha(Path(__file__)), physical_control_established=False)
        progress("complete")
    except Exception as error:
        audit.update(status="failed", error_type=type(error).__name__, error=str(error))
        progress("failed")
        raise


if __name__ == "__main__":
    main()
