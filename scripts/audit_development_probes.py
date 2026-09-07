#!/usr/bin/env python3
"""Audit saved development ridge models without refitting or reading confirmation.

Independently reconstruct discovery normalization, target definitions, shuffled
match blocks, temporal features, selection errors and ridge normal equations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from aggregate_captures import digest, read_json, require, write_json


def weights(ids):
    counts = Counter(ids.tolist())
    return np.asarray([1 / (len(counts) * counts[match]) for match in ids], dtype=np.float64)


def standardization(values, weight):
    mean = np.sum(weight[:, None] * values, axis=0)
    scale = np.sqrt(np.sum(weight[:, None] * (values - mean) ** 2, axis=0))
    floor = np.finfo(np.float64).eps * 16 * np.maximum(1, np.max(np.abs(values), axis=0))
    scale = np.where(scale > floor, scale, 1.)
    return mean, scale


def fingerprint(model):
    h = hashlib.sha256(str(float(model["alpha"])).encode())
    for key in ["x_mean", "x_scale", "y_mean", "y_scale", "coefficient"]:
        array = np.ascontiguousarray(model[key], dtype="<f8")
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/development_probes_v3")
    parser.add_argument("--capture-manifest", type=Path, default=ROOT / "results/development_capture_manifest.json")
    parser.add_argument("--capture-audit", type=Path, default=ROOT / "results/development_capture_audit.json")
    args = parser.parse_args()
    audit_path = args.output_dir / "probe_audit.json"
    started = time.monotonic()
    audit = {"status": "running", "scope": "development saved-model audit; no new confirmation", "models_refitted": False, "confirmation_arrays_read": False, "readouts": []}
    def stage(name):
        audit.update(stage=name, elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(audit_path, audit)
        print(f"[{audit['elapsed_seconds']:.1f}s] {name}: {len(audit['readouts'])} readouts", flush=True)
    stage("prerequisites")
    try:
        selection_path, models_path = args.output_dir / "development_selection.json", args.output_dir / "development_models.npz"
        report_path = args.output_dir / "development_report.json"
        selection, result = read_json(selection_path), read_json(report_path)
        require(selection["status"] == "frozen_development_choices" and result["status"] == "passed_development_analysis", "Development fitting has not completed")
        require(selection["confirmation_used"] is result["confirmation_used"] is False and selection["new_confirmation_required"] is True and result["models_refitted_with_selection"] is False, "Development/confirmation scope differs")
        require(digest(models_path) == selection["model_archive_sha256"] == result["model_archive_sha256"], "Saved model archive changed")
        require(digest(selection_path) == result["development_selection_sha256"], "Frozen selection changed")
        code = selection["analysis_code_sha256"]
        require(code == result["analysis_code_sha256"] and all(digest(ROOT / path) == value for path, value in code.items()), "Analysis code changed")
        manifest, capture_audit = read_json(args.capture_manifest), read_json(args.capture_audit)
        require(manifest["status"] == capture_audit["status"] == "passed" and capture_audit.get("pilot_only") is False, "Independent full capture audit is required")
        require(capture_audit["capture_manifest_sha256"] == digest(args.capture_manifest), "Capture audit manifest binding changed")
        require(capture_audit["all_source_label_joins_exact"] is True and capture_audit["all_layers_complete"] is True, "Capture source/layer audit incomplete")
        provenance = selection["provenance"]
        require(provenance == result["provenance"], "Selection/result provenance differs")
        bindings = {"capture_manifest_sha256": args.capture_manifest, "capture_audit_sha256": args.capture_audit, "registration_sha256": ROOT / "configs/development_v3.json", "split_manifest_sha256": ROOT / "data/qualified_split_manifest.json", "source_manifest_sha256": ROOT / "data/qualified_clip_manifest.json", "source_audit_sha256": ROOT / "results/qualified_data_audit.json", "preparation_manifest_sha256": ROOT / "results/development_prepare.json"}
        # The analyzer may have used the already-audited combined manifest as its
        # gate; the standalone audit remains independently required here.
        if provenance["capture_audit_sha256"] == digest(args.capture_manifest):
            bindings["capture_audit_sha256"] = args.capture_manifest
        for key, path in bindings.items():
            require(provenance[key] == digest(path), f"Analysis input changed: {key}")
        require(provenance["capture_code_sha256"] == manifest["code_sha256"] and all(digest(ROOT / path) == value for path, value in manifest["code_sha256"].items()), "Capture code changed")
        protocol = read_json(bindings["registration_sha256"])
        require(provenance["registration_sha256"] == "607aef226c2cb241e66d00c9a3b7f323802c6bce869f96345c3a0b014a220755", "Unknown development registration")
        alphas = protocol["analysis"]["ridge_alphas"]
        require(result["alpha_grid"] == alphas and protocol["analysis"]["scored_latents"] == list(range(2, 8)), "Analysis alpha/time grid differs")
        split = read_json(bindings["split_manifest_sha256"])
        expected_matches = {row["match_id"]: row["split"] for row in split["matches"] if row["split"] in {"discovery", "selection"}}
        records = sorted(manifest["records"], key=lambda row: (row["match_id"], row["clip_id"]))
        require(len(records) == 336 and len({row["clip_id"] for row in records}) == 336, "Wrong development clip count")
        for row in records:
            require(row["split"] in {"discovery", "selection"} and expected_matches[row["match_id"]] == row["split"], "Forbidden role before NPZ access")
        require(set(expected_matches) == {row["match_id"] for row in records}, "Wrong development match cohort")
        indexed_inputs = {row["clip_id"]: row["sha256"] for row in provenance["clips"]}
        data = defaultdict(list)
        for index, row in enumerate(records):
            path = Path(row["path"])
            require(digest(path) == row["sha256"] == indexed_inputs[row["clip_id"]], "Source feature archive changed")
            with np.load(path, allow_pickle=False) as saved:
                require(saved["split"].tolist() == [row["split"]] * 32 and saved["match_ids"].tolist() == [row["match_id"]] * 32, "Feature archive role mismatch before label access")
                require(saved["view_index"].tolist() == np.repeat(np.arange(4), 8).tolist() and saved["latent_frame_index"].tolist() == np.tile(np.arange(8), 4).tolist(), "Unexpected view/time ordering")
                for key in ["mean_X", "spatial_X", "codec_spatial_X", "codec_mean_X", "RGB_X", "y", "view_index", "match_ids", "split"]:
                    data[key].append(saved[key])
            if index % 75 == 0:
                stage("read_verified_source_features")
        data = {key: np.concatenate(value) for key, value in data.items()}
        current = np.asarray([base + view * 8 + time_index for base in range(0, len(records) * 32, 32) for view in range(4) for time_index in range(2, 8)])
        previous = current - 1
        # Independent explicit view-indexed target construction.
        absolute = data["y"].astype(np.float64)
        ego = np.stack([absolute[row, 6 * (int(view) + 1):6 * (int(view) + 2)] for row, view in enumerate(data["view_index"])])
        y = np.concatenate([absolute, ego, absolute[:, :6] - ego], axis=1)[current]
        ids, roles = data["match_ids"][current], data["split"][current]
        train, valid = roles == "discovery", roles == "selection"
        require(all(n == 192 for n in Counter(ids).values()), "Unequal per-match scored rows")
        require(len(set(ids[train])) == 31 and len(set(ids[valid])) == 11, "Wrong fit/selection match counts")
        wt, wv = weights(ids[train]), weights(ids[valid])
        stages = result["stage1"] + result["stage2"]
        readout_reports = {row["name"]: row for row in stages}
        require(len(readout_reports) == len(stages), "Duplicate readout report names")
        sites = ["block_0_input"] + [f"block_{i}_output" for i in range(16)]
        expected_stage1 = [(f"{summary}/{site}", "residual", {"array": key, "site_index": i, "summary": summary, "temporal": "current"})
                           for summary, key in [("mean", "mean_X"), ("spatial", "spatial_X")] for i, site in enumerate(sites)]
        expected_stage1 += [(key, "baseline", {"array": key, "site_index": None, "summary": key, "temporal": "current"})
                            for key in ["codec_mean_X", "codec_spatial_X", "RGB_X"]]
        require([(row["name"], row["kind"], row["feature_spec"]) for row in result["stage1"]] == expected_stage1, "Stage-one readout grid differs")
        temporal_names = list(dict.fromkeys(result["stage1_residual_winners"][definition]["all"] for definition in ["absolute30", "role12"]))
        expected_stage2 = [(name + "/current_and_delta", "residual", {**readout_reports[name]["feature_spec"], "temporal": "current_and_delta"}) for name in temporal_names]
        require([(row["name"], row["kind"], row["feature_spec"]) for row in result["stage2"]] == expected_stage2, "Temporal readouts differ from the registered selected-site rule")
        groups = defaultdict(list)
        for index, entry in enumerate(selection["models"]):
            groups[entry["readout"]].append((index, entry))
        require(set(groups) == set(readout_reports), "Saved models do not cover the readout reports")
        frozen_paths = [selection_path, models_path, report_path, args.capture_audit, args.output_dir / "stage1_selection.json", Path(__file__), *bindings.values()]
        frozen = {str(path): digest(path) for path in frozen_paths}
        with np.load(models_path, allow_pickle=False) as models:
            for readout, entries in groups.items():
                details = readout_reports[readout]
                spec = details["feature_spec"]
                require(all(entry["feature_spec"] == spec for _, entry in entries), "Feature specifications disagree")
                require({(entry["target_definition"], entry["target_group"]) for _, entry in entries} == {(definition, group) for definition in ["absolute30", "role12"] for group in ["all", "position", "velocity"]}, "Missing target definition/group models")
                features = data[spec["array"]]
                if spec["site_index"] is not None:
                    features = features[:, spec["site_index"]]
                features = features.astype(np.float64)
                x = features[current]
                if spec["temporal"] == "current_and_delta":
                    x = np.concatenate([x, x - features[previous]], axis=1)
                else:
                    require(spec["temporal"] == "current", "Unknown temporal feature definition")
                x_train, x_valid = x[train], x[valid]
                x_mean, x_scale = standardization(x_train, wt)
                z, zv = (x_train - x_mean) / x_scale, (x_valid - x_mean) / x_scale
                mapping = details["label_shuffle_donor_by_recipient_match"]
                require(set(mapping) == set(ids[train]) == set(mapping.values()) and all(k != v for k, v in mapping.items()), "Shuffle is not a complete match derangement")
                unique = sorted(set(ids[train]))
                order = np.random.default_rng(result["seed"]).permutation(len(unique))
                expected_mapping = {unique[int(order[i])]: unique[int(order[(i + 1) % len(order)])] for i in range(len(order))}
                require(mapping == expected_mapping, "Label shuffle differs from the registered seeded derangement")
                y_train = y[train]
                shuffled = np.empty_like(y_train)
                for recipient, donor in mapping.items():
                    shuffled[ids[train] == recipient] = y_train[ids[train] == donor]
                blocks, normalized_targets, valid_truth, ymeans, yscales, penalties, columns = [], [], [], [], [], [], []
                summaries = []
                start_column = 0
                for index, entry in entries:
                    require([row["alpha"] for row in entry["selection_curve"]] == alphas, "Preserved alpha curve changed")
                    indices = entry["joined_target_indices"]
                    definition_indices = np.arange(30) if entry["target_definition"] == "absolute30" else np.arange(30, 42)
                    group = entry["target_group"]
                    expected_indices = definition_indices if group == "all" else definition_indices[definition_indices % 6 < 3] if group == "position" else definition_indices[definition_indices % 6 >= 3]
                    require(indices == expected_indices.tolist(), "Target definition/group indices changed")
                    for variant, fingerprint_key, alpha_key, curve_key in [("main", "model_sha256", "selected_alpha", "main_normalized_mse"), ("shuffled", "shuffle_model_sha256", "shuffle_alpha", "shuffle_normalized_mse")]:
                        model = {key: models[f"site{index}_{variant}_{key}"] for key in ["alpha", "x_mean", "x_scale", "y_mean", "y_scale", "coefficient"]}
                        require(fingerprint(model) == entry[fingerprint_key], "Saved coefficient fingerprint differs")
                        alpha = float(model["alpha"])
                        scores = [row[curve_key] for row in entry["selection_curve"]]
                        require(alpha == entry[alpha_key] == alphas[min(range(len(alphas)), key=lambda i: scores[i])], "Chosen alpha is not the minimum preserved selection loss")
                        fit_y = (y_train if variant == "main" else shuffled)[:, indices]
                        ym, ys = standardization(fit_y, wt)
                        for key, expected in [("x_mean", x_mean), ("x_scale", x_scale), ("y_mean", ym), ("y_scale", ys)]:
                            require(model[key].shape == expected.shape and np.allclose(model[key], expected, rtol=1e-9, atol=1e-10), f"Saved {key} is not discovery-only normalization")
                        b = model["coefficient"]
                        require(b.shape == (x.shape[1], len(indices)) and np.isfinite(b).all(), "Bad saved coefficient matrix")
                        stop_column = start_column + len(indices)
                        columns.append((start_column, stop_column, entry, variant, min(scores)))
                        start_column = stop_column
                        blocks.append(b)
                        normalized_targets.append((fit_y - ym) / ys)
                        valid_truth.append(y[valid][:, indices])
                        ymeans.append(ym)
                        yscales.append(ys)
                        penalties.extend([alpha] * len(indices))
                coefficient = np.concatenate(blocks, axis=1)
                target = np.concatenate(normalized_targets, axis=1)
                prediction_train = z @ coefficient
                rhs = z.T @ (wt[:, None] * target)
                residual = z.T @ (wt[:, None] * (prediction_train - target)) + np.asarray(penalties)[None] * coefficient
                relative = np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-12)
                require(np.max(relative) < 1e-6, "Saved coefficients fail the weighted ridge normal equations")
                ym, ys = np.concatenate(ymeans), np.concatenate(yscales)
                prediction_valid = (zv @ coefficient) * ys + ym
                true_valid = np.concatenate(valid_truth, axis=1)
                target_mse = wv @ (((prediction_valid - true_valid) / ys) ** 2)
                for begin, end, entry, variant, curve_minimum in columns:
                    loss = float(target_mse[begin:end].mean())
                    published = details["targets"][entry["target_definition"]]["groups"][entry["target_group"]][variant]["selection"]["normalized_mse"]
                    require(np.isclose(loss, curve_minimum, rtol=1e-8, atol=1e-10) and np.isclose(loss, published, rtol=1e-8, atol=1e-10), "Recomputed selected loss differs from curve/report")
                    summaries.append({"name": entry["name"], "variant": variant, "alpha": entry["selected_alpha" if variant == "main" else "shuffle_alpha"], "selection_loss_recomputed": loss, "selection_loss_preserved_minimum": curve_minimum, "normal_equation_max_relative_residual": float(relative[begin:end].max())})
                audit["readouts"].append({"name": readout, "feature_width": x.shape[1], "selected_models_checked": len(summaries), "normal_equation_max_relative_residual": float(relative.max()), "discovery_normalization_checked": True, "models": summaries})
                stage("audit_saved_models")
        stage1 = [row for row in result["stage1"] if row["kind"] == "residual"]
        winners = {definition: {group: min(stage1, key=lambda row: row["targets"][definition]["groups"][group]["main"]["selection"]["normalized_mse"])["name"] for group in ["all", "position", "velocity"]} for definition in ["absolute30", "role12"]}
        frozen_stage1 = read_json(args.output_dir / "stage1_selection.json")
        require(winners == result["stage1_residual_winners"] == selection["stage1_residual_winners"] == frozen_stage1["winners"], "Residual winner selection differs")
        require(digest(args.output_dir / "stage1_selection.json") == selection["stage1_selection_sha256"], "Stage-one frozen selection changed")
        require(all(digest(Path(path)) == expected for path, expected in frozen.items()) and all(digest(ROOT / path) == value for path, value in code.items()), "Analysis or inputs changed during audit")
        require(all(digest(ROOT / path) == value for path, value in manifest["code_sha256"].items()) and all(digest(Path(row["path"])) == row["sha256"] for row in records), "Capture code or feature archives changed during audit")
        audit.update(status="passed", development_selection_sha256=digest(selection_path), model_archive_sha256=digest(models_path), development_report_sha256=digest(report_path), capture_manifest_sha256=digest(args.capture_manifest), capture_audit_sha256=digest(args.capture_audit), analysis_code_sha256=code, registration_sha256=provenance["registration_sha256"], audit_script_sha256=digest(Path(__file__)), readouts_checked=len(groups), selected_models_checked=sum(row["selected_models_checked"] for row in audit["readouts"]), normal_equations_checked=True, discovery_normalization_checked=True, selection_losses_recomputed=True, selected_alphas_minimize_preserved_curves=True, residual_winners_checked=True)
        stage("complete")
    except Exception as exc:
        audit.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        stage(audit.get("stage", "failed"))
        raise


if __name__ == "__main__":
    main()
