#!/usr/bin/env python3
"""Compare development readouts on original discovery/selection matches only.

No confirmation row is accepted. Stage one compares every registered mean and
spatial residual site plus visual baselines. Optional stage two adds causal
differences at the stage-one all-target residual winner for each target definition.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira_interp.development import (ALPHAS, SEED, causal_row_indices, check_development_roles,
    fit_readout, residual_winners, role_targets, target_definitions, temporal_features)
from mira_interp.probes import load_selected_models, save_selected_models

CODE_FILES = ("scripts/analyze_development.py", "src/mira_interp/development.py", "src/mira_interp/probes.py")
WIDTHS = {"mean_X": (17, 2048), "spatial_X": (17, 1536), "codec_spatial_X": (4608,),
          "codec_mean_X": (32,), "RGB_X": (1536,)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, payload, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def load_captures(manifest_path, audit_path, registration_path, split_path):
    """Validate split metadata before materializing any per-clip target values."""
    manifest, audit, protocol, split = (json.loads(path.read_text()) for path in
                                      (manifest_path, audit_path, registration_path, split_path))
    require(manifest.get("status") == audit.get("status") == "passed", "Development capture must pass its audit")
    protocol_hash = sha(registration_path)
    combined = manifest_path.resolve() == audit_path.resolve()
    if not combined:
        require(audit.get("capture_manifest_sha256", audit.get("manifest_sha256")) == sha(manifest_path), "Audit is not bound to this capture manifest")
    require(audit.get("registration_sha256", audit.get("protocol_sha256")) == protocol_hash
            and manifest.get("registration_sha256", manifest.get("protocol_sha256")) == protocol_hash,
            "Capture registration mismatch")
    require(manifest.get("confirmation_used") is False, "Manifest must explicitly exclude old confirmation")
    require(protocol["analysis"]["ridge_alphas"] == list(ALPHAS)
            and protocol["analysis"]["scored_latents"] == list(range(2, 8)), "Analysis differs from registered development settings")
    if not combined:
        require(audit.get("all_source_label_joins_exact") is True and audit.get("all_layers_complete") is True,
                "Exact source joins and all-layer capture must be independently audited")
        require(audit.get("split_manifest_sha256") == sha(split_path), "Original role manifest mismatch")
    assignments = {row["match_id"]: row for row in split["matches"]}
    require(len(assignments) == len(split["matches"]), "Duplicate original match assignment")
    expected = {mid: row for mid, row in assignments.items() if row["split"] in {"discovery", "selection"}}
    counts = {role: sum(row["split"] == role for row in expected.values()) for role in ("discovery", "selection")}
    require(counts == {"discovery": 31, "selection": 11}, "Expected the original31/11 development matches")
    records = manifest.get("records", manifest.get("clips", []))
    require(records and len({row["clip_id"] for row in records}) == len(records), "Missing/duplicate capture records")
    # Complete manifest role audit first, before opening any NPZ target array.
    for row in records:
        role = row.get("role", row.get("split"))
        require(role in {"discovery", "selection"}, "Confirmation/pilot capture is forbidden")
        require(row["match_id"] in expected and role == expected[row["match_id"]]["split"], "Capture changes original match role")
    require({row["match_id"] for row in records} == set(expected), "Development cohort is incomplete")
    require(all(sum(row["match_id"] == match for row in records) == 8 for match in expected), "Each development match needs eight clips")
    source_manifest_path = ROOT / protocol["cohort"]["source_manifest"]
    source_manifest = json.loads(source_manifest_path.read_text())
    source_audit_path = ROOT / "results/qualified_data_audit.json"
    source_audit = json.loads(source_audit_path.read_text())
    require(source_audit["status"] == "passed" and source_audit["clip_manifest_sha256"] == sha(source_manifest_path)
            and source_audit["split_manifest_sha256"] == sha(split_path), "Original source/split audit lineage differs")
    source_records = {row["clip_id"]: row for row in source_manifest["records"] if row["match_id"] in expected}
    require(set(source_records) == {row["clip_id"] for row in records}, "Capture is not the exact original development clip set")
    preparation_path = ROOT / "results/development_prepare.json"
    preparation = json.loads(preparation_path.read_text())
    require(preparation.get("status") == "passed" and not preparation.get("errors"), "Development preparation did not pass")
    prepared = {row["clip_id"]: row for row in preparation["records"]}
    require(len(prepared) == len(preparation["records"]) and set(prepared) == set(source_records), "Prepared development clip coverage differs")
    capture_hashes = manifest["code_sha256"]
    require(capture_hashes and {name: sha(ROOT / name) for name in capture_hashes} == capture_hashes, "Capture code changed after capture")
    pieces, inputs, sites, target_names = {}, [], None, None
    for row in sorted(records, key=lambda value: (value["match_id"], value["clip_id"])):
        path = Path(row.get("output_path", row.get("path", "")))
        if not path.is_absolute():
            path = manifest_path.parent / path
        expected_hash = row.get("output_sha256", row.get("sha256"))
        require(sha(path) == expected_hash, f"Capture bytes changed: {row['clip_id']}")
        role = row.get("role", row.get("split"))
        require(row.get("registration_sha256", row.get("protocol_sha256")) == protocol_hash
                and row.get("code_sha256") == capture_hashes, "Per-clip registration/code differs")
        source_row = source_records[row["clip_id"]]
        require(source_row["match_id"] == row["match_id"] and source_row["role"] == role, "Original clip identity/role changed")
        prepared_row = prepared[row["clip_id"]]
        require(prepared_row["match_id"] == row["match_id"] and prepared_row["split"] == role
                and prepared_row["parent_sha256"] == source_row["sha256"]
                and prepared_row["sha256"] == row["input_sha256"], "Fractional input/source lineage differs")
        require(all(prepared_row.get(key) is True for key in ("source_pts_exact", "round_to_previous_pixels_exact", "row_metadata_unchanged")), "Fractional input integrity gate is incomplete")
        with np.load(path, allow_pickle=False) as archive:
            # NPZ roles are checked before y is decoded, even if its manifest lies.
            require(archive["split"].tolist() == [role] * 32, "Per-row partition mismatch; confirmation is forbidden")
            require(archive["match_ids"].tolist() == [row["match_id"]] * 32 and archive["clip_ids"].tolist() == [row["clip_id"]] * 32, "Per-row match/clip mismatch")
            require(np.array_equal(archive["view_index"], np.repeat(np.arange(4), 8))
                    and np.array_equal(archive["latent_frame_index"], np.tile(np.arange(8), 4)), "Rows must preserve all view/time identities")
            values = {key: archive[key] for key in (*WIDTHS, "y", "view_index", "latent_frame_index", "match_ids", "split", "clip_ids")}
            these_sites, these_targets = archive["sites"].tolist(), archive["target_names"].tolist()
            if sites is None:
                sites, target_names = these_sites, these_targets
            require(sites == these_sites and target_names == these_targets, "Inconsistent column identity")
            for key in ("timestamps", "source_frame_index", "canonical_player_ids", "player_ids"):
                if key in archive.files:
                    values[key] = archive[key]
        for key, tail in WIDTHS.items():
            require(values[key].shape == (32, *tail) and np.isfinite(values[key]).all(), f"Invalid {key} capture")
        require(values["y"].shape == (32, 30) and np.isfinite(values["y"]).all(), "Invalid target capture")
        if "canonical_player_ids" in values:
            require(np.array_equal(values["canonical_player_ids"], np.tile(expected[row["match_id"]]["player_ids"], (32, 1))), "Canonical player mapping changed")
        if "player_ids" in values:
            require(values["player_ids"].tolist() == expected[row["match_id"]]["player_ids"], "Canonical player identities changed")
        source_path = Path(source_row["artifact_path"])
        require(sha(source_path) == source_row["sha256"], "Original source clip hash changed")
        with np.load(source_path, allow_pickle=False) as source:
            require(np.array_equal(values["y"], source["targets"][:, 1::2].reshape(32, 30)), "Targets differ from exact original own-view source rows")
            require(np.array_equal(values.get("timestamps"), source["timestamps"][:, 1::2].reshape(32)), "Target timestamps differ from source")
            indices = source["source_frame_indices"]
            if indices.shape == (16,):
                indices = np.broadcast_to(indices, (4, 16))
            require(np.array_equal(values.get("source_frame_index"), indices[:, 1::2].reshape(32)), "Target source-frame indices differ")
            require(source["target_names"].tolist() == target_names and source["player_ids"].tolist() == expected[row["match_id"]]["player_ids"], "Original target/player names differ")
        for key in ("timestamps", "source_frame_index"):
            if key in values:
                require(values[key].shape == (32,) and np.isfinite(values[key]).all(), f"Invalid {key}")
                differences = np.diff(values[key].reshape(4, 8), axis=1)
                require(np.all(differences > 0), "Noncausal within-view timing")
                if key == "source_frame_index":
                    require(np.all(differences == 2), "Latent pairs must advance by two source frames")
        for key in (*WIDTHS, "y", "view_index", "latent_frame_index", "match_ids", "split", "clip_ids"):
            pieces.setdefault(key, []).append(values[key])
        inputs.append({"clip_id": row["clip_id"], "path": str(path), "sha256": expected_hash})
    require(sites == ["block_0_input"] + [f"block_{i}_output" for i in range(16)], "All17 sites required in canonical order")
    reference = json.loads((ROOT / "configs/observational_probe_v2.json").read_text())
    require(target_names == reference["targets"], "Canonical absolute target order changed")
    data = {key: np.concatenate(value) for key, value in pieces.items()}
    data.update(sites=sites, target_names=target_names)
    check_development_roles(data["match_ids"], data["split"])
    return data, {"registration_sha256": protocol_hash, "capture_manifest_sha256": sha(manifest_path),
                  "capture_audit_sha256": sha(audit_path), "split_manifest_sha256": sha(split_path),
                  "match_counts": counts, "clips": inputs, "protocol": protocol,
                  "source_manifest_sha256": sha(source_manifest_path), "source_audit_sha256": sha(source_audit_path),
                  "preparation_manifest_sha256": sha(preparation_path),
                  "capture_code_sha256": capture_hashes, "all_source_label_joins_exact": True,
                  "all_layers_complete": True, "confirmation_arrays_loaded": False}


def plot_report(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, definition in enumerate(("absolute30", "role12")):
        for j, group in enumerate(("all", "position", "velocity")):
            axis = axes[i, j]
            for summary in ("mean", "spatial"):
                rows = [row for row in report["stage1"] if row["kind"] == "residual" and row["feature_spec"]["summary"] == summary]
                axis.plot(range(17), [row["targets"][definition]["groups"][group]["main"]["selection"]["normalized_mse"] for row in rows], marker=".", label=summary)
            for row in report["stage1"]:
                if row["kind"] == "baseline":
                    axis.axhline(row["targets"][definition]["groups"][group]["main"]["selection"]["normalized_mse"], linestyle=":", label=row["name"])
            axis.set(title=f"{definition}: {group}", xlabel="Site: input, then block outputs", ylabel="Development selection normalized MSE")
            axis.legend(fontsize=7)
    fig.suptitle("Development comparisons only — choices use these selection matches")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-audit", type=Path, help="Optional separate audit; defaults to the combined capture manifest/audit")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "data/qualified_split_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temporal-stage", choices=("selected", "off"), default="selected")
    args = parser.parse_args()
    args.capture_audit = args.capture_audit or args.capture_manifest
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require(not any((args.output_dir / name).exists() for name in ("stage1_selection.json", "development_selection.json", "development_models.npz", "development_report.json")), "Use a new output directory; prior results are immutable")
    code_hashes = {name: sha(ROOT / name) for name in CODE_FILES}
    data, provenance = load_captures(args.capture_manifest, args.capture_audit, args.registration, args.split_manifest)
    current, previous = causal_row_indices(data["clip_ids"], data["view_index"], data["latent_frame_index"])
    joined = np.concatenate((data["y"], role_targets(data["y"], data["view_index"])), axis=1)[current]
    ids, roles = data["match_ids"][current], data["split"][current]
    definitions = target_definitions(data["target_names"])
    specs = [(f"{summary}/{site}", "residual", {"array": key, "site_index": index, "summary": summary, "temporal": "current"})
             for summary, key in (("mean", "mean_X"), ("spatial", "spatial_X")) for index, site in enumerate(data["sites"])]
    specs += [(key, "baseline", {"array": key, "site_index": None, "summary": key, "temporal": "current"})
              for key in ("codec_mean_X", "codec_spatial_X", "RGB_X")]
    report = {"status": "running", "scope": "development_only_teacher_forced_annotation_decoding",
              "created_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
              "analysis_code_sha256": code_hashes, "alpha_grid": list(ALPHAS), "seed": SEED,
              "target_convention": "ego absolute WORLD axes; ball-minus-ego relative WORLD axes, not camera heading",
              "row_selection": "latent indices2..7; eight clips x four views x six rows =192rows/match",
              "n_rows": len(current), "stage1": [], "stage2": [], "confirmation_used": False,
              "new_confirmation_required": True, "models_refitted_with_selection": False,
              "selection_rule": "Separate alpha and residual winner for each target definition and all/position/velocity family; ties use alpha then representation/site order",
              "stage2_rule": "current and current-minus-previous only at each definition's stage1 all-target residual winner; at most two unique residual sites; both definitions reported at each; same development selection reused",
              "limits": ["Both stages are adaptive development on these selection matches, not fresh confirmation.",
                         "Original confirmation is refused; a new untouched cohort is required for new confirmatory claims.",
                         "Target pixels and clean observed past are supplied; not future-state prediction or physical control.",
                         "Readout dimensions differ; spatial summaries and temporal concatenation change capacity.",
                         "Relative targets use labels only to define outputs; no true state is appended to features.",
                         "No confidence or significance claim is attached to development winner selection."]}
    models, records = [], []

    def run(name, kind, spec):
        print(f"Development: {name}", flush=True)
        full = data[spec["array"]]
        if spec["site_index"] is not None:
            full = full[:, spec["site_index"], :]
        x = temporal_features(full, current, previous) if spec["temporal"] == "current_and_delta" else full[current]
        result = fit_readout(x, joined, ids, roles, definitions, name=name, kind=kind, feature_spec=spec)
        models.extend(result.models)
        records.extend(result.model_records)
        return result.report

    for name, kind, spec in specs:
        report["stage1"].append(run(name, kind, spec))
        write(args.output_dir / "status.json", {"status": "running_development_stage1", "readouts_completed": len(report["stage1"]), "readouts_expected": len(specs)})
    report["stage1_residual_winners"] = residual_winners(report["stage1"])
    stage1_path = args.output_dir / "stage1_selection.json"
    write(stage1_path, {"status": "frozen_development_stage1", "winners": report["stage1_residual_winners"],
                       "analysis_code_sha256": code_hashes, "provenance": provenance, "stage2_rule": report["stage2_rule"]}, exclusive=True)
    if args.temporal_stage == "selected":
        names = list(dict.fromkeys(report["stage1_residual_winners"][definition]["all"] for definition in definitions))
        by_name = {name: (kind, spec) for name, kind, spec in specs}
        for name in names:
            kind, spec = by_name[name]
            report["stage2"].append(run(name + "/current_and_delta", kind, {**spec, "temporal": "current_and_delta"}))
    require({name: sha(ROOT / name) for name in CODE_FILES} == code_hashes, "Analysis code changed during development")
    require(sha(args.capture_manifest) == provenance["capture_manifest_sha256"] and sha(args.capture_audit) == provenance["capture_audit_sha256"], "Capture provenance changed during analysis")
    require(sha(args.registration) == provenance["registration_sha256"] and sha(args.split_manifest) == provenance["split_manifest_sha256"], "Registered protocol or original roles changed during analysis")
    require(sha(ROOT / "results/development_prepare.json") == provenance["preparation_manifest_sha256"], "Preparation provenance changed during analysis")
    model_path = args.output_dir / "development_models.npz"
    save_selected_models(models, model_path)
    selection = {"status": "frozen_development_choices", "models": records, "stage1_residual_winners": report["stage1_residual_winners"],
                 "model_archive_sha256": sha(model_path), "stage1_selection_sha256": sha(stage1_path), "analysis_code_sha256": code_hashes,
                 "provenance": provenance, "confirmation_used": False, "new_confirmation_required": True,
                 "feature_note": "Saved model predicts its named target subset from the exact feature_spec; normalization is discovery-only."}
    load_selected_models(model_path, selection)  # Verify actual saved coefficient fingerprints.
    write(args.output_dir / "development_selection.json", selection, exclusive=True)
    report.update(status="passed_development_analysis", model_archive_sha256=sha(model_path),
                  development_selection_sha256=sha(args.output_dir / "development_selection.json"))
    write(args.output_dir / "development_report.json", report, exclusive=True)
    plot_report(report, args.output_dir / "development_profiles.png")
    write(args.output_dir / "status.json", {"status": report["status"], "new_confirmation_required": True,
          "readouts_completed": len(report["stage1"]) + len(report["stage2"]), "models_saved": len(models),
          "development_report_sha256": sha(args.output_dir / "development_report.json")})
    print("Development analysis complete; new confirmation remains required.", flush=True)


if __name__ == "__main__":
    main()
