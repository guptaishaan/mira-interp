#!/usr/bin/env python3
"""Gated observational ridge analysis. Selection is persisted/reloaded before confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

# These limits apply only to this process and are set before importing NumPy.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "8"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira_interp.probes import (
    MINIMUM_MATCHES, load_selected_models, match_bootstrap_metrics, paired_mse_gain,
    save_selected_models, select_models, validate_dataset,
)

APPROVED_REGISTRATIONS = {
    "2437f915ceb14a10c33a6b003b7273ee6ba135e8495d4b15b03ff72f0edbd6a3": 1,
    "e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2": 2,
}
SEED = 20260906
BOOTSTRAP_REPLICATES = 500


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def validate_audit(audit, data_hash, registration_hash):
    if registration_hash not in APPROVED_REGISTRATIONS:
        raise ValueError("Unknown or changed observational registration")
    if audit.get("status") != "passed" or audit.get("analysis_npz_sha256") != data_hash:
        raise ValueError("A passed capture/data audit tied to this exact NPZ is required")
    for name in ("real_data", "video_alignment_checked", "physics_alignment_checked",
                 "all_layers_complete", "match_splits_disjoint", "checkpoint_integrity_checked"):
        if audit.get(name) is not True:
            raise ValueError(f"Missing passed prerequisite: {name}")
    if audit.get("registration_sha256") != registration_hash:
        raise ValueError("Capture audit does not refer to the selected frozen observational registration")


def analysis_code_hashes():
    return {name: file_hash(ROOT / name) for name in
            ("scripts/analyze_probes.py", "src/mira_interp/probes.py")}


def validate_selection_audit(audit, frozen, frozen_hash, helper_hash):
    if audit.get("status") != "passed" or audit.get("frozen_selection_sha256") != frozen_hash:
        raise ValueError("A passed independent audit of this frozen selection is required")
    for name in ("input_npz_sha256", "registration_sha256", "capture_audit_sha256",
                 "model_archive_sha256", "analysis_code_sha256", "match_counts"):
        if audit.get(name) != frozen[name]:
            raise ValueError(f"Selection audit provenance mismatch: {name}")
    if (audit.get("audit_script_sha256") != helper_hash
            or audit.get("frozen_winner") != frozen["winner"]
            or audit.get("confirmation_target_values_used") is not False
            or audit.get("models_refitted") is not False
            or audit.get("ridge_normal_equations_checked") is not True):
        raise ValueError("Independent selection audit checks are incomplete or changed")


def plot_profile(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in report["sites"] if row["kind"] == "residual"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, label, color in (("main", "True training labels", "#246BCE"),
                              ("shuffled", "Whole-match shuffled training labels", "#A85E35")):
        values = np.array([row[key]["normalized_mse"] for row in rows])
        bounds = np.array([row[key]["normalized_mse_ci95"] for row in rows])
        ax.plot(range(len(rows)), values, marker="o", color=color, label=label)
        ax.fill_between(range(len(rows)), bounds[:, 0], bounds[:, 1], alpha=0.15, color=color)
    ax.axhline(report["mean_baseline"]["normalized_mse"], linestyle="--", color="gray", label="Discovery mean")
    for row in report["sites"]:
        if row["kind"] == "baseline":
            ax.axhline(row["main"]["normalized_mse"], linestyle=":", label=row["name"])
    winner = [row["name"] for row in rows].index(report["frozen_selection_winner"])
    ax.axvline(winner, color="black", alpha=0.3, label="Site chosen on selection")
    ax.set_xticks(range(len(rows)), ["Input"] + [str(i) for i in range(len(rows) - 1)])
    ax.set_xlabel("Residual site: block-0 input, then every block output")
    ax.set_ylabel("Confirmation standardized MSE (lower is better)")
    ax.set_title("Contemporaneously indexed state-annotation decoding at tau=0.5\nTarget pixels are present; no predictive or causal claim")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("select", "confirm"), default="select")
    parser.add_argument("--registration", type=Path, default=ROOT / "configs/observational_probe_v2.json")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_code_hashes = analysis_code_hashes()
    registration_hash = file_hash(args.registration)
    if registration_hash not in APPROVED_REGISTRATIONS:
        raise ValueError("Registration has changed; an explicit protocol amendment is required")
    registration = json.loads(args.registration.read_text())
    data_hash = file_hash(args.data)
    audit = json.loads(args.audit.read_text())
    validate_audit(audit, data_hash, registration_hash)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    counts = validate_dataset(data)
    if any(counts[split] < minimum for split, minimum in MINIMUM_MATCHES.items()):
        write_json(args.output_dir / "status.json", {"status": "blocked_underpowered", "match_counts": counts,
                   "minimum_match_counts": MINIMUM_MATCHES, "research_analysis_run": False})
        return 2
    if any(counts[split] != registration["cohort"][split] for split in counts):
        raise ValueError("Match cohort differs from the frozen registration")
    if list(data["sites"].astype(str)) != registration["capture"]["sites"] or list(data["target_names"].astype(str)) != registration["targets"]:
        raise ValueError("Site/target ordering differs from the frozen registration")
    if data["X"].shape[2] != 2048 or data["codec_X"].shape[1] != 32:
        raise ValueError("Unexpected residual/codec feature width")
    if "RGB_X" in data and data["RGB_X"].shape[1] != 192:
        raise ValueError("Unexpected RGB descriptor width")
    ids = data["match_ids"].astype(str)
    if any(np.count_nonzero(ids == match) != 256 for match in np.unique(ids)):
        raise ValueError("Each registered match must supply exactly eight clips x four views x eight latent pairs")
    frozen_path = args.output_dir / "frozen_selection.json"
    models_path = args.output_dir / "frozen_models.npz"
    if args.phase == "select":
        if frozen_path.exists() or models_path.exists():
            raise FileExistsError("Selection artifacts already exist; use --phase confirm or a new output directory")
        selected, frozen = select_models(data, seed=SEED, progress=lambda site: print(f"Discovery/selection: {site}", flush=True))
        if analysis_code_hashes() != initial_code_hashes:
            raise ValueError("Analysis code changed during discovery/selection")
        save_selected_models(selected, models_path)
        frozen.update({"schema_version": 1, "evidence_level": "observational_teacher_forced_decoding_only",
                       "created_utc": datetime.now(timezone.utc).isoformat(),
                       "registration_sha256": registration_hash, "input_npz_sha256": data_hash,
                       "registration_version": APPROVED_REGISTRATIONS[registration_hash],
                       "cohort_amendment": registration["cohort"].get("amendment"),
                       "capture_audit_sha256": file_hash(args.audit), "model_archive_sha256": file_hash(models_path),
                       "analysis_code_sha256": initial_code_hashes,
                       "match_counts": counts, "target_names": list(data["target_names"].astype(str)),
                       "confirmation_used_for_selection": False, "refit_on_selection": False,
                       "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": SEED,
                       "numpy_version": np.__version__, "python_version": sys.version.split()[0]})
        write_json(frozen_path, frozen, exclusive=True)
        del selected
        print(f"Frozen selection saved: {frozen_path} SHA256={file_hash(frozen_path)}", flush=True)
        return 0

    # Confirmation always reloads the saved choices and coefficients. No refitting.
    frozen = json.loads(frozen_path.read_text())
    if (frozen["input_npz_sha256"] != data_hash or frozen["registration_sha256"] != registration_hash
            or frozen["capture_audit_sha256"] != file_hash(args.audit)
            or frozen["model_archive_sha256"] != file_hash(models_path)
            or frozen["analysis_code_sha256"] != initial_code_hashes):
        raise ValueError("Frozen selection provenance no longer matches the inputs or model archive")
    selection_audit_path = args.output_dir / "selection_audit.json"
    if not selection_audit_path.exists():
        raise ValueError("Run scripts/audit_probe_selection.py successfully before confirmation")
    selection_audit = json.loads(selection_audit_path.read_text())
    validate_selection_audit(selection_audit, frozen, file_hash(frozen_path),
                             file_hash(ROOT / "scripts/audit_probe_selection.py"))
    selected = load_selected_models(models_path, frozen)
    output_path = args.output_dir / "confirmation.json"
    if output_path.exists():
        raise FileExistsError("Confirmation result already exists; it will not be overwritten")
    confirmation = data["split"].astype(str) == "confirmation"
    y, match_ids = data["y"][confirmation], ids[confirmation]
    baseline_model = selected[0].main
    baseline = np.broadcast_to(baseline_model.y_mean, y.shape)
    def metrics(prediction):
        return match_bootstrap_metrics(y, prediction, match_ids, baseline_model.y_scale,
                                       seed=SEED, replicates=BOOTSTRAP_REPLICATES)
    report = {"status": "passed_observational_analysis", "scientific_scope": registration["scope"],
              "measurement": "Linear decoding of the dataset's contemporaneously indexed physical-state annotations from teacher-forced observed-video representations",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": registration_hash, "input_npz_sha256": data_hash,
              "registration_version": APPROVED_REGISTRATIONS[registration_hash],
              "cohort_amendment": registration["cohort"].get("amendment"),
              "frozen_selection_sha256": file_hash(frozen_path), "model_archive_sha256": file_hash(models_path),
              "selection_audit_sha256": file_hash(selection_audit_path),
              "analysis_code_sha256": frozen["analysis_code_sha256"],
              "frozen_selection_winner": frozen["winner"], "match_counts": counts,
              "target_names": list(data["target_names"].astype(str)),
              "mean_baseline": metrics(baseline), "sites": [],
              "limits": ["Teacher-forced noisy reconstruction includes target pixels; not future-state prediction.",
                         "Model-training match manifest unavailable; interpretability-held-out is not proven unseen to the world model.",
                         f"Only {counts['confirmation']} confirmation matches; intervals are descriptive and do not cover model-fit or seed uncertainty.",
                         "Own-view video/state correspondence relies partly on the publisher's export contract; zero visual-state latency and exact 3D alignment have not been independently established.",
                         "All-site/target profiles do not imply family-wide significance.",
                         "Negative controls compare readout fitting, not representation causality.",
                         "Controlled-pair, independent-evaluator, causal, geometry, sparse-feature, and steering gates remain unmet."],
              "causal_stage_ready": False}
    amendment = registration["cohort"].get("amendment")
    if isinstance(amendment, dict):
        report["limits"].extend(amendment.get("limits", []))
    predictions, shuffled_predictions = {}, {}
    for index, site in enumerate(selected):
        print(f"Frozen confirmation: {site.name}", flush=True)
        if site.kind == "residual":
            features = data["X"][confirmation, index, :]
        else:
            features = data[site.name][confirmation]
        predictions[site.name] = site.main.predict(features)
        shuffled_predictions[site.name] = site.shuffled.predict(features)
        report["sites"].append({"name": site.name, "kind": site.kind,
                                "selected_alpha": site.main.alpha, "shuffle_alpha": site.shuffled.alpha,
                                "main": metrics(predictions[site.name]),
                                "shuffled": metrics(shuffled_predictions[site.name])})
    for row in report["sites"]:
        row["paired_comparisons"] = {
            name: paired_mse_gain(y, predictions[row["name"]], reference, match_ids, baseline_model.y_scale,
                                  seed=SEED, replicates=BOOTSTRAP_REPLICATES)
            for name, reference in (("discovery_mean", baseline), ("codec", predictions["codec_X"]),
                                    ("own_match_shuffled_control", shuffled_predictions[row["name"]]))}
    if analysis_code_hashes() != initial_code_hashes:
        raise ValueError("Analysis code changed during confirmation")
    write_json(output_path, report, exclusive=True)
    plot_profile(report, args.output_dir / "layer_profile.png")
    write_json(args.output_dir / "status.json", {"status": report["status"], "causal_stage_ready": False,
               "confirmation_sha256": file_hash(output_path), "frozen_selection_sha256": file_hash(frozen_path)})
    print(f"Complete observational report: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
