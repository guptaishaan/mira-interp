#!/usr/bin/env python3
"""Export finished confirmation metrics and figures; never load data or fit models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESIDUAL_SITES = ["block_0_input"] + [f"block_{index}_output" for index in range(16)]
COMPARISONS = ("discovery_mean", "codec", "own_match_shuffled_control")


def finite_number(value, *, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError("metric must be finite, or explicitly null for an undefined R2")


def validate_report(report):
    if report.get("status") != "passed_observational_analysis" or report.get("registration_version") != 2:
        raise ValueError("requires a complete passed observational v2 confirmation report")
    if report.get("match_counts") != {"discovery": 31, "selection": 11, "confirmation": 11}:
        raise ValueError("confirmation match counts differ from the frozen qualified study")
    targets = report["target_names"]
    if len(targets) != 30 or len(set(targets)) != 30:
        raise ValueError("expected all30 uniquely named targets")
    sites = report["sites"]
    if [site["name"] for site in sites if site["kind"] == "residual"] != RESIDUAL_SITES:
        raise ValueError("the complete ordered17-site residual profile is required")
    if len(sites) != 19 or {site["name"] for site in sites if site["kind"] == "baseline"} != {"codec_X", "RGB_X"}:
        raise ValueError("expected17 residual sites and codec/RGB baselines")
    if report["frozen_selection_winner"] not in RESIDUAL_SITES:
        raise ValueError("the frozen selection winner is not a residual site")
    for metrics in [report["mean_baseline"], *(site[kind] for site in sites for kind in ("main", "shuffled"))]:
        if metrics["n_matches"] != 11 or metrics["bootstrap_unit"] != "whole_match":
            raise ValueError("metrics must use the11 confirmation matches as independent units")
        finite_number(metrics["normalized_mse"])
        if len(metrics["normalized_mse_ci95"]) != 2:
            raise ValueError("missing normalizedMSE interval")
        for value in metrics["normalized_mse_ci95"]:
            finite_number(value)
        for metric in ("mae", "rmse", "r2"):
            if len(metrics[metric]) != 30 or len(metrics[f"{metric}_ci95"]) != 30:
                raise ValueError("missing target metrics or confidence intervals")
            for value, interval in zip(metrics[metric], metrics[f"{metric}_ci95"]):
                finite_number(value, nullable=metric == "r2")
                if len(interval) != 2:
                    raise ValueError("a target interval is not a lower/upper pair")
                for endpoint in interval:
                    finite_number(endpoint, nullable=metric == "r2")
    for site in sites:
        if set(site["paired_comparisons"]) != set(COMPARISONS):
            raise ValueError("paired comparisons are incomplete")
        for comparison in site["paired_comparisons"].values():
            if comparison["n_matches"] != 11 or comparison["bootstrap_unit"] != "paired_whole_match":
                raise ValueError("paired gain must bootstrap entire matched confirmation units")
            finite_number(comparison["normalized_mse_gain"])
            if len(comparison["ci95"]) != 2:
                raise ValueError("missing paired-gain interval")
            for value in comparison["ci95"]:
                finite_number(value)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(report, output_dir):
    target_rows, site_rows = [], []
    for site in report["sites"]:
        for model_kind in ("main", "shuffled"):
            metrics = site[model_kind]
            for index, target in enumerate(report["target_names"]):
                row = {"site": site["name"], "site_kind": site["kind"], "model_kind": model_kind,
                       "target": target, "raw_error_units": "uu/s" if ".velocity." in target else "uu",
                       "confirmation_matches": metrics["n_matches"],
                       "selected_alpha": site["selected_alpha" if model_kind == "main" else "shuffle_alpha"]}
                for metric in ("mae", "rmse", "r2"):
                    row[metric] = metrics[metric][index]
                    row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = metrics[f"{metric}_ci95"][index]
                row["r2_bootstrap_valid_replicates"] = metrics["r2_bootstrap_valid_replicates"][index]
                target_rows.append(row)
        summary = {"site": site["name"], "site_kind": site["kind"], "confirmation_matches": 11,
                   "chosen_on_original_selection": site["name"] == report["frozen_selection_winner"],
                   "selected_alpha": site["selected_alpha"], "shuffle_alpha": site["shuffle_alpha"]}
        for model_kind in ("main", "shuffled"):
            metrics = site[model_kind]
            summary[f"{model_kind}_normalized_mse"] = metrics["normalized_mse"]
            summary[f"{model_kind}_normalized_mse_ci95_low"], summary[f"{model_kind}_normalized_mse_ci95_high"] = metrics["normalized_mse_ci95"]
        for name in COMPARISONS:
            paired = site["paired_comparisons"][name]
            summary[f"paired_gain_vs_{name}"] = paired["normalized_mse_gain"]
            summary[f"paired_gain_vs_{name}_ci95_low"], summary[f"paired_gain_vs_{name}_ci95_high"] = paired["ci95"]
        mean = report["mean_baseline"]
        summary["discovery_mean_normalized_mse"] = mean["normalized_mse"]
        summary["discovery_mean_normalized_mse_ci95_low"], summary["discovery_mean_normalized_mse_ci95_high"] = mean["normalized_mse_ci95"]
        summary["paired_gain_convention"] = "baseline minus main model; positive means lower main error"
        site_rows.append(summary)
    if len(target_rows) != 19 * 2 * 30 or len(site_rows) != 19:
        raise ValueError("export coverage is incomplete")
    target_path, site_path = output_dir / "target_metrics.csv", output_dir / "site_summary.csv"
    write_csv(target_path, target_rows)
    write_csv(site_path, site_rows)
    return [target_path, site_path]


def figures(report, figure_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    rows = [site for site in report["sites"] if site["kind"] == "residual"]
    winner = RESIDUAL_SITES.index(report["frozen_selection_winner"])
    x = np.arange(17)
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for kind, label, color in (("main", "True training labels", "#1765A8"),
                                ("shuffled", "Whole-match shuffled labels", "#C26824")):
        values = np.asarray([row[kind]["normalized_mse"] for row in rows])
        bounds = np.asarray([row[kind]["normalized_mse_ci95"] for row in rows])
        ax.plot(x, values, color=color, marker="o", markersize=4, linewidth=2, label=label)
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=.16, linewidth=0)
    ax.axhline(report["mean_baseline"]["normalized_mse"], color="#606670", linestyle="--", linewidth=1.5,
               label="Discovery-mean baseline")
    for name, label, color in (("codec_X", "Codec baseline (32D)", "#35854C"),
                                ("RGB_X", "RGB 8×8 baseline (192D)", "#8A559A")):
        site = next(row for row in report["sites"] if row["name"] == name)
        ax.axhline(site["main"]["normalized_mse"], color=color, linestyle=":", linewidth=2, label=label)
    ax.axvline(winner, color="#1A2029", linewidth=1, linestyle="--", alpha=.6)
    ax.scatter([winner], [rows[winner]["main"]["normalized_mse"]], marker="*", s=145, color="#1A2029",
               zorder=5, label="Site chosen on selection")
    ax.set_xticks(x, ["Input"] + [str(i) for i in range(16)])
    ax.set_xlim(-.35, 16.35)
    ax.set_xlabel("Block-0 input, followed by each residual-block output")
    ax.set_ylabel("Confirmation normalized MSE (lower is better)")
    ax.grid(axis="y", alpha=.16)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Contemporaneously indexed state-annotation decoding", fontsize=16, y=.98)
    ax.set_title("11 confirmation matches · teacher-forced reconstruction at τ = 0.5 · target pixels present",
                 fontsize=10, pad=13)
    fig.text(.5, .02, "Shading: 95% whole-match bootstrap intervals. Descriptive uncertainty; no predictive or causal claim.",
             ha="center", fontsize=9, color="#414853")
    fig.subplots_adjust(left=.09, right=.98, top=.87, bottom=.25)
    paths = []
    for suffix in ("png", "pdf"):
        path = figure_dir / f"observational_layers.{suffix}"
        fig.savefig(path, dpi=220)
        paths.append(path)
    plt.close(fig)

    values = np.asarray([[np.nan if value is None else value for value in row["main"]["r2"]] for row in rows])
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("cannot plot an entirely undefined R2 profile")
    low, high = min(0., float(finite.min())), max(1., float(finite.max()))
    normalization = TwoSlopeNorm(vmin=low, vcenter=0., vmax=high) if low < 0 else Normalize(vmin=0., vmax=high)
    color_map = plt.get_cmap("RdBu").copy()
    color_map.set_bad("#D3D5D8")
    fig, ax = plt.subplots(figsize=(19, 9.4))
    plot = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest", cmap=color_map,
                     norm=normalization)
    ax.set_xticks(np.arange(30), report["target_names"], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(17), ["Block 0 input"] + [f"Block {i} output" for i in range(16)], fontsize=9)
    for boundary in (5.5, 11.5, 17.5, 23.5):
        ax.axvline(boundary, color="white", linewidth=2.3)
    for boundary in (2.5, 8.5, 14.5, 20.5, 26.5):
        ax.axvline(boundary, color="white", linewidth=.65, linestyle="--", alpha=.8)
    for index, name in enumerate(("Ball", "Player 0", "Player 1", "Player 2", "Player 3")):
        ax.text(index * 6 + 2.5, 1.018, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=11, fontweight="semibold")
    ax.add_patch(Rectangle((-.5, winner - .5), 30, 1, fill=False, edgecolor="#161B22", linewidth=1.6))
    color_bar = fig.colorbar(plot, ax=ax, pad=.015, fraction=.028)
    ticks = ([low, low / 2] if low < 0 else []) + list(np.linspace(0., high, 6))
    color_bar.set_ticks(ticks, labels=[f"{tick:.3f}" if tick < 0 else f"{tick:.1f}" for tick in ticks])
    color_bar.set_label("R² · full observed range, including negative values")
    ax.set_xlabel("Physical-state annotations (position XYZ, then velocity XYZ, within each entity)", labelpad=12)
    fig.suptitle("Confirmation R² for all 17 residual sites and 30 targets", fontsize=17, y=.98)
    fig.text(.5, .936, "Contemporaneously indexed annotation decoding · 11 confirmation matches · target pixels present",
             ha="center", fontsize=11)
    detail = "Outlined row: site chosen on selection. Negative R² is shown without clipping; gray means undefined."
    fig.text(.5, .018, detail + " No predictive or causal claim.", ha="center", fontsize=9, color="#414853")
    fig.subplots_adjust(left=.085, right=.96, top=.855, bottom=.265)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"observational_targets.{suffix}"
        fig.savefig(path, dpi=220)
        paths.append(path)
    plt.close(fig)
    return paths, {"heatmap_observed_min_r2": float(finite.min()), "heatmap_observed_max_r2": float(finite.max()),
                   "heatmap_color_min": low, "heatmap_color_max": high, "undefined_r2_cells": int(np.isnan(values).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, default=ROOT / "results/probes_v2/confirmation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/probes_v2")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    raw = args.confirmation.read_bytes()
    report = json.loads(raw)
    validate_report(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = tables(report, args.output_dir)
    plots, display = figures(report, args.figure_dir)
    outputs.extend(plots)
    if args.confirmation.read_bytes() != raw:
        raise ValueError("confirmation report changed during export")
    audit = {"status": "completed_format_only_export", "created_utc": datetime.now(timezone.utc).isoformat(),
             "source_confirmation_sha256": hashlib.sha256(raw).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "source_registration_sha256": report["registration_sha256"],
             "source_frozen_selection_sha256": report["frozen_selection_sha256"],
             "frozen_selection_winner": report["frozen_selection_winner"],
             "target_csv_rows": 1140, "site_csv_rows": 19, "confirmation_matches": 11,
             "raw_data_or_activations_loaded": False, "models_fit_or_reselected": False,
             "confidence_intervals": "copied from completed report; no recomputation",
             "r2_csv_nulls": "empty fields preserve undefined R2 values and bounds", **display,
             "outputs": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                         {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in outputs}}
    (args.output_dir / "export_manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
