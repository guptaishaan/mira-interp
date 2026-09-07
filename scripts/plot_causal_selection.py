#!/usr/bin/env python3
"""Plot audited exact effects, paired controls, collateral shifts and AtP agreement."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"self": "Self replacement", "exact_donor": "Full donor", "component_transfer": "Height component",
          "ablate_to_discovery_mean": "Height ablation", "restore": "Restoration", "norm_matched_random": "Random: component norm",
          "norm_matched_random_full": "Random: full donor norm", "norm_matched_random_ablation": "Random: ablation norm",
          "wrong_variable_ball_x": "Wrong variable: ball x", "wrong_view_component": "Other view: same component"}


def save(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def dots_and_interval(axis, rows, *, paired=False, scale=1.):
    for index, row in enumerate(rows):
        mean_key = "mean_error_gain_above_control" if paired else "mean_standardized_error_gain"
        ci_key = "paired_match_bootstrap_ci95" if paired else "match_bootstrap_ci95"
        values_key = "match_differences" if paired else "match_gains"
        values = [value*scale for value in row[values_key].values()]
        axis.scatter(values, index+np.linspace(-.18, .18, len(values)), s=13, color="#555555", alpha=.55)
        low, high = np.asarray(row[ci_key])*scale
        axis.plot([low, high], [index, index], color="#1764a5", linewidth=3)
        axis.scatter([row[mean_key]*scale], [index], s=40, color="#1764a5", marker="D", zorder=4)
    axis.axvline(0, color="black", linewidth=.7)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/causal_development_v1")
    args = parser.parse_args()
    audit_path = args.directory / "selection_audit.json"
    audit = json.loads(audit_path.read_text())
    result_path = args.directory / "selection_results.json"
    if audit["status"] != "passed" or audit["selection_results_sha256"] != hashlib.sha256(result_path.read_bytes()).hexdigest():
        raise ValueError("Exact passed selection audit is required")
    sites = [row["site"] for row in audit["attribution_vs_exact"]]
    conditions = list(LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for axis, site in zip(axes, sites):
        rows = [next(row for row in audit["comparisons"] if row["site"] == site and row["condition"] == condition) for condition in conditions]
        dots_and_interval(axis, rows)
        axis.set(title=site, yticks=range(len(conditions)), yticklabels=[LABELS[c] for c in conditions], xlabel="Standardized donor-target error gain")
    fig.suptitle("Exact internal interventions: all 11 matches, two seeds averaged within each match")
    fig.text(.5, .015, "Dots: matched observations. Diamond/line: mean and descriptive whole-match bootstrap 95% interval. No physical-control claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    save(fig, args.directory, "selection_exact_effects")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, site in zip(axes, sites):
        rows = [row for row in audit["paired_control_comparisons"] if row["site"] == site]
        dots_and_interval(axis, rows, paired=True)
        labels = [LABELS[row["intervention"]]+" − "+LABELS[row["control"]] for row in rows]
        axis.set(title=site, yticks=range(len(rows)), yticklabels=labels, xlabel="Error gain above matched control")
    fig.suptitle("Paired specificity comparisons; positive favors the named intervention")
    fig.text(.5, .015, "Same 11 matches and paired seeds. Intervals are descriptive and conditional on development choices.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, args.directory, "selection_paired_controls")

    small_conditions = [condition for condition in conditions if condition not in ("exact_donor", "norm_matched_random_full")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for axis, site in zip(axes, sites):
        rows = [next(row for row in audit["comparisons"] if row["site"] == site and row["condition"] == condition) for condition in small_conditions]
        dots_and_interval(axis, rows, scale=1000.)
        axis.set(title=site, yticks=range(len(rows)), yticklabels=[LABELS[c] for c in small_conditions], xlabel="Error gain (10⁻³ standardized units)")
    fig.suptitle("Component effects at a narrower scale; full-donor conditions remain in the full-scale figure")
    fig.text(.5, .015, "All 11 match averages retained. Plot values are multiplied by 1,000 for readability; effects remain small.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, args.directory, "selection_component_effects_zoom")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, site in zip(axes, sites):
        rows = [row for row in audit["paired_control_comparisons"] if row["site"] == site and row["control"] != "norm_matched_random_full"]
        dots_and_interval(axis, rows, paired=True, scale=1000.)
        axis.set(title=site, yticks=range(len(rows)),
                 yticklabels=[LABELS[row["intervention"]]+" − "+LABELS[row["control"]] for row in rows],
                 xlabel="Gain above control (10⁻³ standardized units)")
    fig.suptitle("Paired component controls at a narrower scale; no observations omitted within these comparisons")
    fig.text(.5, .015, "Dots are all 11 paired match differences. Lines are descriptive whole-match bootstrap 95% intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, args.directory, "selection_component_controls_zoom")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, row in zip(axes, audit["attribution_vs_exact"]):
        x = np.asarray([value["attribution"] for value in row["match_values"]])
        y = np.asarray([value["exact_donor_gain"] for value in row["match_values"]])
        axis.scatter(x, y, color="#1764a5", s=40)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([low, high], [low, high], "--", color="gray", linewidth=.8)
        axis.axhline(0, color="black", linewidth=.5)
        axis.axvline(0, color="black", linewidth=.5)
        corr = row["pearson_across_seed_averaged_matches"]
        spearman = row["spearman_across_seed_averaged_matches"]
        sign_agreement = row["positive_sign_agreement_fraction"]
        axis.set(title=f"{row['site']}\nr={corr:.3f}; Spearman={spearman:.3f}; sign={sign_agreement:.0%}" if corr is not None else row["site"],
                 xlabel="Attribution estimate", ylabel="Exact full-donor error gain")
    fig.suptitle("Attribution versus exact replacement: one point per match, averaging two seeds")
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, args.directory, "selection_attribution_vs_exact")

    matrices = [np.asarray([next(row for row in audit["comparisons"] if row["site"] == site and row["condition"] == condition)["proxy_shift_standardized_mean_abs"] for condition in conditions]) for site in sites]
    maximum = max(value.max() for value in matrices)
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
    names = [name.replace(".location.", " ").replace("player_", "p") for name in audit["target_names"]]
    for axis, site, matrix in zip(axes, sites, matrices):
        im = axis.imshow(matrix, aspect="auto", vmin=0, vmax=maximum, cmap="magma")
        axis.set(title=site, yticks=range(len(conditions)), yticklabels=[LABELS[c] for c in conditions],
                 xticks=range(15), xticklabels=names)
        axis.tick_params(axis="x", labelrotation=70, labelsize=8)
        axis.get_xticklabels()[2].set_color("#1574c1")
    colorbar_axis = fig.add_axes((.88, .25, .012, .5))
    fig.colorbar(im, cax=colorbar_axis, label="Mean absolute proxy shift / discovery target SD")
    fig.suptitle("Target and collateral position readouts; blue label is the primary ball-height target")
    fig.text(.5, .02, "These are shifts in a frozen codec readout of model endpoint estimates, not independently measured physical states.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.19, right=.84, bottom=.19, top=.89, wspace=.1)
    save(fig, args.directory, "selection_collateral_proxies")


if __name__ == "__main__":
    main()
