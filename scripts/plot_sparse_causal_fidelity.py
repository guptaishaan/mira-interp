#!/usr/bin/env python3
"""Plot every independently audited dictionary and shared mean-descriptor control."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "results/sparse_causal_fidelity_v1")
    args = parser.parse_args()
    result_path, audit_path = args.directory / "fidelity_results.json", args.directory / "fidelity_audit.json"
    result, audit, registration = read(result_path), read(audit_path), read(args.directory / "registration.json")
    if audit["status"] != "passed" or audit["fidelity_results_sha256"] != sha(result_path):
        raise ValueError("Matching independent full fidelity audit required")
    if result["registration_sha256"] != sha(args.directory / "registration.json") or result["conditions_checked"] != 1364:
        raise ValueError("Registered full conditional-fidelity study incomplete")
    dictionaries = registration["dictionaries"]
    variants = ["relu", "signed", "block", "block_temporal", "block_temporal_shuffled"]
    groups = [(active, variant) for active in (32, 64) for variant in variants]
    names = {"relu": "ReLU", "signed": "Signed", "block": "Block", "block_temporal": "Temporal view", "block_temporal_shuffled": "Shuffled view"}
    metrics = [("gain_difference_from_native_reference", "Signed donor-objective gain difference\nfrom native reference (standardized units)"),
               ("flow_distance_from_native_reference_l2", "Full-flow distance from native reference\nL2 (codec flow units)"),
               ("descriptor_reconstruction_rmse", "Descriptor reconstruction RMSE\n(raw residual units)")]
    colors = ["#2474a8", "#e48632", "#379c66"]
    comparisons = {(row["dictionary_id"], row["condition"]): row for row in result["comparisons"]}
    improvements = {(row["dictionary_id"], row["condition"]): row for row in result["paired_discovery_mean_comparisons"]}
    artifacts = {}
    for condition in ("recipient_reconstruction", "donor_reconstruction"):
        fig, axes = plt.subplots(1, 4, figsize=(17, 7.5), sharey=True, layout="constrained")
        for dictionary in dictionaries:
            identifier, seed = dictionary["id"], dictionary["seed"]
            y = groups.index((dictionary["active"], dictionary["variant"])) + (seed-1)*.2
            for axis, (metric, _) in zip(axes[:3], metrics):
                value = comparisons[identifier, condition]["metrics"][metric]
                lo, hi = value["match_bootstrap_ci95"]
                axis.plot([lo, hi], [y, y], color=colors[seed], alpha=.75, linewidth=1.4)
                axis.scatter(value["mean"], y, color=colors[seed], s=22, zorder=3)
            value = improvements[identifier, condition]["metrics"]["absolute_native_gain_error_improvement_over_mean_control"]
            lo, hi = value["match_bootstrap_ci95"]
            axes[3].plot([lo, hi], [y, y], color=colors[seed], alpha=.75, linewidth=1.4)
            axes[3].scatter(value["mean"], y, color=colors[seed], s=22, zorder=3)
        for axis, (metric, label) in zip(axes[:3], metrics):
            control = comparisons["discovery_mean", condition]["metrics"][metric]
            axis.axvspan(*control["match_bootstrap_ci95"], color="gray", alpha=.12)
            axis.axvline(control["mean"], color="gray", linestyle="--", linewidth=1.2)
            limits = [0., *control["match_bootstrap_ci95"]]
            for dictionary in dictionaries:
                limits.extend(comparisons[dictionary["id"], condition]["metrics"][metric]["match_bootstrap_ci95"])
            padding = (max(limits)-min(limits))*.04
            axis.set_xlim(min(limits)-padding, max(limits)+padding)
            axis.set_xlabel(label)
        axes[3].set_xlabel("Reduction of absolute native-gain error\nversus discovery mean (positive = closer)")
        for axis in axes:
            axis.axvline(0, color="black", linewidth=.65)
            axis.grid(axis="y", alpha=.15)
        axes[0].set_yticks(range(len(groups)), [f"{names[variant]} · K={active}" for active, variant in groups])
        axes[0].invert_yaxis()
        reference = "native recipient output" if condition.startswith("recipient") else "original full-donor replacement"
        fig.suptitle(f"Conditional sparse reconstruction fidelity: {condition.replace('_', ' ')}\nReference: {reference}; native descriptor nullspace retained; all 30 dictionaries shown", fontsize=12)
        fig.supxlabel("Blue / orange / green: training seeds 0 / 1 / 2. Dashed gray line and band: shared discovery-mean control and its interval.\n"
                      "11 development matches; paired noise seeds averaged within each match. Descriptive 95% whole-match bootstrap intervals; no sufficiency or physical-control claim.", fontsize=8)
        for extension in ("png", "pdf"):
            path = args.directory / f"{condition}_fidelity.{extension}"
            fig.savefig(path, dpi=180)
            artifacts[path.name] = sha(path)
        plt.close(fig)
    report = {"status": "passed_audited_plot_export", "fidelity_results_sha256": sha(result_path),
              "fidelity_audit_sha256": sha(audit_path), "script_sha256": sha(Path(__file__)),
              "artifacts": artifacts, "all_dictionary_seeds_retained": True, "physical_control_claim": False}
    (args.directory / "plot_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
