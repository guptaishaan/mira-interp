#!/usr/bin/env python3
"""Plot numerical changes from the reserved-pilot audit; no physical score."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/review/numerics_pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("figures/numerics_pilot"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["status"] != "passed":
        raise ValueError("Numerical audit must pass before plotting")
    values = report["comparisons"]["original_full_bf16__to__publisher_precision"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].bar(["Codec latents", "Prediction", "Pooled means"], [100 * values[key]["relative_l2"] for key in ["codec", "prediction", "means"]], color=["#426b9b", "#49856e", "#9c6747"])
    axes[0].set_ylabel("Relative L2 change (%)")
    axes[0].set_title("All saved values")
    site_values = [100 * row["relative_l2"] for row in values["sites"]]
    axes[1].plot(np.arange(17), site_values, marker="o", color="#426b9b")
    axes[1].set_xticks([0, 1, 5, 9, 13, 16], ["Input", "0", "4", "8", "12", "15"])
    axes[1].set_xlabel("Residual block output")
    axes[1].set_ylabel("Pooled means: relative L2 change (%)")
    axes[1].set_title("All 17 readout sites")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.suptitle("Reserved pilot: FP32 checkpoint parameters versus full BF16 casting", fontsize=12)
    fig.text(.5, .02, "Identical video, actions and noise. Numerical reproducibility audit; no physical accuracy claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(args.output.with_suffix(suffix), dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
