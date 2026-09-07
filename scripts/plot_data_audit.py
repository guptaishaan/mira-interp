#!/usr/bin/env python3
"""Show the measured pilot-only event/source clock correction."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "results/pilot_data_audit.json").read_text())
if report["status"] != "passed" or report["prepared_clips"] != 8:
    raise RuntimeError("The reserved data pilot must pass before plotting")
calibrations = report["timeline_calibrations"]
if len(calibrations) != 1:
    raise RuntimeError("This plot is restricted to the one reserved pilot match")
calibration = next(iter(calibrations.values()))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for view in calibration["views"]:
    pairs = view["goal_pairs"]
    x = [p["score_total"] for p in pairs]
    axes[0].plot(x, [p["offset_seconds"] for p in pairs], ".-", label=f"View {view['view']}")
    axes[1].plot(x, [1000 * p["residual_seconds"] for p in pairs], ".-")
axes[0].set(xlabel="Goal number", ylabel="Goal-event time − source score-change time (s)",
            title="Measured source-clock origin")
axes[0].legend(fontsize=8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set(xlabel="Goal number", ylabel="Residual after median correction (ms)",
            title="Calibration residuals")
for ax in axes:
    ax.grid(alpha=0.2)
fig.suptitle("Rocket Science data pilot: correct the trim origin before filtering replays")
fig.text(0.5, 0.015, "One reserved pilot; calibration uses discrete score events, without probe scores or activations.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
for suffix in ("png", "pdf"):
    fig.savefig(root / f"figures/pilot_clock_calibration.{suffix}", dpi=160)
