#!/usr/bin/env python3
"""Plot the actual all-layer magnitude check, without implying physical geometry."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "results/pretrained_smoke.json").read_text())
if report["status"] != "passed":
    raise RuntimeError("The pretrained smoke must pass before plotting its result")
rows = report["capture"]["layers"]
if [r["layer"] for r in rows] != list(range(16)) or not all(r["finite"] for r in rows):
    raise RuntimeError("Expected a finite report for every one of the 16 layers")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot([r["layer"] for r in rows], [r["rms"] for r in rows], "o-", color="#245c88", label="Residual RMS")
ax.set(xlabel="Residual block output (zero-based)", ylabel="Activation RMS",
       title="MIRA Mini 4P: all-layer capture sanity check", xticks=list(range(16)))
ax.grid(axis="y", alpha=0.25)
fig.text(0.5, 0.025,
         "One public context, teacher-forced input, tau=0.5. No physical labels or layer selection.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for suffix in ("png", "pdf"):
    fig.savefig(root / f"figures/pretrained_layer_rms.{suffix}", dpi=160)
