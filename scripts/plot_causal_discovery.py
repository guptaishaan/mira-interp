#!/usr/bin/env python3
"""Plot all17 audited discovery attribution means and per-match dispersion."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/causal_development_v1")
    args = parser.parse_args()
    path = args.directory / "discovery_selection.json"
    lock, audit = json.loads(path.read_text()), json.loads((args.directory / "discovery_audit.json").read_text())
    if audit["status"] != "passed" or audit["discovery_selection_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Exact passed discovery audit is required")
    groups = {}
    for entry in lock["pairs"]:
        content = Path(entry["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("Discovery report changed")
        result = json.loads(content)
        groups.setdefault(result["match_id"], []).append([row["attribution"] for row in result["scores"]])
    if len(groups) != 31 or any(len(values) != 2 for values in groups.values()):
        raise ValueError("Exactly31matches with2seeds required")
    matches = np.asarray([np.asarray(values).mean(0) for _, values in sorted(groups.items())])
    sites = ["block_0_input"] + [f"block_{i}_output" for i in range(16)]
    means = np.asarray([lock["all_site_mean_attributions"][site] for site in sites])
    colors = ["#206aab" if index in lock["chosen_site_indices"] else "#9aa3ac" for index in range(17)]
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.barh(range(17), means, height=.55, color=colors, alpha=.8)
    jitter = np.linspace(-.22, .22, len(matches))
    for index in range(17):
        axis.scatter(matches[:, index], index+jitter, s=10, color="#202020", alpha=.35, zorder=3)
    axis.axvline(0, color="black", linewidth=.7)
    axis.set(yticks=range(17), yticklabels=sites, xlabel="Predicted standardized donor-target error gain (gradient × donor difference)")
    axis.invert_yaxis()
    axis.set_title("All 17 discovery sites; blue marks the frozen top 3\nBar: mean across 31 matches; dots: each match averaged over 2 seeds", fontsize=12)
    fig.text(.5, .025, "Internal output-proxy screening only. Exact causal effects are a separate experiment.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(args.directory / f"discovery_attribution.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
