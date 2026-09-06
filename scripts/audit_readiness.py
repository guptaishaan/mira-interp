#!/usr/bin/env python3
"""Audit completed readiness artifacts without advancing any scientific stage."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    checks = []
    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})
    def read(name):
        return json.loads((ROOT / "results" / name).read_text())

    versions = read("environment.json")
    check("only authorized GPU IDs configured", versions["authorized_physical_gpus"] == [6, 7])
    download = read("model_download.json")
    check("pinned assets passed integrity check", download["status"] == "PASS" and
          len(download["files"]) == 6 and all(f["verified"] for f in download["files"]))
    for label in ("cpu", "gpu6"):
        smoke = read(f"instrumentation_smoke_{label}.json")
        c = smoke["checks"]
        check(f"{label} instrumentation passed", smoke["status"] == "PASS" and
              c["all_layers_captured"] == smoke["config"]["n_layers"] and
              c["capture_noop_max_abs_error"] == 0 and c["replacement_noop_max_abs_error"] == 0 and
              c["restoration_max_abs_error"] == 0 and c["masked_ablation_locality"] and
              c["gradients_retained_all_layers"] and c["all_hooks_removed"])
        check(f"{label} evidence excludes scientific claims", not smoke["scientific_claims"])
    suite = ET.parse(ROOT / "results/tests.xml").getroot()
    suites = list(suite.iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ("tests", "errors", "failures", "skipped")}
    check("unit tests complete without failures", counts["tests"] >= 19 and
          counts["errors"] == counts["failures"] == counts["skipped"] == 0)
    trained = read("pretrained_smoke.json")
    check("pretrained run complete", trained["status"] == "passed")
    if trained["status"] == "passed":
        check("strict codec and world model loading", trained["codec_strict_load"]["passed"] and
              trained["world_model_strict_load"]["passed"])
        check("trained observation and replay controls", trained["forward_controls"]["noop_hooks_bitwise_equal"] and
              trained["forward_controls"]["repeated_forward_bitwise_equal"])
        first = trained["block_zero_input"]
        check("block zero input persisted and finite", first["finite"] and
              digest(Path(first["activation_file"])) == first["sha256"])
        rows = trained["capture"]["layers"]
        check("all 16 trained layers finite", {r["layer"] for r in rows} == set(range(16)) and
              all(r["finite"] for r in rows) and trained["capture"]["all_layers_finite"])
        for row in rows:
            path = Path(row["activation_file"])
            check(f"layer {row['layer']} tensor hash", path.is_file() and digest(path) == row["sha256"])
        check("finite generated video", trained["sample"]["finite"])
        check("future placeholders cannot affect sampled output", trained["generation_controls"]["generated_latents_bitwise_equal"] and
              trained["generation_controls"]["generated_video_bitwise_equal"])
        for key in ("saved_generated_frames", "saved_generated_latents"):
            item = trained["sample"][key]
            check(key + " digest", digest(Path(item["path"])) == item["sha256"])
        check("physical evidence not claimed", trained["physical_validation"] is False)
        for label in ("figure", "layer_statistics"):
            check(f"{label} artifact exists", Path(trained["artifacts"][label]).is_file())
    data = read("data_access.json")
    passed = all(row["passed"] for row in checks)
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS" if passed else "FAIL", "scope": "engineering readiness only",
              "tests": counts, "checks": checks, "data_status": data["status"],
              "scientific_study_complete": False,
              "remaining_gate": "Prepare and audit real synchronized physics-labeled match partitions",
              "scientific_stages_evaluated_by_this_audit": []}
    output = ROOT / "results/readiness_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "tests": counts,
                      "data_status": data["status"], "scientific_study_complete": False}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
