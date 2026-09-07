#!/usr/bin/env python3
"""Audit finished observational evidence while keeping later proposal gates explicit."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    checks = []

    def require(condition, name):
        if not condition:
            raise ValueError(name)
        checks.append(name)

    registration = read("results/observational_registration_v2.json")
    protocol = read(registration["protocol_path"])
    digest = registration["sha256"]
    require(sha(ROOT / registration["protocol_path"]) == digest, "frozen_protocol_unchanged")
    for name, expected in registration["parent_artifact_sha256"].items():
        require(sha(ROOT / name) == expected, "original_artifact_unchanged:" + name)
    require(read("results/cohort_data_audit.json")["status"] == "failed", "original_data_failure_preserved")
    data = read("results/qualified_data_audit.json")
    require(data["status"] == "passed" and data["verified_clips"] == 424, "all_qualified_source_clips_verified")
    require(data["protocol_sha256"] == digest, "data_protocol_binding")
    require(sha(ROOT / "data/qualified_clip_manifest.json") == data["clip_manifest_sha256"], "qualified_clip_manifest_binding")
    require(sha(ROOT / "data/qualified_split_manifest.json") == data["split_manifest_sha256"], "qualified_split_manifest_binding")
    aggregate = read("results/capture_aggregate_v2.json")
    require(aggregate["status"] == "passed" and aggregate["rows"] == 13568
            and aggregate["expected_clips"] == 424, "complete_research_capture_and_aggregate")
    require(aggregate["registration_sha256"] == digest, "aggregate_protocol_binding")
    require(aggregate["data_audit_sha256"] == sha(ROOT / "results/qualified_data_audit.json"), "aggregate_data_binding")
    require(aggregate["match_counts"] == {"discovery": 31, "selection": 11, "confirmation": 11}, "frozen_match_counts")
    require(sha(Path(aggregate["analysis_npz_path"])) == aggregate["analysis_npz_sha256"], "aggregate_bytes_unchanged")
    for name in ("all_layers_complete", "match_splits_disjoint", "checkpoint_integrity_checked", "all_source_label_joins_exact"):
        require(aggregate[name] is True, name)
    require(not aggregate["duplicate_or_missing_rows"], "unique_complete_rows")
    for name, expected in aggregate["capture_code_sha256"].items():
        require(sha(ROOT / name) == expected, "capture_code_unchanged:" + name)
    require(sha(ROOT / "scripts/aggregate_captures.py") == aggregate["aggregate_script_sha256"], "aggregate_code_unchanged")
    for name, expected in aggregate["capture_report_sha256"].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        worker = json.loads(path.read_text())
        require(sha(path) == expected and worker["status"] == "passed" and worker["completed_clips"] == 212,
                "worker_complete_and_unchanged:" + path.name)
    pilot = read("results/capture_pilot_v2.json")
    require(pilot["status"] == "passed" and aggregate["pilot_report_sha256"] == sha(ROOT / "results/capture_pilot_v2.json"), "pilot_gate_binding")
    require(read("results/capture_pilot_v2_row_audit.json")["all_source_labels_and_timestamps_exact"], "independent_pilot_row_audit")
    base = "results/probes_v2/"
    frozen, selection, confirmation, status = (read(base + name) for name in
        ("frozen_selection.json", "selection_audit.json", "confirmation.json", "status.json"))
    frozen_hash = sha(ROOT / base / "frozen_selection.json")
    model_hash = sha(ROOT / base / "frozen_models.npz")
    require(selection["status"] == "passed" and selection["ridge_normal_equations_checked"], "independent_selection_audit")
    require(selection["confirmation_target_rows_materialized"] == 0 and not selection["models_refitted"], "selection_audit_no_confirmation_or_refit")
    for label, report in (("frozen", frozen), ("selection_audit", selection), ("confirmation", confirmation)):
        require(report["registration_sha256"] == digest and report["input_npz_sha256"] == aggregate["analysis_npz_sha256"], label + "_input_binding")
        require(report["model_archive_sha256"] == model_hash, label + "_coefficient_binding")
    aggregate_hash = sha(ROOT / "results/capture_aggregate_v2.json")
    require(frozen["capture_audit_sha256"] == selection["capture_audit_sha256"] == aggregate_hash, "selection_aggregate_audit_binding")
    require(confirmation["analysis_code_sha256"] == frozen["analysis_code_sha256"] == selection["analysis_code_sha256"], "confirmation_analysis_code_binding")
    require(confirmation["frozen_selection_winner"] == frozen["winner"] == selection["frozen_winner"], "selection_winner_preserved")
    require(selection["frozen_selection_sha256"] == confirmation["frozen_selection_sha256"] == frozen_hash, "immutable_selection_lock")
    require(confirmation["selection_audit_sha256"] == sha(ROOT / base / "selection_audit.json"), "confirmation_requires_selection_audit")
    require(confirmation["status"] == "passed_observational_analysis" and status["confirmation_sha256"] == sha(ROOT / base / "confirmation.json"), "confirmation_complete_and_bound")
    for name, expected in frozen["analysis_code_sha256"].items():
        require(sha(ROOT / name) == expected, "analysis_code_unchanged:" + name)
    require(selection["audit_script_sha256"] == sha(ROOT / "scripts/audit_probe_selection.py"), "selection_auditor_unchanged")
    expected_sites = protocol["capture"]["sites"] + ["codec_X", "RGB_X"]
    require([row["name"] for row in confirmation["sites"]] == expected_sites, "all_sites_and_baselines_reported")
    require(confirmation["target_names"] == protocol["targets"] and len(protocol["targets"]) == 30, "all_targets_reported")
    for row in confirmation["sites"]:
        for kind in ("main", "shuffled"):
            values = row[kind]
            require(values["n_matches"] == 11 and values["bootstrap_unit"] == "whole_match"
                    and values["bootstrap_replicates"] == 500 and math.isfinite(values["normalized_mse"])
                    and all(len(values[key]) == 30 for key in ("mae", "rmse", "r2")),
                    "complete_metrics:" + row["name"] + ":" + kind)
    require(not confirmation["causal_stage_ready"], "observational_claim_boundary")
    feasibility = read("results/controlled_pair_feasibility.json")
    require(feasibility["roles_read"] == ["discovery"] and feasibility["within_match_pairs_evaluated"] == 868,
            "controlled_pair_availability_audited")
    require(not feasibility["causal_identification_gate_passed"], "causal_gate_remains_unmet")
    tests = ET.parse(ROOT / "results/tests.xml").getroot()
    suites = list(tests.iter("testsuite"))
    require(suites and all(int(item.get("failures", 0)) == int(item.get("errors", 0)) == 0 for item in suites), "software_tests_pass")
    figures = [f"figures/observational_{name}.{extension}" for name in ("layers", "targets") for extension in ("png", "pdf")]
    require(all((ROOT / name).is_file() and (ROOT / name).stat().st_size > 1000 for name in figures), "exportable_figures_present")
    result = {"status": "passed_observational_completion_audit", "created_utc": datetime.now(timezone.utc).isoformat(),
              "checks_passed": len(checks), "checks": checks, "observational_stage": "complete",
              "full_proposal_complete": False, "proposal_step_1_causal_identification": "blocked_prerequisites",
              "proposal_step_2_geometry_sparse_features_steering": "not_run",
              "remaining_prerequisites": ["Controlled single-variable state-reset/rendered pairs", "Independent physical evaluator validated on generated videos", "Confirmed physical intervention effects"],
              "match_counts": aggregate["match_counts"], "clips": 424, "rows": aggregate["rows"],
              "test_cases": sum(int(item.get("tests", 0)) for item in suites),
              "confirmation_sha256": sha(ROOT / base / "confirmation.json"),
              "aggregate_audit_sha256": aggregate_hash, "registration_sha256": digest,
              "figure_sha256": {name: sha(ROOT / name) for name in figures},
              "audit_script_sha256": sha(Path(__file__))}
    (ROOT / "results/observational_completion_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
