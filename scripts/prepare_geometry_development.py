#!/usr/bin/env python3
"""Export one explicitly chosen, causally audited spatial development descriptor.

This script does not fit geometry/dictionaries or authorize their analysis gate.
Source ball labels stay local; the descriptor summarizes a whole camera view.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np
from package_development import ROOT, fingerprint, read, require, sha

PROTOCOL = "607aef226c2cb241e66d00c9a3b7f323802c6bce869f96345c3a0b014a220755"
CAUSAL_AUDIT = "69f36fec2aaf396bf0c0276e014c8c9f857757ba417952924e302dee654b54a9"
SITES = ["block_0_input"] + [f"block_{i}_output" for i in range(16)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=SITES)
    parser.add_argument("--output", type=Path, default=Path("/data2/ishaangp/mira-interp/features/geometry_development_v1.npz"))
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results/geometry_development_v1")
    args = parser.parse_args()
    require(not args.output.exists() and not args.report_dir.exists(), "Prior export is immutable; use new paths")
    started = time.monotonic()
    args.report_dir.mkdir(parents=True)
    report_path = args.report_dir / "input_audit.json"
    selected_path = args.report_dir / "selected_probe.json"
    audit = {"status": "running", "scope": "development_only", "clips_checked": 0,
             "analysis_executed": False, "confirmation_arrays_read": False,
             "contains_source_labels": True, "physical_causality_claim": False}
    def progress(stage):
        audit.update(stage=stage, elapsed_seconds=round(time.monotonic() - started, 3))
        report_path.write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps({k: audit[k] for k in ["stage", "elapsed_seconds", "clips_checked"]}), flush=True)
    progress("prerequisites")
    try:
        directory = ROOT / "results/causal_development_v1"
        causal_path, reg_path, causal_lock_path, causal_results_path = [directory / name for name in
                    ["selection_audit.json", "registration.json", "discovery_selection.json", "selection_results.json"]]
        causal, registration, causal_lock = read(causal_path), read(reg_path), read(causal_lock_path)
        require(sha(causal_path) == CAUSAL_AUDIT and causal["status"] == "passed"
                and causal["all_saved_output_metrics_recomputed"] is True and causal["conditions_checked"] == 660,
                "The independently reviewed exact causal audit must pass")
        require(causal["registration_sha256"] == sha(reg_path) and causal["discovery_selection_sha256"] == sha(causal_lock_path)
                and causal["selection_results_sha256"] == sha(causal_results_path), "Exact causal provenance changed")
        require(causal["confirmation_arrays_read"] is causal["physical_control_established"] is False, "Unexpected causal scope")
        site_index = SITES.index(args.site)
        require(causal_lock["chosen_sites"][0] == args.site and causal_lock["chosen_site_indices"][0] == site_index,
                "This export preserves the original top attribution site; it cannot reselect on exact/fresh results")
        require(any(row["site"] == args.site and row["condition"] == "component_transfer" for row in causal["comparisons"]), "Chosen site lacks exact audited evidence")
        frozen = {path: sha(path) for path in [Path(__file__), ROOT / "scripts/package_development.py",
                    causal_path, reg_path, causal_lock_path, causal_results_path]}
        for section in ["input_bindings", "probe_bindings"]:
            for entry in registration[section].values():
                path = Path(entry["path"])
                require(sha(path) == entry["sha256"], "Registered source/probe input changed")
                frozen[path] = entry["sha256"]
        for relative, expected in registration["code_sha256"].items():
            require(sha(ROOT / relative) == expected, "Frozen causal execution source changed")
            frozen[ROOT / relative] = expected
        inputs, probes = registration["input_bindings"], registration["probe_bindings"]
        manifest, capture_audit, preparation, split, selection, probe_audit = [read(path) for path in
            [inputs["capture_manifest"]["path"], inputs["capture_audit"]["path"], inputs["preparation"]["path"],
             inputs["split"]["path"], probes["selection"]["path"], probes["audit"]["path"]]]
        require(manifest["status"] == capture_audit["status"] == preparation["status"] == probe_audit["status"] == "passed", "Capture/probe prerequisites incomplete")
        require(manifest["registration_sha256"] == capture_audit["registration_sha256"] == probe_audit["registration_sha256"]
                == inputs["development_protocol"]["sha256"] == PROTOCOL, "Development protocol changed")
        require(capture_audit["capture_manifest_sha256"] == probe_audit["capture_manifest_sha256"] == inputs["capture_manifest"]["sha256"]
                and probe_audit["capture_audit_sha256"] == inputs["capture_audit"]["sha256"]
                and probe_audit["development_selection_sha256"] == probes["selection"]["sha256"]
                and probe_audit["model_archive_sha256"] == probes["archive"]["sha256"], "Capture/probe evidence bindings differ")
        probe_name = f"spatial/{args.site}/absolute30/position"
        entries = [row for row in selection["models"] if row["name"] == probe_name]
        require(len(entries) == 1, "Selected spatial position probe is missing/duplicated")
        entry = entries[0]
        require(entry["feature_spec"] == {"array": "spatial_X", "site_index": site_index, "summary": "spatial", "temporal": "current"},
                "Probe descriptor identity differs")
        expected_matches = {row["match_id"]: row["split"] for row in split["matches"] if row["split"] in {"discovery", "selection"}}
        records = sorted(manifest["records"], key=lambda row: (row["match_id"], row["clip_id"]))
        require(len(records) == 336 and len({row["clip_id"] for row in records}) == 336
                and Counter(expected_matches.values()) == {"discovery": 31, "selection": 11}, "Wrong development cohort")
        # Reject every role before opening any capture/label NPZ.
        require(all(row["split"] in {"discovery", "selection"} and expected_matches.get(row["match_id"]) == row["split"] for row in records)
                and {row["match_id"] for row in records} == set(expected_matches)
                and set(Counter(row["match_id"] for row in records).values()) == {8}, "Confirmation or incomplete development cohort refused")
        source_manifest_path = ROOT / "data/qualified_clip_manifest.json"
        require(sha(source_manifest_path) == capture_audit["source_manifest_sha256"], "Original source manifest changed")
        frozen[source_manifest_path] = sha(source_manifest_path)
        raw_records = {row["clip_id"]: row for row in read(source_manifest_path)["records"] if row["role"] in {"discovery", "selection"}}
        prepared = {row["clip_id"]: row for row in preparation["records"]}
        audited = {row["clip_id"]: row for row in capture_audit["clips"]}
        require(set(raw_records) == set(prepared) == set(audited) == {row["clip_id"] for row in records}, "Source clip identities differ")
        values = defaultdict(list)
        clip_checks = []
        current = np.asarray([view * 8 + latent for view in range(4) for latent in range(2, 8)])
        source_hashes = {}
        for index, row in enumerate(records):
            cid = row["clip_id"]
            source_path, capture_path = Path(raw_records[cid]["artifact_path"]), Path(row["path"])
            require(row["sha256"] == audited[cid]["capture_sha256"] and sha(capture_path) == row["sha256"]
                    and sha(source_path) == raw_records[cid]["sha256"] == audited[cid]["original_sha256"], "Audited source bytes differ")
            require(row["input_sha256"] == prepared[cid]["sha256"] == audited[cid]["prepared_sha256"], "Preparation identity changed")
            source_hashes[source_path], source_hashes[capture_path] = raw_records[cid]["sha256"], row["sha256"]
            with np.load(capture_path, allow_pickle=False) as captured, np.load(source_path, allow_pickle=False) as raw:
                require(captured["split"].tolist() == [row["split"]] * 32 and captured["match_ids"].tolist() == [row["match_id"]] * 32
                        and captured["clip_ids"].tolist() == [cid] * 32, "Role/identity differs before label access")
                require(captured["sites"].tolist() == SITES and captured["view_index"].tolist() == np.repeat(np.arange(4), 8).tolist()
                        and captured["latent_frame_index"].tolist() == np.tile(np.arange(8), 4).tolist(), "Site/view/time row convention changed")
                target = raw["targets"][:, 1::2].reshape(32, 30)
                require(np.array_equal(captured["y"], target) and np.array_equal(captured["timestamps"], raw["timestamps"][:, 1::2].reshape(32))
                        and np.array_equal(captured["source_frame_index"], np.tile(raw["source_frame_indices"][1::2], 4))
                        and np.array_equal(captured["canonical_player_ids"], np.tile(raw["player_ids"], (32, 1))), "Own-view source labels/times/identity differ")
                target_names = captured["target_names"].tolist()
                require(target_names[:6] == [f"ball.{kind}.{axis}" for kind in ["location", "velocity"] for axis in "xyz"], "Ball target columns changed")
                x = captured["spatial_X"][current, site_index]
                require(x.shape == (24, 1536) and x.dtype == np.float16 and np.isfinite(x).all()
                        and target.dtype == np.float64 and np.isfinite(target).all(), "Invalid feature/target values")
                values["X"].append(x)
                values["position"].append(target[current, :3])
                values["velocity"].append(target[current, 3:6])
                for key in ["match_ids", "split", "clip_ids", "view_index", "source_frame_index", "timestamps", "canonical_player_ids"]:
                    values[key].append(captured[key][current])
                values["frame_index"].append(captured["latent_frame_index"][current])
                values["entity_ids"].append(np.repeat("ball", 24))
                for view in range(4):
                    times = captured["timestamps"][current[view * 6:(view + 1) * 6]]
                    require(np.allclose(np.diff(times), .1, rtol=0, atol=1e-6), "Within-view latent timestamps are not0.1seconds apart")
            clip_checks.append({"clip_id": cid, "match_id": row["match_id"], "split": row["split"], "rows": 24,
                                "capture_sha256": row["sha256"], "source_sha256": raw_records[cid]["sha256"], "own_view_label_join_exact": True})
            audit["clips_checked"] = index + 1
            if index % 48 == 0:
                progress("verify_and_extract")
        arrays = {key: np.concatenate(items) for key, items in values.items()}
        require(arrays["X"].shape == (8064, 1536) and arrays["position"].shape == arrays["velocity"].shape == (8064, 3)
                and Counter(arrays["split"]) == {"discovery": 5952, "selection": 2112}
                and set(Counter(arrays["match_ids"]).values()) == {192}, "Final feature cohort/shape differs")
        identities = list(zip(arrays["match_ids"], arrays["clip_ids"], arrays["view_index"], arrays["frame_index"]))
        require(len(set(identities)) == len(identities), "Duplicated descriptor view/time identity")
        descriptor = f"spatial/{args.site}"
        discovery = arrays["split"] == "discovery"
        horizontal_speed = np.hypot(arrays["velocity"][:, 0], arrays["velocity"][:, 1])
        heading = (np.arctan2(arrays["velocity"][:, 1], arrays["velocity"][:, 0]) + np.pi) % (2 * np.pi) - np.pi
        supports = {}
        for variable, physical, interval, eligible in [
                ("height", arrays["position"][:, 2], [400., 800.], discovery),
                ("vz", arrays["velocity"][:, 2], [-250., 250.], discovery),
                ("horizontal_speed", horizontal_speed, [800., 1200.], discovery),
                ("heading", heading, [-np.pi / 8, np.pi / 8], discovery & (horizontal_speed >= 200.))]:
            low, high = interval
            inside = eligible & (physical >= low) & (physical <= high)
            outside = eligible & ~inside
            supported = physical[eligible]
            supports[variable] = {"fixed_candidate_interval": interval, "bounds_inclusive": True,
                "eligible_discovery_rows": int(eligible.sum()), "eligible_discovery_matches": len(set(arrays["match_ids"][eligible])),
                "eligible_discovery_min": float(supported.min()) if len(supported) else None,
                "eligible_discovery_max": float(supported.max()) if len(supported) else None,
                "discovery_rows_inside_interval": int(inside.sum()), "discovery_matches_inside_interval": len(set(arrays["match_ids"][inside])),
                "discovery_rows_outside_interval": int(outside.sum()), "discovery_matches_outside_interval": len(set(arrays["match_ids"][outside])),
                "interval_strictly_inside_discovery_support": bool(len(supported) and supported.min() < low < high < supported.max()),
                "per_discovery_match_rows_inside": {str(match): int((inside & (arrays["match_ids"] == match)).sum()) for match in sorted(set(arrays["match_ids"][discovery]))},
                "minimum_horizontal_speed": 200. if variable == "heading" else None}
        selected = {"status": "selected_after_exact_causal_audit", "selected_utc": datetime.now(timezone.utc).isoformat(),
                    "selection_rule": "Explicitly requested original first attribution-ranked site; no reselection on exact or fresh outcomes",
                    "descriptor_identity": descriptor, "feature_dim": 1536, "temporal_readout": "current",
                    "site": args.site, "site_index": site_index, "probe_name": probe_name, "model_sha256": entry["model_sha256"],
                    "probe_archive_sha256": probes["archive"]["sha256"], "selected_probe_record": entry,
                    "development_selection_sha256": probes["selection"]["sha256"], "probe_audit": probes["audit"],
                    "causal_audit": {"path": str(causal_path), "sha256": CAUSAL_AUDIT}, "causal_scope": "internal_model_output",
                    "descriptor_entity_localized": False, "ball_visibility_verified": False,
                    "annotation_entity": "global ball", "physical_causality_claim": False}
        selected_path.write_text(json.dumps(selected, indent=2) + "\n")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        expected_fingerprints = {key: fingerprint(value) for key, value in arrays.items()}
        with np.load(args.output, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays) and all(fingerprint(saved[key]) == expected_fingerprints[key] for key in saved.files), "Saved feature input differs from source slices")
        require(all(sha(path) == expected for path, expected in {**frozen, **source_hashes}.items()), "Frozen evidence or source changed during export")
        audit.update(status="passed_feature_input_export", input_npz_path=str(args.output), input_npz_sha256=sha(args.output),
                     input_bytes=args.output.stat().st_size, selected_probe_path=str(selected_path), selected_probe_sha256=sha(selected_path),
                     descriptor_identity=descriptor, feature_dim=1536, temporal_readout="current", site=args.site, site_index=site_index,
                     rows=8064, match_counts={"discovery": 31, "selection": 11}, row_counts={"discovery": 5952, "selection": 2112},
                     capture_manifest=inputs["capture_manifest"], capture_audit=inputs["capture_audit"], probe_audit=probes["audit"],
                     causal_audit={"path": str(causal_path), "sha256": CAUSAL_AUDIT}, registration_sha256=PROTOCOL,
                     exact_causal_registration_sha256=sha(reg_path), discovery_site_selection_sha256=sha(causal_lock_path),
                     split_manifest_sha256=inputs["split"]["sha256"], all_source_label_joins_exact=True, all_arrays_readback_exact=True,
                     view_identity_verified=True, annotation_ball_identity_verified=True, entity_mapping_verified=False,
                     descriptor_entity_localized=False, ball_visibility_verified=False,
                     temporal_correspondence_scope="same_match_clip_view", frame_index_semantics="consecutive codec latent index2..7; source index separate",
                     expected_dt_seconds=.1, adjacent_view_edges={"discovery": 4960, "selection": 1760},
                     entity_ids_semantics="ball labels only; descriptor is the entire camera view, not localized ball tokens",
                     discovery_physical_support=supports, physical_support_scope="discovery metadata only; fixed intervals supplied before export; no fitting",
                     causal_scope="internal_model_output", analysis_gate_authorized=False,
                     source_array_fingerprints=expected_fingerprints, clips=clip_checks, export_script_sha256=sha(Path(__file__)))
        progress("complete")
    except Exception as error:
        audit.update(status="failed", error_type=type(error).__name__, error=str(error))
        progress("failed")
        raise


if __name__ == "__main__":
    main()
