#!/usr/bin/env python3
"""Package audited fresh-confirmation features and provenance, excluding raw inputs."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np
from package_development import ROOT, FEATURES, METADATA, SHAPES, fingerprint, read, require, sha

LOCK = "2750c3a9df2478898aaef6f5bf470d5241eb4315257a2e75f59daddd6a7476b9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/data2/ishaangp/mira-interp/releases/fresh-confirmation-v3-2026-09-07.tar.gz"))
    parser.add_argument("--report", type=Path, default=ROOT / "results/fresh_confirmation_archive.json")
    args = parser.parse_args()
    partial = args.output.with_name(args.output.name + ".partial")
    require(not args.output.exists() and not partial.exists() and not args.report.exists(), "Existing publication artifact must not be overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "publication_state": "local_only", "clips_written": 0,
              "archive_members_verified": 0, "source_labels_video_actions_included": False}
    def progress(stage):
        report.update(stage=stage, elapsed_seconds=round(time.monotonic() - started, 3))
        temp = args.report.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.report)
        print(json.dumps({k: report[k] for k in ["stage", "elapsed_seconds", "clips_written", "archive_members_verified"]}), flush=True)
    progress("prerequisites")
    try:
        paths = {"fresh_evaluation_v3.json": ROOT / "configs/fresh_evaluation_v3.json",
                 "development_v3.json": ROOT / "configs/development_v3.json",
                 "confirmation.json": ROOT / "results/fresh_confirmation_v3/confirmation.json",
                 "confirmation_audit.json": ROOT / "results/fresh_confirmation_v3/confirmation_audit.json",
                 "capture_audit.json": ROOT / "results/fresh_confirmation_v3/capture_audit.json",
                 "qualified_clip_manifest.json": ROOT / "data/qualified_fresh_confirmation_clip_manifest.json",
                 "candidate_split_manifest.json": ROOT / "data/fresh_confirmation_split_manifest.json",
                 "original_clip_manifest.json": ROOT / "data/fresh_confirmation_clip_manifest.json",
                 "qualification.json": ROOT / "results/fresh_confirmation_qualification.json",
                 "original_source_audit.json": ROOT / "results/fresh_confirmation_data_audit.json",
                 "source_download.json": ROOT / "results/fresh_confirmation_download.json",
                 "float_preparation.json": ROOT / "results/fresh_confirmation_float.json",
                 "development_probe_audit.json": ROOT / "results/development_probes_v3/probe_audit.json",
                 "development_publication.json": ROOT / "results/development_publication.json",
                 "THIRD_PARTY.md": ROOT / "THIRD_PARTY.md"}
        frozen = {path: sha(path) for path in paths.values()}
        frozen[Path(__file__)] = sha(Path(__file__))
        frozen[ROOT / "scripts/package_development.py"] = sha(ROOT / "scripts/package_development.py")
        lock, result, audit, capture_audit, qualified, qualification, source_audit, prep, publication = (read(paths[key]) for key in
                ["fresh_evaluation_v3.json", "confirmation.json", "confirmation_audit.json", "capture_audit.json",
                 "qualified_clip_manifest.json", "qualification.json", "original_source_audit.json", "float_preparation.json", "development_publication.json"])
        require(audit["status"] == capture_audit["status"] == qualified["status"] == qualification["status"] == prep["status"] == publication["status"] == "passed"
                and result["status"] == "passed_fresh_observational_confirmation", "Independent fresh audits and qualification must pass")
        require(sha(paths["fresh_evaluation_v3.json"]) == LOCK == result["registered_choices_sha256"] == audit["registered_choices_sha256"]
                == capture_audit["registered_choice_sha256"], "Frozen choice registration changed")
        require(audit["confirmation_sha256"] == sha(paths["confirmation.json"]) and audit["capture_audit_sha256"] == result["capture_audit_sha256"]
                == sha(paths["capture_audit.json"]) and audit["prior_probe_audit_sha256"] == sha(paths["development_probe_audit.json"]), "Fresh audit chain differs")
        require(audit["all_source_label_joins_exact"] is True and audit["all_published_metrics_and_intervals_recomputed"] is True
                and audit["models_refitted"] is audit["new_hypotheses_or_choices"] is False, "Incomplete fresh audit")
        require(qualified["source_audit_sha256"] == qualification["source_audit_sha256"] == sha(paths["original_source_audit.json"])
                and qualified["registered_manifest_sha256"] == sha(paths["candidate_split_manifest.json"]) == lock["candidate_manifest_sha256"]
                and qualification["qualified_clip_manifest_sha256"] == prep["qualified_manifest_sha256"] == audit["source_manifest_sha256"]
                == sha(paths["qualified_clip_manifest.json"]), "Source qualification lineage differs")
        require(source_audit["status"] == "failed", "Original failed source audit must remain preserved")
        require(qualified["parent_clip_manifest_sha256"] == source_audit["clip_manifest_sha256"] == sha(paths["original_clip_manifest.json"])
                and source_audit["download_report_sha256"] == sha(paths["source_download.json"]), "Original source/download hash chain differs")
        require(audit["n_matches"] == qualified["n_matches"] == result["n_matches"] == 23 and audit["n_clips"] == 184
                and audit["scored_rows"] == result["n_rows"] == 4416, "Fresh cohort coverage differs")
        for group in ["capture_code_sha256", "evaluation_code_sha256"]:
            for relative, expected in lock[group].items():
                require(sha(ROOT / relative) == expected, "Frozen source code changed")
                frozen[ROOT / relative] = expected
        model_path = ROOT / "results/development_probes_v3/development_models.npz"
        require(sha(model_path) == lock["model_archive_sha256"] == audit["model_archive_sha256"], "Frozen model archive changed")
        frozen[model_path] = lock["model_archive_sha256"]
        model_assets = [r for r in publication["assets"] if r["name"] == model_path.name]
        require(len(model_assets) == 1 and model_assets[0]["digest"] == "sha256:" + lock["model_archive_sha256"], "Published model artifact differs")
        records = []
        for path in sorted((ROOT / "results").glob("fresh_capture_worker*.json")):
            worker = read(path)
            require(worker["status"] == "passed" and not worker["errors"] and worker["registration_sha256"] == LOCK
                    and worker["preparation_sha256"] == sha(paths["float_preparation.json"]), "Capture worker incomplete")
            records.extend(worker["records"])
            paths[path.name] = path
            frozen[path] = sha(path)
        expected = {r["clip_id"]: r for r in audit["clips"]}
        originals = {r["clip_id"]: r for r in qualified["records"]}
        require(len(records) == len(expected) == len(originals) == 184 and {r["clip_id"] for r in records} == set(expected) == set(originals)
                and Counter(r["split"] for r in records) == {"confirmation": 184}
                and len({r["match_id"] for r in records}) == 23 and set(Counter(r["match_id"] for r in records).values()) == {8}, "Missing/duplicate fresh captures")
        records.sort(key=lambda r: (r["match_id"], r["clip_id"]))
        members = []
        def add(archive, name, content, **metadata):
            info = tarfile.TarInfo("fresh_confirmation/" + name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            archive.addfile(info, io.BytesIO(content))
            members.append({"path": info.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), **metadata})
        with partial.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for index, rec in enumerate(records):
                    path, cid = Path(rec["path"]), rec["clip_id"]
                    require(sha(path) == rec["sha256"] == expected[cid]["sha256"] and expected[cid]["source_sha256"] == originals[cid]["sha256"]
                            and rec["match_id"] == originals[cid]["match_id"] and rec["registration_sha256"] == LOCK
                            and rec["code_sha256"] == lock["capture_code_sha256"], "Audited clip identity or bytes changed")
                    with np.load(path, allow_pickle=False) as source:
                        require(set(source.files) == FEATURES | METADATA | {"y"}, "Unknown publication schema")
                        # Never access source['y']; only these derived arrays are published.
                        arrays = {key: source[key] for key in sorted(FEATURES | METADATA)}
                    for key, shape in SHAPES.items():
                        require(arrays[key].shape == shape and arrays[key].dtype == np.float16 and np.isfinite(arrays[key]).all(), "Invalid derived feature")
                    require(arrays["clip_ids"].tolist() == [cid] * 32 and arrays["match_ids"].tolist() == [rec["match_id"]] * 32
                            and arrays["split"].tolist() == ["confirmation"] * 32 and sha(path) == rec["sha256"], "Source changed while reading")
                    buffer = io.BytesIO()
                    np.savez(buffer, **arrays)
                    add(archive, "captures/" + cid + ".npz", buffer.getvalue(), kind="derived_capture", clip_id=cid,
                        source_sha256=rec["sha256"], arrays={key: fingerprint(value) for key, value in arrays.items()})
                    report["clips_written"] = index + 1
                    if index % 32 == 0 or index + 1 == len(records):
                        report["compressed_bytes_so_far"] = raw.tell()
                        progress("write_derived_captures")
                for name, path in paths.items():
                    add(archive, "provenance/" + name, path.read_bytes(), kind="provenance")
                doc_path = ROOT / "docs/review/fresh_confirmation_archive.md"
                frozen[doc_path] = sha(doc_path)
                add(archive, "README.md", doc_path.read_bytes(), kind="documentation")
                manifest = {"scope": "all184 audited fresh observation feature captures and audit/protocol provenance",
                    "clips": 184, "captured_rows": 5888, "scored_rows": 4416, "matches": 23,
                    "source_labels_video_actions_included": False, "derived_downsampled_RGB_included": True,
                    "registered_choices_sha256": LOCK, "confirmation_sha256": sha(paths["confirmation.json"]),
                    "independent_confirmation_audit_sha256": sha(paths["confirmation_audit.json"]),
                    "original_source_audit_status": source_audit["status"], "qualified_source_status": qualification["status"],
                    "source_excluded_matches": qualified["excluded_matches"],
                    "separate_model_archive": {**model_assets[0], "included_in_this_archive": False},
                    "compression": "NumPy ZIP_STORED; outer gzip level1", "files": members.copy()}
                add(archive, "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode(), kind="manifest")
        require(partial.stat().st_size < 2_000_000_000, "Archive exceeds conservative2GB asset limit")
        progress("read_back_archive")
        expected_members = {r["path"]: r for r in members}
        seen = set()
        with tarfile.open(partial, "r|gz") as archive:
            for member in archive:
                require(member.isfile() and member.name in expected_members and member.name not in seen, "Unexpected/duplicate archive member")
                entry = expected_members[member.name]
                content = archive.extractfile(member).read()
                require(len(content) == entry["bytes"] and hashlib.sha256(content).hexdigest() == entry["sha256"], "Archive member bytes changed")
                if "arrays" in entry:
                    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
                        require(set(saved.files) == FEATURES | METADATA and not {"y", "actions", "frames", "targets", "physics"} & set(saved.files), "Excluded source arrays appeared")
                        for key in saved.files:
                            require(fingerprint(saved[key]) == entry["arrays"][key], "Archived array differs from audited capture")
                elif entry["kind"] == "manifest":
                    require(json.loads(content) == manifest, "Archived manifest differs")
                seen.add(member.name)
                report["archive_members_verified"] += 1
                if len(seen) % 32 == 0:
                    progress("read_back_archive")
        require(seen == set(expected_members) and all(sha(p) == h for p, h in frozen.items())
                and all(sha(Path(r["path"])) == r["sha256"] for r in records), "Archive incomplete or source/provenance changed")
        partial.rename(args.output)
        report.update(status="passed", archive_path=str(args.output), archive_bytes=args.output.stat().st_size,
                      archive_sha256=sha(args.output), all_archive_members_verified=True,
                      all_derived_arrays_equal_audited_source=True, all_source_capture_hashes_checked=True,
                      packaging_script_sha256=sha(Path(__file__)), packaging_helper_sha256=sha(ROOT / "scripts/package_development.py"), **manifest)
        progress("complete")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), partial_path=str(partial))
        progress("failed")
        raise


if __name__ == "__main__":
    main()
