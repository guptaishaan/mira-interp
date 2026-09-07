#!/usr/bin/env python3
"""Archive audited development features without source videos, actions or labels."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "607aef226c2cb241e66d00c9a3b7f323802c6bce869f96345c3a0b014a220755"
FEATURES = {"mean_X", "spatial_X", "codec_spatial_X", "codec_mean_X", "RGB_X"}
METADATA = {"target_names", "sites", "view_index", "latent_frame_index", "source_frame_index",
            "timestamps", "canonical_player_ids", "match_ids", "split", "clip_ids"}
SHAPES = {"mean_X": (32, 17, 2048), "spatial_X": (32, 17, 1536),
          "codec_spatial_X": (32, 4608), "codec_mean_X": (32, 32), "RGB_X": (32, 1536)}
NUMERICAL_SHAPES = {"codec": (4, 8, 9, 16, 32), "interpolant": (1, 8, 36, 16, 32),
                    "prediction": (1, 8, 36, 16, 32), "means": (32, 17, 2048)}
NUMERICAL_NAMES = {f"{variant}__{kind}" for variant in
                   ("original_full_bf16", "fp32_model_legacy_mix", "publisher_precision")
                   for kind in NUMERICAL_SHAPES} | {"sites"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/data2/ishaangp/mira-interp/releases/development-v3-2026-09-07.tar.gz"))
    parser.add_argument("--report", type=Path, default=ROOT / "results/development_archive.json")
    args = parser.parse_args()
    partial = args.output.with_name(args.output.name + ".partial")
    require(not args.output.exists() and not partial.exists(), "Archive or partial already exists; do not overwrite")
    require(not args.report.exists(), "Packaging report already exists; do not overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "stage": "prerequisites", "clips_written": 0,
              "archive_members_verified": 0, "source_labels_video_actions_included": False}

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=round(time.monotonic() - started, 3))
        temp = args.report.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.report)
        print(json.dumps({key: report[key] for key in ["stage", "elapsed_seconds", "clips_written", "archive_members_verified"]}), flush=True)

    progress("prerequisites")
    try:
        paths = {"capture_manifest.json": ROOT / "results/development_capture_manifest.json",
                 "capture_audit.json": ROOT / "results/development_capture_audit.json",
                 "probe_audit.json": ROOT / "results/development_probes_v3/probe_audit.json",
                 "numerics_pilot.json": ROOT / "results/review/numerics_pilot.json",
                 "development_v3.json": ROOT / "configs/development_v3.json",
                 "THIRD_PARTY.md": ROOT / "THIRD_PARTY.md"}
        frozen = {str(path): sha(path) for path in paths.values()}
        frozen[str(Path(__file__))] = sha(Path(__file__))
        manifest, capture_audit, probe_audit, numerics = (read(paths[key]) for key in
                ["capture_manifest.json", "capture_audit.json", "probe_audit.json", "numerics_pilot.json"])
        require(manifest["status"] == capture_audit["status"] == probe_audit["status"] == numerics["status"] == "passed", "Capture, probe and numerical audits must pass")
        require(capture_audit["pilot_only"] is False and capture_audit["all_source_label_joins_exact"] is True
                and capture_audit["all_layers_complete"] is True and capture_audit["checkpoint_integrity_checked"] is True,
                "Complete capture audit required")
        require(sha(paths["capture_manifest.json"]) == capture_audit["capture_manifest_sha256"] == probe_audit["capture_manifest_sha256"]
                and sha(paths["capture_audit.json"]) == probe_audit["capture_audit_sha256"], "Capture/probe audit lineage differs")
        require(manifest["registration_sha256"] == capture_audit["registration_sha256"] == probe_audit["registration_sha256"]
                == sha(paths["development_v3.json"]) == PROTOCOL, "Development registration differs")
        require(manifest["confirmation_used"] is probe_audit["confirmation_arrays_read"] is False
                and probe_audit["models_refitted"] is False and probe_audit["normal_equations_checked"] is True,
                "Unexpected development analysis scope")
        for key, relative in [("development_selection_sha256", "development_selection.json"),
                              ("development_report_sha256", "development_report.json"),
                              ("model_archive_sha256", "development_models.npz")]:
            path = ROOT / "results/development_probes_v3" / relative
            require(sha(path) == probe_audit[key], f"Audited probe input changed: {relative}")
            frozen[str(path)] = probe_audit[key]
        for path, expected in {**manifest["code_sha256"], **probe_audit["analysis_code_sha256"]}.items():
            require(sha(ROOT / path) == expected, f"Frozen source code changed: {path}")
            frozen[str(ROOT / path)] = expected
        records = sorted(manifest["records"], key=lambda row: (row["split"], row["match_id"], row["clip_id"]))
        checked = {row["clip_id"]: row for row in capture_audit["clips"]}
        require(len(records) == len(checked) == 336 and len({row["clip_id"] for row in records}) == 336
                and set(checked) == {row["clip_id"] for row in records}, "Exactly 336 audited clips required")
        require(Counter(row["split"] for row in records) == {"discovery": 248, "selection": 88}
                and set(Counter(row["match_id"] for row in records).values()) == {8}, "Development roles/clip counts differ")
        for row in records:
            audited = checked[row["clip_id"]]
            require(row["sha256"] == audited["capture_sha256"] and row["match_id"] == audited["match_id"]
                    and row["split"] == audited["split"] and audited["rows"] == 32, "Clip identity differs from passed audit")
        numerical_path = Path(numerics["output_path"])
        require(sha(numerical_path) == numerics["output_sha256"] and numerics["role"] == "pilot"
                and numerics["original_pooled_capture_replay_bitwise_equal"] is True, "Numerical pilot provenance changed")
        frozen[str(numerical_path)] = numerics["output_sha256"]

        members = []
        def add_bytes(archive, name, content, **metadata):
            info = tarfile.TarInfo("development/" + name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            archive.addfile(info, io.BytesIO(content))
            members.append({"path": info.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), **metadata})

        with partial.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for index, row in enumerate(records):
                    path = Path(row["path"])
                    require(sha(path) == row["sha256"], f"Source feature bytes changed: {row['clip_id']}")
                    with np.load(path, allow_pickle=False) as source:
                        require(set(source.files) == FEATURES | METADATA | {"y"}, "Unknown source schema; publication denied")
                        # Deliberately never access source['y'].
                        arrays = {key: source[key] for key in sorted(FEATURES | METADATA)}
                    require(sha(path) == row["sha256"], "Source changed while reading")
                    for key, shape in SHAPES.items():
                        require(arrays[key].shape == shape and arrays[key].dtype == np.float16
                                and np.isfinite(arrays[key]).all(), f"Invalid feature array: {key}")
                    require(arrays["clip_ids"].tolist() == [row["clip_id"]] * 32
                            and arrays["match_ids"].tolist() == [row["match_id"]] * 32
                            and arrays["split"].tolist() == [row["split"]] * 32, "Row metadata changed")
                    fingerprints = {key: fingerprint(value) for key, value in arrays.items()}
                    buffer = io.BytesIO()
                    np.savez(buffer, **arrays)  # ZIP_STORED; only the outer tar is compressed.
                    add_bytes(archive, f"captures/{row['split']}/{row['clip_id']}.npz", buffer.getvalue(),
                              kind="derived_capture", source_sha256=row["sha256"], clip_id=row["clip_id"],
                              match_id=row["match_id"], split=row["split"], arrays=fingerprints)
                    report["clips_written"] = index + 1
                    if index % 32 == 0 or index + 1 == len(records):
                        report["compressed_bytes_so_far"] = raw.tell()
                        report["write_seconds_per_clip"] = round((time.monotonic() - started) / (index + 1), 4)
                        progress("write_derived_captures")
                with np.load(numerical_path, allow_pickle=False) as source:
                    require(set(source.files) == NUMERICAL_NAMES, "Unknown numerical pilot schema; publication denied")
                    arrays = {key: source[key] for key in sorted(source.files)}
                for key, value in arrays.items():
                    if key != "sites":
                        require(value.shape == NUMERICAL_SHAPES[key.split("__")[1]] and value.dtype == np.float32
                                and np.isfinite(value).all(), "Invalid numerical derived array")
                buffer = io.BytesIO()
                np.savez(buffer, **arrays)
                add_bytes(archive, "numerics_pilot.npz", buffer.getvalue(), kind="derived_numerics",
                          source_sha256=numerics["output_sha256"], arrays={key: fingerprint(value) for key, value in arrays.items()})
                for name, path in paths.items():
                    add_bytes(archive, "provenance/" + name, path.read_bytes(), kind="provenance")
                readme = (ROOT / "docs/review/development_archive.md").read_bytes()
                add_bytes(archive, "README.md", readme, kind="documentation")
                public_manifest = {"scope": "336 development feature captures and all numerical pilot derived arrays",
                    "clips": 336, "rows": 10752, "match_counts": {"discovery": 31, "selection": 11},
                    "source_labels_video_actions_included": False, "registration_sha256": PROTOCOL,
                    "source_capture_manifest_sha256": sha(paths["capture_manifest.json"]),
                    "source_capture_audit_sha256": sha(paths["capture_audit.json"]),
                    "source_probe_audit_sha256": sha(paths["probe_audit.json"]),
                    "separate_model_archive": {"name": "development_models.npz", "sha256": probe_audit["model_archive_sha256"],
                        "bytes": (ROOT / "results/development_probes_v3/development_models.npz").stat().st_size,
                        "included_in_this_archive": False},
                    "compression": "NumPy ZIP_STORED arrays, outer gzip level1", "files": members.copy()}
                add_bytes(archive, "manifest.json", (json.dumps(public_manifest, indent=2) + "\n").encode(), kind="manifest")
        require(partial.stat().st_size < 2_000_000_000, "Archive exceeds conservative 2GB release-asset limit")
        progress("read_back_archive")
        expected = {item["path"]: item for item in members}
        seen = set()
        with tarfile.open(partial, "r|gz") as archive:
            for member in archive:
                require(member.isfile() and member.name in expected and member.name not in seen, "Unexpected/duplicate archive member")
                item = expected[member.name]
                require(member.size == item["bytes"], "Archived member size changed")
                content = archive.extractfile(member).read()
                require(hashlib.sha256(content).hexdigest() == item["sha256"], "Archived member bytes changed")
                if "arrays" in item:
                    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
                        require(set(saved.files) == set(item["arrays"]), "Archive array coverage differs")
                        require(not {"y", "targets", "actions", "frames", "physics"} & set(saved.files), "Source array leaked into archive")
                        for key in saved.files:
                            require(fingerprint(saved[key]) == item["arrays"][key], f"Archived array differs from audited source: {member.name}/{key}")
                elif item["kind"] == "manifest":
                    require(json.loads(content) == public_manifest, "Archived manifest differs")
                seen.add(member.name)
                report["archive_members_verified"] += 1
                if len(seen) % 32 == 0:
                    progress("read_back_archive")
        require(seen == set(expected), "Missing archive members")
        require(all(sha(Path(path)) == value for path, value in frozen.items()), "Audit/code/model provenance changed while packaging")
        require(all(sha(Path(row["path"])) == row["sha256"] for row in records), "Source captures changed during packaging")
        partial.rename(args.output)
        report.update(status="passed", archive_path=str(args.output), archive_sha256=sha(args.output),
                      archive_bytes=args.output.stat().st_size, all_archive_members_verified=True,
                      all_derived_arrays_equal_audited_source=True, all_source_capture_hashes_checked=True,
                      packaging_script_sha256=sha(Path(__file__)), **public_manifest)
        progress("complete")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), partial_path=str(partial))
        progress("failed")
        raise


if __name__ == "__main__":
    main()
