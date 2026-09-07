#!/usr/bin/env python3
"""Publish audited derived features and pilot residuals, excluding source labels/video."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-audit", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, default=ROOT / "results/observational_completion_audit.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.aggregate_audit.read_text())
    confirmation = json.loads(args.confirmation.read_text())
    pilot = json.loads(args.pilot_report.read_text())
    completion = json.loads(args.completion_audit.read_text())
    if audit["status"] != "passed" or pilot["status"] != "passed":
        raise ValueError("Capture and aggregation must pass before packaging")
    if confirmation["status"] != "passed_observational_analysis":
        raise ValueError("The observational analysis has not completed")
    if (completion["status"] != "passed_observational_completion_audit"
            or completion["confirmation_sha256"] != sha(args.confirmation)
            or completion["aggregate_audit_sha256"] != sha(args.aggregate_audit)
            or confirmation["registration_sha256"] != audit["registration_sha256"]
            or completion["registration_sha256"] != audit["registration_sha256"]):
        raise ValueError("Completed study provenance does not match these publication inputs")
    source = Path(audit["analysis_npz_path"])
    if sha(source) != audit["analysis_npz_sha256"] or confirmation["input_npz_sha256"] != sha(source):
        raise ValueError("Completed analysis is not bound to the aggregate")
    if audit["pilot_report_sha256"] != sha(args.pilot_report):
        raise ValueError("Wrong capture pilot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("A published archive must not be overwritten")
    with tempfile.TemporaryDirectory(prefix="publish_features_", dir=args.output.parent) as temp:
        features = Path(temp) / "derived_features.npz"
        with np.load(source, allow_pickle=False) as arrays:
            allowed = {"X", "codec_X", "RGB_X", "match_ids", "split", "clip_ids", "view_index",
                       "latent_frame_index", "source_frame_index", "timestamps", "target_names", "sites",
                       "player_ids", "row_player_ids", "player_id", "canonical_player_ids", "chunk_index"}
            names = set(arrays.files) - {"y"}
            if not names <= allowed or not {"X", "codec_X", "RGB_X", "clip_ids"} <= names:
                raise ValueError(f"Unexpected aggregate publication schema: {sorted(names)}")
            np.savez_compressed(features, **{key: arrays[key] for key in sorted(names)})
        with np.load(features, allow_pickle=False) as arrays:
            if "y" in arrays.files or len(arrays["X"]) != audit["rows"]:
                raise ValueError("Published feature coverage or source-label exclusion failed")
            with np.load(source, allow_pickle=False) as original:
                if set(arrays.files) != set(original.files) - {"y"}:
                    raise ValueError("A derived array was omitted")
                for name in arrays.files:
                    if not np.array_equal(arrays[name], original[name]):
                        raise ValueError(f"Published array differs from audited aggregate: {name}")
        files = [("derived_features.npz", features)]
        controls = pilot["clips"][0]["controls"]
        if not controls["full_tensor_pooling_passed"] or len(controls["full_tensor_pooling"]) != 17:
            raise ValueError("Missing full residual references from the passing pilot")
        for item in controls["full_tensor_pooling"]:
            path = Path(item["path"])
            if sha(path) != item["sha256"]:
                raise ValueError("Pilot residual hash changed")
            files.append(("pilot_full/" + path.name, path))
        files += [("aggregate_audit.json", args.aggregate_audit), ("capture_pilot.json", args.pilot_report),
                  ("confirmation.json", args.confirmation), ("completion_audit.json", args.completion_audit),
                  ("THIRD_PARTY.md", ROOT / "THIRD_PARTY.md")]
        manifest = {"scope": "All pooled research features and baselines; all 17 full pilot residuals",
                    "rows": audit["rows"], "match_counts": audit["match_counts"],
                    "source_labels_and_video_included": False,
                    "all_derived_arrays_equal_audited_source": True,
                    "source_label_reproduction": "Accept Rocket Science terms and use the pinned data/capture/aggregation scripts; source y is deliberately excluded from this public derived-feature archive.",
                    "source_aggregate_sha256": audit["analysis_npz_sha256"],
                    "registration_sha256": audit["registration_sha256"],
                    "files": [{"path": name, "sha256": sha(path), "bytes": path.stat().st_size}
                              for name, path in files]}
        with args.output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, path in files:
                    info = tarfile.TarInfo("observations/" + name)
                    info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                content = (json.dumps(manifest, indent=2) + "\n").encode()
                info = tarfile.TarInfo("observations/manifest.json")
                info.size, info.mode, info.mtime = len(content), 0o644, 0
                archive.addfile(info, io.BytesIO(content))
        # Read the actual finished archive and verify every member, not only the inputs.
        with tarfile.open(args.output, "r:gz") as archive:
            expected = {"observations/" + item["path"]: item for item in manifest["files"]}
            if set(archive.getnames()) != set(expected) | {"observations/manifest.json"}:
                raise ValueError("Archive member coverage failed")
            for name, item in expected.items():
                digest = hashlib.sha256()
                with archive.extractfile(name) as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b""):
                        digest.update(chunk)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"Archived member hash mismatch: {name}")
        report = {**manifest, "status": "passed", "archive": args.output.name,
                  "archive_sha256": sha(args.output), "archive_bytes": args.output.stat().st_size,
                  "all_archive_members_verified": True, "packaging_script_sha256": sha(Path(__file__)),
                  "release_url": "https://github.com/guptaishaan/mira-interp/releases/tag/observations-2026-09-06"}
        (ROOT / "results/observation_archive.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
