#!/usr/bin/env python3
"""Package all audited conditional-fidelity tensors; no source labels or pixels."""
from __future__ import annotations
import argparse
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

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_inventory(file):
    with np.load(file, allow_pickle=False) as data:
        result = {}
        for name in data.files:
            array = data[name]
            require(array.dtype == np.float32 and np.isfinite(array).all(), "Expected finite FP32 derived tensors")
            result[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/sparse_causal_fidelity_v1")
    parser.add_argument("--archive", type=Path, default=Path("/data2/ishaangp/mira-interp/publication/mira_sparse_conditional_fidelity_v1.tar.gz"))
    parser.add_argument("--report", type=Path, default=ROOT / "results/sparse_fidelity_archive.json")
    args = parser.parse_args()
    started = time.monotonic()
    partial = args.archive.with_suffix(args.archive.suffix + ".partial")
    require(not args.archive.exists() and not partial.exists() and not args.report.exists(), "Preserve existing archive, partial attempt, or completion report")
    registration = read(args.directory / "registration.json")
    result, audit = read(args.directory / "fidelity_results.json"), read(args.directory / "fidelity_audit.json")
    pilot, pilot_audit = read(args.directory / "pilot.json"), read(args.directory / "pilot_audit.json")
    require(audit["status"] == pilot_audit["status"] == "passed" and audit["conditions_checked"] == 1364
            and pilot_audit["conditions_checked"] == 62 and audit["fidelity_results_sha256"] == sha(args.directory / "fidelity_results.json")
            and pilot_audit["pilot_sha256"] == sha(args.directory / "pilot.json"), "Matching full/pilot independent audits required")
    reg_hash = sha(args.directory / "registration.json")
    require(all(value["registration_sha256"] == reg_hash for value in (result, audit, pilot, pilot_audit)), "Registration lineage differs")
    require(result["n_pairs_seeds"] == 22 and result["dictionary_count"] == 30 and len(registration["dictionaries"]) == 30,
            "Incomplete full study")
    for relative, digest in registration["code_sha256"].items():
        require(sha(ROOT / relative) == digest, "Frozen execution source changed")
    prerequisite_hashes = {}
    for check in (audit, pilot_audit):
        for item in check["artifact_bindings"]:
            require(sha(Path(item["path"])) == item["sha256"], "Audited prerequisite changed")
            prerequisite_hashes[item["path"]] = item["sha256"]
    identifiers = {"discovery_mean", *[item["id"] for item in registration["dictionaries"]]}
    expected = {"baseline_flow", "native_donor_flow", "recipient_tile", "donor_tile", "recipient_z_t", "recipient_clean_codec", "native_descriptors"}
    expected |= {identifier + "_" + suffix for identifier in identifiers for suffix in
                 ("reconstructed_descriptors", "codes", "recipient_reconstruction_flow", "donor_reconstruction_flow")}
    shapes = {"baseline_flow": [1, 8, 36, 16, 32], "native_donor_flow": [1, 8, 36, 16, 32],
              "recipient_z_t": [1, 8, 36, 16, 32], "recipient_clean_codec": [4, 8, 9, 16, 32],
              "recipient_tile": [9, 16, 2048], "donor_tile": [9, 16, 2048], "native_descriptors": [2, 1536]}
    for identifier in identifiers:
        shapes[identifier + "_reconstructed_descriptors"] = [2, 1536]
        shapes[identifier + "_codes"] = [2, 0 if identifier == "discovery_mean" else 3072]
        for condition in ("recipient_reconstruction", "donor_reconstruction"):
            shapes[identifier + "_" + condition + "_flow"] = [1, 8, 36, 16, 32]
    files, tensors = {}, {}
    def add(path, member, digest=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and member not in files, "Nonregular or duplicate archive member")
        actual = sha(path)
        require(digest is None or actual == digest, "Archive input changed")
        files[member] = {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}
    entries = [("pilot", pilot["pair"])] + [("selection", item) for item in result["pairs"]]
    seen = set()
    for role, entry in entries:
        pair = read(entry["path"])
        require(sha(Path(entry["path"])) == entry["sha256"] and pair["status"] == "passed"
                and pair["split"] == role and pair["registration_sha256"] == reg_hash and pair["raw_source_labels_included"] is False,
                "Pair identity or label policy differs")
        identity = (role, pair["match_id"], pair["seed"])
        require(identity not in seen, "Duplicate tensor pair")
        seen.add(identity)
        path = Path(pair["artifact_path"])
        inventory = array_inventory(path)
        require(set(inventory) == expected and all(inventory[key]["shape"] == shapes[key] for key in expected),
                "Tensor archive includes unapproved source fields or invalid derived dimensions")
        member = f"tensors/{role}/{path.name}"
        add(path, member, pair["artifact_sha256"])
        tensors[member] = inventory
        add(entry["path"], f"reports/{role}/{Path(entry['path']).name}", entry["sha256"])
    require(len(seen) == 23 and len([x for x in seen if x[0] == "pilot"]) == 1, "All pilot/full tensors must be included")
    for name in ("registration.json", "pilot.json", "pilot_audit.json", "fidelity_results.json", "fidelity_audit.json", "plot_export.json"):
        add(args.directory / name, "reports/" + name)
    plots = read(args.directory / "plot_export.json")
    require(plots["status"] == "passed_audited_plot_export" and plots["fidelity_results_sha256"] == sha(args.directory / "fidelity_results.json")
            and plots["fidelity_audit_sha256"] == sha(args.directory / "fidelity_audit.json"), "Audited figures required")
    for name, digest in plots["artifacts"].items():
        require(Path(name).name == name and Path(name).suffix in {".png", ".pdf"}, "Unapproved figure path")
        add(args.directory / name, "figures/" + name, digest)
    add(ROOT / "docs/review/sparse_causal_fidelity.md", "docs/sparse_causal_fidelity.md")
    add(ROOT / "THIRD_PARTY.md", "docs/THIRD_PARTY.md")
    for name in ("scripts/sparse_causal_fidelity.py", "src/mira_interp/sparse_causal_fidelity.py",
                 "scripts/audit_sparse_causal_fidelity.py", "scripts/plot_sparse_causal_fidelity.py",
                 "scripts/package_sparse_causal_fidelity.py", "tests/test_sparse_causal_fidelity.py"):
        add(ROOT / name, "code/" + name)
    manifest = {"scope": "Derived conditional sparse causal fidelity; no raw source physical labels, actions or pixels",
                "registration_sha256": reg_hash, "fidelity_audit_sha256": sha(args.directory / "fidelity_audit.json"),
                "tensor_archives": 23, "conditions_full": 1364, "conditions_pilot": 62,
                "members": {key: {k: value[k] for k in ("sha256", "bytes")} for key, value in files.items()},
                "tensor_arrays": tensors, "physical_control_established": False,
                "note": "Codec/residual arrays are derived model representations. All original source labels/actions/video remain excluded. Native descriptor complement is retained, so results do not establish dictionary sufficiency."}
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=1, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for member, item in sorted(files.items()):
                info = tarfile.TarInfo(member)
                info.size, info.mode, info.mtime = item["bytes"], 0o644, 0
                with Path(item["path"]).open("rb") as source:
                    archive.addfile(info, source)
            info = tarfile.TarInfo("MANIFEST.json")
            info.size, info.mode, info.mtime = len(manifest_bytes), 0o644, 0
            archive.addfile(info, io.BytesIO(manifest_bytes))
    checked_arrays = 0
    with tarfile.open(partial, "r|gz") as archive:
        seen_members = set()
        for member in archive:
            require(member.isfile() and member.name not in seen_members, "Nonregular or duplicate stored member")
            seen_members.add(member.name)
            contents = archive.extractfile(member).read()
            if member.name == "MANIFEST.json":
                require(contents == manifest_bytes, "Stored manifest changed")
                continue
            require(member.name in files and hashlib.sha256(contents).hexdigest() == files[member.name]["sha256"], "Archive member bytes differ")
            if member.name in tensors:
                inventory = array_inventory(io.BytesIO(contents))
                require(inventory == tensors[member.name], "Saved tensor archive array readback differs")
                checked_arrays += len(inventory)
        require(seen_members == {*files, "MANIFEST.json"}, "Missing/extra archive members")
    require(all(sha(Path(item["path"])) == item["sha256"] for item in files.values()), "Inputs changed during packaging")
    require(all(sha(Path(path)) == digest for path, digest in prerequisite_hashes.items()), "Audited prerequisite changed during packaging")
    require(partial.stat().st_size < 2_000_000_000, "Release asset exceeds the registered 2GB size limit")
    partial.rename(args.archive)
    report = {"status": "passed_derived_sparse_fidelity_archive", "archive_path": str(args.archive.resolve()),
              "archive_sha256": sha(args.archive), "archive_bytes": args.archive.stat().st_size,
              "members_verified": len(files)+1, "tensor_archives_verified": 23, "arrays_verified": checked_arrays,
              "registration_sha256": reg_hash, "fidelity_audit_sha256": sha(args.directory / "fidelity_audit.json"),
              "all_members_and_arrays_read_back": True, "source_physical_labels_included": False,
              "source_actions_or_video_included": False, "physical_control_established": False,
              "script_sha256": sha(Path(__file__)), "elapsed_seconds": time.monotonic()-started}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
