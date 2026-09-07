#!/usr/bin/env python3
"""Publish an audited derived archive to the user's authorized repository."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GH = "/ccn2/u/ishaangp/bin/gh"
REPO = "guptaishaan/mira-interp"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def api(suffix):
    return json.loads(subprocess.check_output([GH, "api", "repos/" + REPO + "/" + suffix]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-report", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise RuntimeError("Publication report is immutable; inspect the completed record")
    report_hash, notes_hash = sha(args.archive_report), sha(args.notes_file)
    source = json.loads(args.archive_report.read_text())
    if source["status"] == "passed_derived_sparse_fidelity_archive":
        if not source["all_members_and_arrays_read_back"] or source["tensor_archives_verified"] != 23 or source["arrays_verified"] != 3013:
            raise RuntimeError("Sparse fidelity archive readback is incomplete")
        source = {**source, "status":"passed"}
    if source["status"] == "passed_derived_artifact_release_audit":
        if not (source["all_members_readback_verified"] and source["all_included_derived_source_hashes_verified"]):
            raise RuntimeError("Derived archive readback/provenance audit is incomplete")
        source = {**source, "status":"passed", "archive_path":source["archive"]["path"],
                  "archive_bytes":source["archive"]["bytes"], "archive_sha256":source["archive"]["sha256"]}
    archive = Path(source["archive_path"])
    if source["status"] != "passed" or source["archive_sha256"] != sha(archive):
        raise RuntimeError("Archive must pass local audit before publication")
    if archive.stat().st_size != source["archive_bytes"] or archive.stat().st_size >= 2_000_000_000:
        raise RuntimeError("Archive size differs or exceeds the conservative asset bound")
    if len(args.target) != 40 or api("commits/" + args.target)["sha"] != args.target:
        raise RuntimeError("An exact already-published target commit is required")
    status_path = args.report.with_name(args.report.stem + "_status.json")
    state = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
             "tag": args.tag, "target_commit": args.target, "archive_path": str(archive),
             "archive_sha256": source["archive_sha256"], "archive_report_sha256": report_hash,
             "release_notes_sha256": notes_hash}
    if status_path.exists():
        old = json.loads(status_path.read_text())
        for key in ("tag", "target_commit", "archive_path", "archive_sha256", "archive_report_sha256", "release_notes_sha256"):
            if old[key] != state[key]:
                raise RuntimeError("Resumed publication does not match the original request")
    save(status_path, state)
    try:
        # This read permits resumption without overwriting an existing release or asset.
        probe = subprocess.run([GH, "api", "repos/" + REPO + "/releases/tags/" + args.tag],
                               capture_output=True, text=True)
        if probe.returncode:
            if "404" not in probe.stderr:
                raise RuntimeError("Cannot establish whether the release exists")
            subprocess.run([GH, "release", "create", args.tag, "--repo", REPO, "--target", args.target,
                            "--title", args.title, "--notes-file", str(args.notes_file)], check=True)
            release = api("releases/tags/" + args.tag)
        else:
            release = json.loads(probe.stdout)
        if release["target_commitish"] != args.target:
            raise RuntimeError("Release points at another commit; refusing to overwrite")
        assets = [value for value in release["assets"] if value["name"] == archive.name]
        if not assets:
            state["stage"] = "uploading"
            save(status_path, state)
            subprocess.run([GH, "release", "upload", args.tag, str(archive), "--repo", REPO], check=True)
        release = api("releases/tags/" + args.tag)
        assets = [value for value in release["assets"] if value["name"] == archive.name]
        if len(assets) != 1:
            raise RuntimeError("Exactly one matching uploaded asset is required")
        asset = assets[0]
        if (asset["state"] != "uploaded" or asset["size"] != source["archive_bytes"]
                or asset["digest"] != "sha256:" + source["archive_sha256"]):
            raise RuntimeError("Remote asset state, size or digest differs")
        if sha(archive) != source["archive_sha256"] or sha(args.archive_report) != report_hash or sha(args.notes_file) != notes_hash:
            raise RuntimeError("Publication inputs changed during upload")
        result = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
                  "release_url": release["html_url"], "release_id": release["id"], "target_commit": args.target,
                  "asset": {key: asset[key] for key in ("id", "name", "size", "digest", "state", "browser_download_url")},
                  "api_size_and_digest_equal_local_verified_archive": True,
                  "archive_report_sha256": report_hash, "release_notes_sha256": notes_hash,
                  "publication_script_sha256": sha(Path(__file__))}
        save(args.report, result)
        state.update(status="passed", stage="complete", publication_report_sha256=sha(args.report))
        save(status_path, state)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        save(status_path, state)
        raise


if __name__ == "__main__":
    main()
