#!/usr/bin/env python3
"""Publish the verified fresh archive to the explicitly authorized fixed commit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
GH = "/ccn2/u/ishaangp/bin/gh"
REPO = "guptaishaan/mira-interp"
TAG = "fresh-confirmation-v3-2026-09-07"
COMMIT = "b713fe976674248bbde9fdbc96a7b13d1cb91f49"
STATUS = ROOT / "results/fresh_confirmation_upload_status.json"
REPORT = ROOT / "results/fresh_confirmation_publication.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def api(path):
    return json.loads(subprocess.check_output([GH, "api", "repos/" + REPO + "/" + path]))


def main():
    if REPORT.exists() or STATUS.exists():
        raise RuntimeError("A publication or upload record already exists; inspect it before retrying")
    archive_report_path = ROOT / "results/fresh_confirmation_archive.json"
    archive_report = json.loads(archive_report_path.read_text())
    archive = Path(archive_report["archive_path"])
    if archive_report["status"] != "passed" or sha(archive) != archive_report["archive_sha256"]:
        raise RuntimeError("Local verified archive changed")
    if api("commits/" + COMMIT)["sha"] != COMMIT:
        raise RuntimeError("Authorized target commit is not available")
    state = {"status": "running", "job_pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "tag": TAG, "target_commit": COMMIT, "archive": str(archive), "archive_sha256": archive_report["archive_sha256"]}
    save(STATUS, state)
    try:
        notes = archive.parent / (TAG + "-release-notes.md")
        notes.write_text("Frozen observational evaluation on 23 fresh matches (184 clips; 4,416 scored rows), with independently verified predictions, metrics and whole-match bootstrap intervals.\n\n"
            "This archive contains all derived feature captures and audit/protocol reports. Source videos, actions and simulator labels are excluded; derived downsampled RGB baselines are included. Horizontal velocity readout remains weak, including negative per-axis R². These results do not establish physical control.\n\n"
            "The exact probe coefficients are the separate development_models.npz asset in the [development release](https://github.com/guptaishaan/mira-interp/releases/tag/development-v3-2026-09-07).\n\n"
            "Archive SHA256: `" + archive_report["archive_sha256"] + "`.\n")
        subprocess.run([GH, "release", "create", TAG, "--repo", REPO, "--target", COMMIT,
                        "--title", "Fresh-match observational confirmation v3", "--notes-file", str(notes)], check=True)
        state["stage"] = "uploading"
        process = subprocess.Popen([GH, "release", "upload", TAG, str(archive), "--repo", REPO])
        state["upload_pid"] = process.pid
        save(STATUS, state)
        code = process.wait()
        state.update(stage="verify_remote", upload_exit_code=code)
        save(STATUS, state)
        release = api("releases/tags/" + TAG)
        assets = [item for item in release["assets"] if item["name"] == archive.name]
        if len(assets) != 1:
            raise RuntimeError("Uploaded archive is missing or duplicated")
        asset = assets[0]
        if release["target_commitish"] != COMMIT or asset["size"] != archive.stat().st_size or asset["digest"] != "sha256:" + archive_report["archive_sha256"] or asset["state"] != "uploaded":
            raise RuntimeError("Remote commit, size, state or digest differs")
        if sha(archive) != archive_report["archive_sha256"]:
            raise RuntimeError("Local archive changed during upload")
        result = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
                  "release_url": release["html_url"], "release_id": release["id"], "target_commit": COMMIT,
                  "asset": {key: asset[key] for key in ["id", "name", "size", "digest", "state", "browser_download_url"]},
                  "api_size_and_digest_equal_local_verified_archive": True,
                  "archive_report_sha256": sha(archive_report_path), "publication_script_sha256": sha(Path(__file__)), "upload_exit_code": code}
        save(REPORT, result)
        state.update(status="passed", stage="complete", publication_report_sha256=sha(REPORT))
        save(STATUS, state)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        save(STATUS, state)
        raise


if __name__ == "__main__":
    main()
