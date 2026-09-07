#!/usr/bin/env python3
"""Download and hash-audit the pinned 62-match test set within a 40 GiB budget."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "kyutai/rocket-science"
REVISION = "a248fc918baa93389e3242fefba193f1bfb0474a"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "repo": REPO, "revision": REVISION,
              "budget_gib": 40, "disk_reserve_gib": 25, "files": [],
              "scope": "test split download integrity; video/state alignment audited separately"}
    output = ROOT / "results/dataset_download.json"
    def save():
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
    save()
    try:
        info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        selected = sorted([f for f in info.siblings if f.rfilename.startswith("test/") and
                           (f.rfilename.endswith(".tar") or f.rfilename.endswith("index.json"))],
                          key=lambda f: f.rfilename)
        if info.sha != REVISION or len(selected) != 12:
            raise ValueError("Expected the pinned test index plus exactly 11 shards")
        report["expected_bytes"] = sum(f.size for f in selected)
        if report["expected_bytes"] > 40 * 2**30:
            raise RuntimeError("Requested data exceed the declared total budget")
        # Full duplicate-space allowance handles corrupt/partial cached assets safely.
        if shutil.disk_usage(args.destination).free < report["expected_bytes"] + 25 * 2**30:
            raise RuntimeError("Insufficient disk space including full fresh-copy allowance")
        save()
        for f in selected:
            if shutil.disk_usage(args.destination).free < f.size + 25 * 2**30:
                raise RuntimeError("Storage reserve failed before the next shard")
            print("Download/verify", f.rfilename, flush=True)
            path = Path(hf_hub_download(REPO, f.rfilename, repo_type="dataset", revision=REVISION,
                                        local_dir=args.destination))
            sha = digest(path)
            if path.stat().st_size != f.size:
                raise ValueError("Downloaded file size mismatch")
            expected = f.lfs.sha256 if f.lfs else None
            if expected and sha != expected:
                raise ValueError("Downloaded file SHA-256 mismatch")
            if not expected:
                blob = hashlib.sha1(f"blob {f.size}\0".encode() + path.read_bytes()).hexdigest()
                if blob != f.blob_id:
                    raise ValueError("Index Git blob hash mismatch")
            report["files"].append({"path": f.rfilename, "bytes": f.size, "sha256": sha,
                                     "publisher_sha256": expected, "verified": True})
            save()
        report["status"] = "PASS"
        save()
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__,
                      http_status=getattr(getattr(exc, "response", None), "status_code", None))
        save()
        raise


if __name__ == "__main__":
    main()
