#!/usr/bin/env python3
"""Download pinned, publicly released MIRA Mini assets and verify publisher hashes."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "alakazamworld/mira-mini-4p"
REVISION = "d58ee2f9bca27289554c1e652943dc0e539e8971"
FILES = ["README.md", "world_model_config.yaml", "codec/codec_config.yaml",
         "context/default.npz", "codec/checkpoint-125000/checkpoint.pth",
         "checkpoint-90000/checkpoint.pth"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--report", type=Path, default=Path("results/model_download.json"))
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    metadata = {f.rfilename: f for f in info.siblings}
    report = {"repo": REPO, "revision": REVISION, "scope": "third-party pretrained reproduction",
              "license": "CC-BY-NC-SA-4.0", "status": "RUNNING", "files": []}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = args.report.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.report)
    save()
    try:
        # Conservatively allow a full fresh copy even when partial/stale files exist.
        required = sum(metadata[f].size for f in FILES)
        if shutil.disk_usage(args.destination).free < required + 25 * 2**30:
            raise RuntimeError("Download requires expected bytes plus a 25 GiB free-space reserve")
        for name in FILES:
            if shutil.disk_usage(args.destination).free < metadata[name].size + 25 * 2**30:
                raise RuntimeError("Storage reserve failed before the next asset")
            print(f"Downloading/verifying {name}", flush=True)
            path = Path(hf_hub_download(REPO, name, revision=REVISION,
                                       local_dir=args.destination, token=False))
            digest = hashlib.sha256()
            blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 2**20), b""):
                    digest.update(block)
                    blob.update(block)
            expected = metadata[name].lfs.sha256 if metadata[name].lfs else None
            expected_blob = None if metadata[name].lfs else metadata[name].blob_id
            if (path.stat().st_size != metadata[name].size or
                    (expected and digest.hexdigest() != expected) or
                    (not expected and (not expected_blob or blob.hexdigest() != expected_blob))):
                raise RuntimeError(f"Integrity mismatch: {name}")
            report["files"].append({"path": name, "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(), "publisher_sha256": expected,
                "publisher_git_blob_sha1": expected_blob, "verified": True})
            save()
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__)
        save()
        raise
    report["status"] = "PASS"
    save()
    print("All model assets verified", flush=True)


if __name__ == "__main__":
    main()
