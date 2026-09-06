#!/usr/bin/env python3
"""Package every saved public-context readiness tensor for a GitHub release."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((ROOT / "results/pretrained_smoke.json").read_text())
    if report["status"] != "passed":
        raise RuntimeError("Pretrained readiness must pass before packaging")
    directory = Path(report["artifacts"]["activation_directory"]).resolve()
    files = [{"path": row["activation_file"], "sha256": row["sha256"]}
             for row in [report["block_zero_input"], *report["capture"]["layers"]]]
    files.extend(report["sample"][key] for key in ("saved_generated_frames", "saved_generated_latents"))
    paths = {Path(row["path"]).resolve() for row in files}
    if len(paths) != 19 or paths != {path.resolve() for path in directory.iterdir() if path.is_file()}:
        raise RuntimeError("Expected exactly the 19 documented outputs in the capture directory")
    manifest = []
    for row in files:
        path = Path(row["path"]).resolve()
        if path.parent != directory or sha256(path) != row["sha256"]:
            raise RuntimeError("A tensor path or digest does not match the completed run")
        manifest.append({"path": path.name, "sha256": row["sha256"], "bytes": path.stat().st_size})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for item in sorted(manifest, key=lambda x: x["path"]):
                path = directory / item["path"]
                info = tarfile.TarInfo("readiness/" + path.name)
                info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            extras = {
                "manifest.json": json.dumps(manifest, indent=2).encode(),
                "pretrained_smoke.json": (ROOT / "results/pretrained_smoke.json").read_bytes(),
                "THIRD_PARTY.md": (ROOT / "THIRD_PARTY.md").read_bytes(),
            }
            for name, content in extras.items():
                info = tarfile.TarInfo("readiness/" + name)
                info.size, info.mode, info.mtime = len(content), 0o644, 0
                archive.addfile(info, io.BytesIO(content))
    summary = {"scope": "public-context pretrained engineering outputs only",
               "archive": args.output.name, "archive_sha256": sha256(args.output),
               "archive_bytes": args.output.stat().st_size, "files": manifest,
               "license": "CC-BY-NC-SA-4.0; see THIRD_PARTY.md",
               "release_url": "https://github.com/guptaishaan/mira-interp/releases/tag/readiness-2026-09-06"}
    (ROOT / "results/readiness_archive.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
