#!/usr/bin/env python3
"""Check authorized Rocket Science access, or freeze a local whole-match split.

The selected index is downloaded, with an optional single budgeted shard. This
command does not accept gating terms or claim a prepared research dataset exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira_interp.data import make_split_manifest

PINNED_REVISION = "a248fc918baa93389e3242fefba193f1bfb0474a"
DISK_RESERVE_BYTES = 25 * 1024**3


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_cached_bytes(path: Path, metadata) -> int:
    """Do not credit truncated, corrupt, or old-revision files toward disk reuse."""
    if not path.is_file() or path.stat().st_size != metadata.size:
        return 0
    expected = getattr(metadata.lfs, "sha256", None)
    if expected:
        return metadata.size if file_sha256(path) == expected else 0
    digest = hashlib.sha1(f"blob {metadata.size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return metadata.size if digest.hexdigest() == metadata.blob_id else 0


def safe_relative(path: str) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
        raise ValueError("unsafe relative dataset path")
    return candidate.as_posix()


def check_access(output: Path, upstream_split: str, budget_bytes: int,
                 revision: str, data_dir: Path) -> tuple[dict, Path | None]:
    from huggingface_hub import HfApi, get_token, hf_hub_download

    filename = f"{upstream_split}/index.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(data_dir)
    report = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "repo_id": "kyutai/rocket-science", "requested_file": filename,
              "cached_auth_present": bool(get_token()), "disk_free_bytes": disk.free,
              "payload_budget_bytes": budget_bytes, "dataset_payload_downloaded_bytes": 0,
              "required_disk_reserve_bytes": DISK_RESERVE_BYTES, "requested_revision": revision,
              "terms_accepted_by_this_run": False, "video_shards_downloaded": 0,
              "status": "checking", "research_dataset_ready": False}
    local_path = None
    try:
        info = HfApi().dataset_info(report["repo_id"], revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("resolved dataset revision differs from the pinned commit")
        report.update(source_revision=info.sha, gated=info.gated,
                      index_files=[{"name": f.rfilename, "size_bytes": f.size} for f in info.siblings
                                   if f.rfilename.endswith("/index.json")])
        metadata = next((f for f in info.siblings if f.rfilename == filename), None)
        if metadata is None or metadata.size is None:
            raise ValueError("requested index has no verified size metadata")
        existing_bytes = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file() and ".cache" not in p.parts)
        cached_size = verified_cached_bytes(data_dir / filename, metadata)
        new_bytes = max(0, metadata.size - cached_size)
        if existing_bytes + new_bytes > budget_bytes or disk.free - new_bytes < DISK_RESERVE_BYTES:
            report["status"] = "blocked_download_budget_or_disk"
        else:
            local_path = Path(hf_hub_download(report["repo_id"], filename, repo_type="dataset",
                                             revision=info.sha, local_dir=data_dir))
            if local_path.stat().st_size != metadata.size:
                raise ValueError("downloaded index size differs from hub metadata")
            # Hub verifies transport hashes; record a content hash for the frozen manifest.
            report.update(status="index_available_only", dataset_payload_downloaded_bytes=local_path.stat().st_size,
                          index_sha256=file_sha256(local_path), local_index=str(local_path))
    except Exception as exc:
        # Never serialize exception text: it can contain signed URLs or headers.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        report["error"] = {"type": type(exc).__name__, "http_status": status}
        report["status"] = "blocked_data_access" if status in (401, 403) else "blocked_access_check_error"
        report["next_required_action"] = (
            "Authenticate this node with the user's Hugging Face account that has dataset access, then rerun this command."
            if status in (401, 403) else "Resolve the recorded access-check failure, then rerun this command.")
    save_json(output, report)
    return report, local_path


def download_one_shard(report: dict, index: dict, data_dir: Path, budget_bytes: int,
                       upstream_split: str, output: Path) -> Path:
    """Fetch one referenced shard within the initial total raw-data budget; never extract."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(report["repo_id"], revision=report["source_revision"], files_metadata=True)
    if info.sha != report["source_revision"]:
        raise ValueError("shard metadata revision differs from the pinned index")
    files = {item.rfilename: item for item in info.siblings}
    candidates = []
    for shard in {entry["shard"] for entry in index["entries"]}:
        filename = f"{upstream_split}/{safe_relative(shard)}"
        item = files.get(filename)
        size = item.size if item else None
        if size is None or not filename.endswith(".tar"):
            raise ValueError("indexed shard lacks valid file-size metadata")
        candidates.append((size, filename))
    if not candidates:
        raise ValueError("index contains no shards")
    size, filename = min(candidates)
    existing_bytes = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file() and ".cache" not in p.parts)
    cached_size = verified_cached_bytes(data_dir / filename, files[filename])
    new_bytes = max(0, size - cached_size)
    free = shutil.disk_usage(data_dir).free
    if existing_bytes + new_bytes > budget_bytes or free - new_bytes < DISK_RESERVE_BYTES:
        report.update(status="blocked_shard_budget_or_disk", requested_shard=filename,
                      requested_shard_bytes=size, disk_free_before_shard_bytes=free)
        save_json(output, report)
        raise RuntimeError("shard exceeds initial budget or disk reserve")
    path = Path(hf_hub_download(report["repo_id"], filename, repo_type="dataset",
                               revision=report["source_revision"], local_dir=data_dir))
    if path.stat().st_size != size:
        raise ValueError("downloaded shard size differs from hub metadata")
    actual_sha256 = file_sha256(path)
    expected_sha256 = getattr(files[filename].lfs, "sha256", None)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError("shard digest cannot be verified against hub LFS metadata")
    # Read only tar headers. No member is extracted or its path written to disk.
    member_names = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            safe_relative(member.name)
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("unexpected special file in dataset archive")
            if member.isfile():
                basename = PurePosixPath(member.name).name
                if basename in member_names:
                    raise ValueError("ambiguous duplicate archive member basename")
                member_names.add(basename)
    shard_relative = filename.removeprefix(upstream_split + "/")
    entries = [entry for entry in index["entries"] if entry["shard"] == shard_relative]
    required = set()
    for entry in entries:
        chunks = entry.get("chunk_indices") or range(len(entry["chunk_frames"]))
        for chunk in chunks:
            key = f"{entry['match_id']}_c{chunk:05d}"
            for player in range(4):
                required.update(f"{key}.p{player}.{suffix}" for suffix in ("mp4", "jsonl", "physics.jsonl"))
    missing = required - member_names
    report.update(status="shard_inventory_available_only" if not missing else "blocked_shard_inventory",
                  video_shards_downloaded=1, shard_filename=filename, shard_size_bytes=size,
                  shard_sha256=actual_sha256, shard_expected_sha256=expected_sha256, shard_match_count=len(entries),
                  shard_member_count=len(member_names), expected_member_count=len(required),
                  missing_expected_members=sorted(missing),
                  dataset_payload_downloaded_bytes=report["dataset_payload_downloaded_bytes"] + size,
                  research_dataset_ready=False, decoded_video_checked=False, physics_alignment_checked=False)
    save_json(output, report)
    if missing:
        raise ValueError("indexed chunk components are missing from shard")
    save_json(path.parent / "downloaded_shard_index.json", {"total_samples": len(entries), "entries": entries})
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, action="append", help="local upstream index; repeat for multiple splits")
    parser.add_argument("--index-upstream-split", choices=("train", "dev", "test", "unknown"), action="append",
                        help="declared original partition for each local --index, in the same order")
    parser.add_argument("--revision", default=PINNED_REVISION, help="exact pinned source commit; defaults to the study revision")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw", help="raw-data directory; may be on /data2")
    parser.add_argument("--download-one-shard", action="store_true", help="also fetch the smallest indexed shard and audit its member inventory")
    parser.add_argument("--pilot-match-ids", type=Path, help="JSON list of engineering match IDs excluded from all research partitions")
    parser.add_argument("--upstream-split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--budget-gib", type=float, default=3.0)
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "data_access.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "split_manifest.json")
    parser.add_argument("--seed", default="mira-interp-v1")
    args = parser.parse_args()
    if not 0 < args.budget_gib <= 3:
        parser.error("the initial payload budget must be positive and at most 3 GiB")
    if args.index and args.download_one_shard:
        parser.error("--download-one-shard currently requires the hub-index path; omit --index")
    if args.index_upstream_split and (not args.index or len(args.index_upstream_split) != len(args.index)):
        parser.error("supply one --index-upstream-split for every --index")
    if len(args.revision) != 40 or any(char not in "0123456789abcdef" for char in args.revision):
        parser.error("--revision must be an exact lowercase 40-character commit SHA")
    if args.index:
        indices, revision = args.index, args.revision
    else:
        report, index = check_access(args.report, args.upstream_split, int(args.budget_gib * 1024**3),
                                     args.revision, args.data_dir)
        print(json.dumps(report, indent=2))
        if index is None:
            return 2
        indices, revision = [index], report["source_revision"]
    loaded_indices = [json.loads(index.read_text()) for index in indices]
    pilots = json.loads(args.pilot_match_ids.read_text()) if args.pilot_match_ids else []
    if args.index:
        sources = [{"index_path": str(path.resolve()), "upstream_split": split,
                    "basis": "explicit CLI declaration" if args.index_upstream_split else "unknown; not inferred from directory name"}
                   for path, split in zip(indices, args.index_upstream_split or ["unknown"] * len(indices))]
    else:
        sources = [{"index_path": str(indices[0].resolve()), "upstream_split": args.upstream_split,
                    "hub_path": f"{args.upstream_split}/index.json", "basis": "pinned Hub path"}]
    manifest = make_split_manifest(loaded_indices, revision=revision, seed=args.seed, pilot_match_ids=pilots,
                                   index_sources=sources)
    manifest["input_index_sha256"] = [file_sha256(index) for index in indices]
    save_json(args.manifest, manifest)
    if args.download_one_shard:
        try:
            path = download_one_shard(report, loaded_indices[0], args.data_dir,
                                      int(args.budget_gib * 1024**3), args.upstream_split, args.report)
            print(json.dumps({"downloaded_shard": str(path), "inventory_only": True}))
        except Exception as exc:
            report.update(status="blocked_shard_preparation", shard_error={"type": type(exc).__name__,
                          "http_status": getattr(getattr(exc, "response", None), "status_code", None)})
            save_json(args.report, report)
            print(json.dumps(report, indent=2))
            return 2
    print(json.dumps({"manifest": str(args.manifest), "match_counts": manifest["match_counts"],
                      "all_splits_nonempty": manifest["all_splits_nonempty"],
                      "research_dataset_ready": False,
                      "remaining": "download budgeted shards; audit pixels, physics, events, and source actions"}, indent=2))
    return 0 if manifest["all_splits_nonempty"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
