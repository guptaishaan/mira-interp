#!/usr/bin/env python3
"""Export audited generated future pixels only; previews are not physical evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def future_uint8(frames: np.ndarray) -> np.ndarray:
    """View, future time, height, width, RGB; frames 0..15 never reach renderers."""
    if frames.shape != (4, 24, 3, 288, 512) or frames.dtype != np.float32:
        raise ValueError("Expected saved float32 [4,24,3,288,512] frames")
    future = frames[:, 16:]
    if not np.isfinite(future).all() or future.min() < 0 or future.max() > 1:
        raise ValueError("Generated pixels outside finite [0,1]")
    return np.rint(future.transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def contact_sheet(videos: list[np.ndarray], labels: list[str]) -> Image.Image:
    canvas = Image.new("RGB", (2048, 714), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), "Reserved engineering pilot: generated future only, view 0", fill="black", font=font(24))
    draw.text((10, 38), "Baseline and one quadratic-path +300 uu edit; paired seed. Visual comparison is not calibrated physical validation.", fill="black", font=font(18))
    for row, (video, label) in enumerate(zip(videos, labels, strict=True)):
        top = 70 + row * 320
        draw.text((10, top), label, fill="black", font=font(21))
        for col, t in enumerate((1, 3, 5, 7)):
            canvas.paste(Image.fromarray(video[0, t]), (512 * col, top + 28))
            draw.text((512 * col + 8, top + 34), f"Frame {16+t}; +{(t+1)/20:.2f} s", fill="white", stroke_fill="black", stroke_width=2, font=font(18))
    return canvas


def four_view_video(future: np.ndarray, label: str) -> np.ndarray:
    """2x2 canonical view grid; only eight generated frames enter this function."""
    if future.shape != (4, 8, 288, 512, 3):
        raise ValueError("Expected only generated frames")
    result = []
    for t in range(8):
        canvas = Image.new("RGB", (1024, 640), "black")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), label, fill="white", font=font(20))
        draw.text((10, 34), f"Generated only | original frame {16+t} | +{(t+1)/20:.2f} s | views 0,1 / 2,3", fill="white", font=font(18))
        for view in range(4):
            x, y = (view % 2) * 512, 64 + (view // 2) * 288
            canvas.paste(Image.fromarray(future[view, t]), (x, y))
            draw.text((x + 8, y + 5), f"View {view}", fill="white", stroke_fill="black", stroke_width=2, font=font(17))
        result.append(np.asarray(canvas))
    return np.stack(result)


def write_video(path: Path, frames: np.ndarray, fps: int) -> dict:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium", "threads": "2"}
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
        if len(decoded) != 8 or float(stream.average_rate) != fps:
            raise ValueError("Encoded video frame count or fps differs")
        if any((frame.height, frame.width) != frames.shape[1:3] for frame in decoded):
            raise ValueError("Encoded video dimensions differ")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size,
            "frames": 8, "fps": fps, "duration_seconds": 8 / fps, "decoded_readback_passed": True,
            "encoding": "H264 CRF18, yuv420p; lossy viewing copy", "playback": "original" if fps == 20 else "4x slower"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/rollout_steering_v1/pilot.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "results/rollout_steering_v1/pilot_audit.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rollout_steering_v1/pilot_previews")
    args = parser.parse_args()
    started = time.monotonic()
    manifest_path, audit_path, out = args.manifest.resolve(), args.audit.resolve(), args.output_dir.resolve()
    bound = {manifest_path: sha(manifest_path), audit_path: sha(audit_path), Path(__file__).resolve(): sha(Path(__file__))}
    manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
    if manifest.get("status") != "passed_rollout_steering_pilot" or audit.get("status") != "passed_rollout_generation_audit":
        raise ValueError("Passed generation pilot and independent audit required")
    if audit.get("phase") != "pilot" or audit.get("manifest_sha256") != bound[manifest_path]:
        raise ValueError("Audit is not bound to this pilot")
    if audit.get("registration_sha256") != manifest.get("registration_sha256"):
        raise ValueError("Registration differs")
    records = manifest["records"]
    if len(records) != 2 or {r["intervention_type"] for r in records} != {"baseline", "quadratic_forward"}:
        raise ValueError("Expected exactly the reserved baseline and active pilot")
    records.sort(key=lambda r: r["intervention_type"] != "baseline")
    baseline, active = records
    if baseline["dose"] != 0 or active["dose"] != 300 or baseline["split"] != "pilot" or active["split"] != "pilot":
        raise ValueError("Unexpected pilot condition")
    if any(r["baseline_record_id"] != baseline["record_id"] for r in records):
        raise ValueError("Conditions are not paired to baseline")
    if any(baseline[k] != active[k] for k in ("match_id", "clip_id", "seed", "action_batch_sha256")):
        raise ValueError("Conditions differ in source, seed or actions")
    audited = {r["record_id"]: r["generated_sha256"] for r in audit["records"]}
    if audited != {r["record_id"]: r["generated_sha256"] for r in records}:
        raise ValueError("Audit record set differs")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    videos, sources = [], []
    for record in records:
        path = Path(record["generated_artifact_path"])
        bound[path] = record["generated_sha256"]
        if sha(path) != bound[path]:
            raise ValueError("Generated artifact hash differs")
        report = Path(record["record_report"]["path"])
        bound[report] = record["record_report"]["sha256"]
        if sha(report) != bound[report]:
            raise ValueError("Record report hash differs")
        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"]
            fp32_future_sha = hashlib.sha256(np.ascontiguousarray(frames[:, 16:]).tobytes()).hexdigest()
            future = future_uint8(frames)
        videos.append(future)
        sources.append({"record_id": record["record_id"], "generated_sha256": bound[path],
                        "future_float32_sha256": fp32_future_sha, "future_uint8_sha256": hashlib.sha256(future.tobytes()).hexdigest(),
                        "source_frame_indices": list(range(16,24)), "decoded_context_exact_per_generation_audit": record["decoded_context_bitwise_equal_to_baseline"]})
    labels = ["Baseline (no edit)", "Quadratic path: requested +300 uu (one internal edit)"]
    sheet = contact_sheet(videos, labels)
    png = out / "baseline_vs_quadratic_pilot.png"
    sheet.save(png)
    with Image.open(png) as loaded:
        if not np.array_equal(np.asarray(loaded), np.asarray(sheet)):
            raise ValueError("PNG readback differs")
    outputs = [{"path": str(png.relative_to(ROOT)), "sha256": sha(png), "bytes": png.stat().st_size,
                "source_frame_indices": [17,19,21,23], "view_index": 0, "png_readback_exact": True}]
    for stem, future, label in zip(("baseline", "quadratic_plus300"), videos, labels, strict=True):
        frames = four_view_video(future, label + " | 20 fps original")
        outputs.append(write_video(out / f"{stem}_four_views_20fps.mp4", frames, 20))
        slow = four_view_video(future, label + " | 5 fps, 4x slower")
        outputs.append(write_video(out / f"{stem}_four_views_5fps_slow.mp4", slow, 5))
    for path, expected in bound.items():
        if sha(path) != expected:
            raise ValueError(f"Input changed during export: {path}")
    report = {"status": "passed_generated_future_preview_export", "manifest_sha256": bound[manifest_path],
              "audit_sha256": bound[audit_path], "registration_sha256": manifest["registration_sha256"],
              "script_sha256": bound[Path(__file__).resolve()], "sources": sources, "outputs": outputs,
              "observed_context_exported": False, "source_physical_labels_loaded": False,
              "physical_control_established": False, "view_order": "0,1 top; 2,3 bottom",
              "conversion": "round-to-nearest-even(float32 generated future * 255), uint8",
              "elapsed_seconds": time.monotonic() - started}
    (out / "preview_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
