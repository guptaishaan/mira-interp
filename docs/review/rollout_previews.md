# Audited generated-future previews

The reserved generation pilot has two saved conditions: a paired baseline and one quadratic-path edit requesting +300 unreal units at the fixed internal site. The exporter requires the completed generation audit, validates its manifest/registration/record bindings, and rechecks every source hash after rendering. It never publishes observed context: rendering receives only `frames[:,16:]`, the eight generated frames. It does not load source physical labels.

Run from the repository root into a new or empty output directory:

```bash
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  .venv/bin/python scripts/export_rollout_preview.py
```

The [contact sheet](../../results/rollout_steering_v1/pilot_previews/baseline_vs_quadratic_pilot.png) compares view 0 at original frame indices 17, 19, 21 and 23, respectively 0.10, 0.20, 0.30 and 0.40 seconds after the last observed frame. Four-view videos use canonical views 0/1 on top and 2/3 below. Files ending `20fps.mp4` preserve original timing; `5fps_slow.mp4` show the same eight frames four times slower and label that difference in the video. These short clips are lossy H264 viewing copies; the original generated float32 arrays remain the quantitative artifacts. Uint8 conversion is round-to-nearest-even after multiplying by 255.

The [export report](../../results/rollout_steering_v1/pilot_previews/preview_export.json) binds the sources and all five preview files. PNG readback is exact; each MP4 is decoded again to verify eight frames, timing and dimensions. A synthetic test marks every observed frame as NaN and encodes a different value for each future view/time, checking that context cannot enter the renderer and that grid ordering is preserved.

Visual inspection of the saved contact sheet shows similar baseline and edited futures without an obvious ball-height change at this display scale. This one reserved example checks presentation and visible decode quality. It does not establish a physical steering effect, monotonicity, persistence or evaluator accuracy. The generation audit reports context pixels unchanged for these two conditions; those context pixels are absent from every preview.
