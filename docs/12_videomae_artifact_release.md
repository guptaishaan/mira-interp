# VideoMAE derived artifact package

The completed development evaluator is packaged in `/data2/ishaangp/mira-interp/publication/videomae_v1/videomae-derived-v1.tar` (163,870,720 bytes, SHA256 `97c653601ea53914b62a67d3c53c2d0125c1f62810b59f88d9e1ff523f9ef7ec`). It contains 413 members: four saved neural checkpoints, two ridge checkpoints, 337 frozen feature artifacts (336 development clips and one reserved pilot), six prediction files, and the original code, aggregate reports, figures, and upstream attribution. It is below the requested 2 GB asset limit. Publication is handled separately by the parent workflow.

`results/videomae_artifact_release.json` records the completed audit. Every included private artifact was checked against its reported SHA256 where one was provided, every sanitized NPZ was reopened and compared array for array, and every tar member was read back and hashed. A second independent archive build produced the same SHA256. The embedded and sidecar `MANIFEST.json` has SHA256 `76982cbba6400cd34b9ca9ea0ceaf32ce6ab87edc13f4bc631d6eae070a22f92` and inventories every absolute path referenced by the 14 completed source reports, including explicit exclusion decisions.

The archive excludes source video, actions, per-row simulator states, and large reproducible prefix caches. Feature files retain only `X`, optional `X_codec_reconstruction`, `view_index`, and target schema names. Prediction files retain predictions and row identifiers; `y` is removed. Original private files and their reports are unchanged. The manifest maps original hashes to the new sanitized hashes. Checkpoint `y_mean` and `y_scale` remain because they are aggregate fitted model parameters.

The full upstream VideoMAE encoder is not bundled. The archive includes the official model card and its CC-BY-NC-4.0 attribution for the adapted encoder blocks. The pinned model revision, original model hash, and strict loading code identify the encoder needed to load the partial fine-tuning checkpoint. See [the evaluator methods and results](review/video_evaluator.md).

The package cannot by itself reproduce supervised scores because source labels are deliberately absent. Authorized users can reattach the original labels using the recorded source hashes and clip/view ordering. The results remain development selection results: 31 discovery matches and 11 selection matches, with weak velocity prediction and substantial position error. Codec reconstruction calibration does not establish accuracy on generated or intervened video.

Rebuild from the same private artifacts and repository snapshot into new destinations:

```bash
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 .venv/bin/python scripts/package_videomae_artifacts.py \
  --output-dir /data2/your-new-release-directory \
  --report /data2/your-new-release-report.json
```

The script refuses to overwrite a completed archive or report. Tar names are sorted, timestamps and ownership are fixed, and sanitized NPZ entries use fixed ZIP metadata. It never extracts archive paths or invokes a GPU.
