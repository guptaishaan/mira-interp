# Independent capture audit and analysis archive

`scripts/aggregate_captures.py` runs on CPU after both complete capture workers
pass. It publishes the analysis NPZ only after checking the entire registered
cohort. It does not fit a probe or read probe performance.

**The v2 aggregate audit passed on 2026-09-07 UTC.** All 424 clips and 13,568 rows
passed in 121.3 seconds. The analysis NPZ is 884,598,487 bytes with SHA-256
`b007c5476951bd064b3b7a30545dbe448e5f9c02eacf598d26cfcc9f72b734f2`.
Its SHA-256 was independently recomputed after completion. The durable evidence
is [`results/capture_aggregate_v2.json`](../results/capture_aggregate_v2.json).
The reserved [pilot row audit](../results/capture_pilot_v2_row_audit.json) also
confirmed that eight deliberately corrupted inputs were rejected, including
rehash-valid files with wrong labels, timestamps, view order and nonfinite X.

The audit checks the approved protocol hash, frozen registration record, exact
qualified clip/split manifests, original failed data audit, and reserved pilot.
It preserves the original cohort's failed status. Every worker must use the same
capture/helper code, trained model/codec hashes, strict-load proof, and pilot.
Each worker must cover its exact disjoint sorted-ID modulo assignment.

For every clip, the auditor rehashes the actual source and capture files, checks
all 17 sites and finite features, and independently joins every saved physical
label, timestamp, player, view and source frame back to the prepared source NPZ.
It rejects missing/duplicate clips and repeated source-view/frame rows. The
reserved pilot is excluded from the analysis archive. Its no-op/repeat controls
and all 17 saved full-reference tensor hashes are verified separately.

The output preserves sorted clip-ID order, then view, then latent time. The
53-match v2 cohort contains 424 clips and 13,568 rows: 256 rows per match.
Arrays `X`, `y`, `codec_X`, `RGB_X`, `sites` and `target_names` follow the
[capture schema](04_capture.md). Per-row metadata also includes `match_ids`,
`split`, `clip_ids`, `view_index`, `latent_frame_index`, `source_frame_index`,
`timestamps`, `player_id`, `canonical_player_ids` and `chunk_index`.

```bash
.venv/bin/python scripts/aggregate_captures.py \
  --protocol configs/observational_probe_v2.json \
  --protocol-sha256 e144ef6c2c5e257114474b9a48e83b11cc4de67a6423924e4b058822339c19d2 \
  --registration-record results/observational_registration_v2.json \
  --data-manifest data/qualified_clip_manifest.json \
  --data-audit results/qualified_data_audit.json \
  --split-manifest data/qualified_split_manifest.json \
  --original-data-audit results/cohort_data_audit.json \
  --pilot-data-manifest data/qualified_pilot_clip_manifest.json \
  --pilot-data-audit results/qualified_pilot_data_audit.json \
  --pilot-report results/capture_pilot_v2.json \
  --capture-reports results/capture_worker0_v2.json results/capture_worker1_v2.json \
  --output /data2/ishaangp/mira-interp/captures/observations_v2.npz \
  --report results/capture_aggregate_v2.json
```

The final report records all provenance hashes, exact row coverage and the
analysis NPZ SHA-256. Its `registration_sha256` means the protocol file hash;
`registration_record_sha256` separately identifies the frozen registration JSON.
An existing output is refused. A temporary NPZ is read back and compared exactly
before being renamed to the final path. An incomplete or failed audit cannot
satisfy the analyzer gate.

"Alignment checked" means the publisher's own-view correspondence plus audited
counts, PTS, identities, calibrated score events and exact label joins. Zero
visual-to-state latency has not been independently established. These captures
support only observational teacher-forced reconstruction decoding.
