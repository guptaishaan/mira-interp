# Complete generated-output publication

`scripts/package_generated_outputs.py` packages every condition of one completed phase. It requires the phase's successful generation audit and independent frozen-video measurement audit, bound to the exact manifests. It checks the full registered coverage: two reserved pilot conditions, 420 selection conditions, or 924 confirmation conditions. It does not select examples based on visual quality or measured response.

Each public NPZ contains exactly six arrays. `frames` is the original `frames[:,16:]`, with shape `[4,8,3,288,512]`; `latents` is the original `latents[:,8:]`, with shape `[4,4,9,16,32]`. Both remain FP32 with every bit preserved. The native and edited residual tiles and descriptors are copied in full. Observed video context, action arrays, simulator-label arrays, and model checkpoints are excluded. Source file hashes and both original/exported array fingerprints document the exact slicing.

The package includes the completed generation/measurement manifests and audits, registrations, complete predictions and per-record prediction caches, each condition's metadata, the exact registered experiment-code files, model/source revision and checkpoint hashes, `THIRD_PARTY.md`, model attribution, and upstream license material. Original paths inside metadata identify private evidence; the corresponding source dataset files are not bundled. Predictions are learned video estimates. Their 16-frame inference windows included decoded context, so reproducing the exact predictor forward pass still requires authorized context inputs. The public predictions and complete condition grid support reproducing the reported dose summaries.

Public NPZs use lossless DEFLATE level 1. Deterministic USTAR parts use sorted names, fixed ownership/mode, and zero timestamps. Part planning includes each actual encoded file's bytes, 512-byte headers/padding, the per-part manifest, and final tar alignment. Every completed tar is at most 1,800,000,000 bytes. The script reopens every member and rechecks all hashes; it also reloads each public NPZ and compares every exported array fingerprint to its verified source slice. It never extracts untrusted archive paths.

Each part has a `PART_MANIFEST.json` and a sidecar `.tar.json` report. The external `archive_manifest.json` lists all parts, members, and records. The completed phase report contains:

```json
{
  "status": "passed_generated_artifact_package",
  "phase": "selection",
  "packaged_records": 420,
  "parts": [{
    "status": "passed_generated_artifact_part",
    "completed": true,
    "readback_verified": true,
    "filename": "mira-generated-v2-selection-part0001.tar",
    "path": "/data2/.../mira-generated-v2-selection-part0001.tar",
    "bytes": 0,
    "sha256": "actual-file-sha256",
    "manifest_sha256": "embedded-part-manifest-sha256",
    "member_count": 0,
    "record_count": 0,
    "report": {"path": "/data2/.../part0001.tar.json", "sha256": "sidecar-sha256"}
  }],
  "archive_manifest": {"path": "/data2/.../archive_manifest.json", "sha256": "manifest-sha256"}
}
```

This is a schema illustration; zero sizes/counts are placeholders, not measurements. The uploader should verify each part's SHA256 and size against these reports, then upload the parts, their sidecar reports, and the external archive manifest. Packaging does not itself commit, push, create a release, or use a GPU.

After the selected phase and both audits complete:

```bash
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  scripts/package_generated_outputs.py --phase selection
```

Use `pilot` or `confirmation` for the other phases. Defaults write archives below `/data2/ishaangp/mira-interp/publication/generated_v2_complete/{phase}/` and the completed report to `results/generated_publication_{phase}_complete.json`. Existing destinations are immutable. The script maintains a 25 GiB disk reserve and checks space for staging and archives before writing. Staged public NPZs are retained for reproducible rebuilds.

CPU tests cover future-only slicing, rejection of additional source arrays, exact padded part sizes, deterministic byte identity, corruption detection, and unsafe member paths. The final reviewed pilot packages both conditions in one 32,051,200-byte archive with 44 members, including attribution and complete code provenance. A second full build in a different directory produced the identical SHA256 `96cb76ce2e1fa525d98e4a05baa198a09fe6ca42703f6fa5d7955b046c88ad8e`. The authoritative report is `results/generated_publication_pilot_reviewed.json`; the rebuild proof is `results/generated_publication_reviewed_selftest.json`. Earlier pilot packages are preserved as preliminary versions.

`scripts/watch_generated_publication.py --phase selection` waits on the existing supervisor's completed measurement-audit stage, verifies the recorded audit hash, and then invokes the reviewed packager once on CPU. It snapshots the packager/helper/review hashes, records progress in `results/generated_publication_waiter_v2/selection.json`, and stops on changed bindings, an earlier study failure, or its six-hour budget. The accepted audit hash remains fixed before child execution and through completion. It can package an already completed selection phase even if a later phase fails. Its child cleanup targets only its own process group. A waiting watcher is not a completed package, and the watcher performs no publication. The original waiters were stopped before either packaged anything; their code/state/logs are preserved under `results/publication_waiter_attempts/01_accepted_audit_hash/`.
