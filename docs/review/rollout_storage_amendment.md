# Faster lossless storage before research rollouts

The corrected v1 generation pilot and frozen video-evaluation pilot both passed
their independent audits. They remain unchanged. The pilot measured roughly5–7
seconds of model inference per rollout, while default NPZ compression took about8
seconds per output in a separate four-trial benchmark. This would spend more time
compressing arrays than running the model.

The benchmark reloaded every array exactly. DEFLATE level1 took a median1.446
seconds to write45.249MB, versus8.082 seconds and36.230MB with default compression.
Uncompressed files were faster but required173.127MB each. The chosen level1
format projects60.8GB for1,344 outputs, with approximately74 minutes less total
two-worker writing time than the default. These are pilot-based estimates.

`scripts/rollout_steering_v2.py` preserves the successful v1 producer and changes
only storage, output directories, and provenance checks. `array_storage.py` uses
NumPy's standard array format inside a ZIP archive with explicit compression
level1. There is no quantization, downsampling or dtype conversion. Model
arithmetic, context, action values/timing, seeds, intervention sites, paths,
physical doses and all research matches remain identical.

The separate v2 registration precedes another reserved five-inference pilot.
Its outputs must pass the same generation audit and exactly match every v1 pilot
array, not only the rendered images. The already frozen measurement protocol is
reused with v2 generation/output/cache directories, and its pilot is audited
again. Main selection then runs420 rollouts across two disjoint match workers;
confirmation uses924, after the required selection audits. No research result
was available when this storage decision was made.

Only physical GPUs6 and7 are authorized. The25GiB free-disk reserve remains
enforced. Large arrays stay under `/data2/ishaangp/mira-interp`; public derived
archives exclude observed context and are split below GitHub's asset-size limit.
