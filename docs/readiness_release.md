# Pretrained readiness snapshot

This release contains the executable preparation, source/weight pins, numerical
hook tests, and measured pretrained readiness outputs for Alakazam MIRA Mini 4P.
It is an independent 1B reproduction, not the original 5B MIRA.

All 936 codec and 1,183 world-model state tensors load strictly. Block-0 input and
all 16 residual outputs are captured. Observation-only hooks and repeated fresh
forwards have zero numerical difference. The generated continuation remains
bitwise identical when hidden future input frames change from zero to 255 with
the same actions and seed. Peak GPU allocation is 4.65 GiB for this short test.

The archive contains all 19 saved numerical outputs: 17 residual tensors,
both generated frames from all four views, and the generated latent. It also
contains their manifest, the complete report and third-party attribution.
Archive and per-file hashes are in `results/readiness_archive.json`.

The context is a public example without physical labels or match identity.
These outputs establish engineering readiness only. They do not establish
physical-state decoding, causality, geometry, sparse features or steering.
Rocket League content is copyright Psyonix LLC / Epic Games; model/example
derivatives use CC BY-NC-SA 4.0 with attribution to Alakazam and MIRA.
