# Initial progress

The first question is whether physical state is readable from the model, and
whether that finding holds on new matches. The current evidence supports position
decoding. It does not yet establish reliable physical control.

## Setup

- Model: Alakazam's MIRA Mini 4P, an independent reproduction of MIRA. The original
  5B MIRA weights were not available in the checked release channels.
- Data: synchronized Rocket Science video, actions, and simulator state. Match
  identities stay separate across fitting, selection, and confirmation.
- Readouts: regularized linear probes at all 17 residual sites, compared with
  codec and RGB baselines. Features and targets are normalized using fitting data.

Video and simulator clocks are joined explicitly. Matches that fail the fixed
alignment or quality rules are excluded without replacement. The checkpoint is
loaded strictly, codec pairing is checked, and inference follows the released
precision and preprocessing conventions.

## First result: position is readable

The development comparison used 31 fitting matches and 11 selection matches.
It compared pooled and spatial readouts, absolute coordinates, and coordinates
for the viewing player and ball relative to that player. The layer and ridge
penalty were fixed before evaluating new matches.

The spatial readout uses a fixed projection within a 3×4 grid. Its comparison
with pooling tests these particular readouts; it does not rule out stronger
spatial or object-localized methods. Twenty-four new match candidates were
reserved using metadata, and 23 passed the unchanged data checks, with no
replacement.

On 23 new matches, the selected position probe reduced standardized error by
80.9% relative to the fitting-set mean predictor. Player XYZ position R² was
0.660 / 0.900 / 0.974; ball-relative XYZ was 0.520 / 0.694 / 0.926. Horizontal
velocity remained weak. Keeping that result visible matters: the readout works
better for position than for motion.

The evaluation covers 184 clips and 4,416 scored view/time rows. Uncertainty
resamples whole matches rather than treating neighboring video frames as
independent examples. Independent checks reproduced the saved predictions and
reported metrics. These intervals describe match variation; they do not cover
all model-selection or training uncertainty.

The two [progress figures](../../figures/progress/) show the complete layer
comparison for position and the fixed position/velocity readouts on new matches.
Their source tables and hashes are in [results/progress](../../results/progress/).
The [full confirmation report](../11_fresh_confirmation_results.md) includes
the other fixed readouts and exact definitions.

## What this establishes

This is a useful first step: the data join, model capture, and held-out readout
work well enough to study position. The probes see activations from observed
target clips. They do not measure the model's ability to predict an unseen
future, identify a unique physical feature, or prove that a decoded direction
controls generated behavior.

The next question is causal use. Initial exact patching tests found small
probe-direction effects and inconsistent full-donor effects across matches.
Geometry and sparse-feature experiments have not established a reliable causal
coordinate. The registered rollout study continues separately, with its settings
held fixed. Its engineering completion will not turn those open questions into
positive findings.

See the [diagnosis and repairs](../13_diagnosis_and_repairs.md) for what improved,
what did not, and the evidence behind each change. The longer
[experiment index](../README.md) keeps exploratory work separate from this
initial progress summary.
