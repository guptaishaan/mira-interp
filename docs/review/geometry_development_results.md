# Registered geometry results

The four registered comparisons **do not establish useful nonlinear geometry**
in the selected whole-view descriptor. Quadratic fits make small improvements
for height, vertical velocity and horizontal speed, with paired intervals that
cross zero. Adding the second heading harmonic makes held-value prediction
worse; both heading models also lose to the discovery-mean baseline.

All comparisons use `spatial/block_15_output`, the original top attribution site,
and the same 8,064 development rows. They predict the full1536-dimensional
descriptor from a physical value, fitting on discovery matches outside a fixed
interval and scoring selection matches inside it. Ridge alpha is0.01. Heading
uses world-frame `atan2(vy,vx)` with horizontal speed at least200.

| Variable / held interval | Held rows / matches | Mean baseline MSE | Simple MSE | More complex MSE | Complex-over-simple gain, 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Height[400,800] | 426 /10 | 0.349510 | 0.348739 | 0.348349 | +0.000390[-0.000519,0.001393] |
| Vertical velocity[-250,250] | 1051 /11 | 0.311605 | 0.311551 | 0.310232 | +0.001319[-0.001007,0.003255] |
| Horizontal speed[800,1200] | 419 /10 | 0.335602 | 0.335933 | 0.335305 | +0.000628[-0.000476,0.001577] |
| Heading[-pi/8,pi/8] | 188 /8 | 0.356401 | 0.360722 | 0.367722 | **-0.007001[-0.009911,-0.003093]** |

Simple/complex means affine/quadratic for scalar variables and first/second
harmonic for heading. MSE averages original descriptor-coordinate squared
errors, then weights matches equally. Positive gain favors the more complex
fit. Intervals use1,000 paired whole-match resamples; they are descriptive
development intervals, without a family-wide inference claim.

The exporter reproduced all eight map fingerprints and held-match errors. The
separate [geometry audit](../../results/geometry_development_v1/geometry_audit.json)
recomputed discovery normalization, checked saved coefficients against ridge
normal equations, recomputed held errors and bootstrap intervals, and verified
all displayed conditional means. The maximum relative normal-equation residual
was7.48e-13. No coefficients were refitted by that audit. Its SHA256 is
`3baffdf98dd634647e5c0a1622e96695a574c7bb8964c287d9aaf858809669b1`.

Each variable's folder under `results/geometry_development_v1/` contains the
analysis report, reproduced `geometry_maps.npz`, export audit, and PNG/PDF figure.
PCA displays discovery conditional means, including discovery rows in the held
interval; those rows remain excluded from the forward fits. The display has no
role in the measured raw-descriptor errors or model selection.

Camera, actions, position and other state vary with the physical values. The
whole-view descriptor is not a localized ball feature, and ball visibility was
not established. These measurements therefore characterize conditional
observational variation. Bent PCA curves cannot supply the missing evidence for
a physical manifold, causality, or reliable geometry-aware trajectory steering.
The completed software/measurement gate permits the separately reviewed sparse
experiments; it does not imply a positive geometry result.
