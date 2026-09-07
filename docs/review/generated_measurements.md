# Frozen measurements of generated steering

`scripts/evaluate_rollouts.py` applies the previously selected VideoMAE step500
checkpoint without any adaptation. Its configuration is registered before the
generated pilot. Each phase requires an independent generation audit; selection
also requires the measured pilot and its audit, and confirmation requires audited
selection measurements. No generated outcome changes a path, dose, or model.

The four16-frame windows end at frames17,19,21 and23, corresponding to0.1,0.2,
0.3 and0.4 seconds after the observed context. All four views and all12 role
coordinates are saved. The primary outcome is view0 estimated absolute ball
height: estimated ego height plus estimated ball-minus-ego height. The all-view
mean is secondary. Each edit is compared with its explicit same-match, same-clip,
same-seed baseline.

All four windows still contain the first edited generated frames16–17. A shift
at a later measured endpoint can therefore reflect those shared early pixels;
four shifted estimates do not independently prove persistent later-frame
physical motion. Saved later-frame pixels and latents provide separate evidence
of numerical rollout changes, not simulator-validated state.

The five fixed requested doses include a shared zero baseline. Reports retain
actual clipped doses, estimate dose-response slopes and rank correlations, and
compare the+600 and−600 conditions. A constant response does not count as steering.
Contradictory responses at identical effective doses cannot count as monotonic.
Undefined correlations remain missing, with counts reported; finite seeds are
averaged within each match before equal-match averaging.

Two seeds are averaged within each match before1000 match-bootstrap draws.
Paired contrast differences against both the norm-matched random direction and
wrong-variable direction use those same matches. The random/wrong-variable norms
match the quadratic path, as specified by the generation protocol. Nuisance
changes are reported in every original coordinate and as an11-coordinate RMS
using fixed discovery target scales, excluding only ball-minus-ego height. Ego
height remains a nuisance coordinate and is not silently removed.

These are learned video measurements. Recorded and codec-reconstructed
development clips established an absolute ball-height MAE of approximately264
and265 simulator units, respectively. Those errors are not uncertainty intervals
for edited generated videos or thresholds for detecting paired changes. Paired
errors may cancel, or may acquire edit-dependent bias. Match-bootstrap intervals do not include evaluator
bias or simultaneous testing uncertainty. Changed pixels, monotonic estimates,
or successful execution alone do not establish accurate physical control.

The review's synthetic checks include increasing, decreasing, constant and
clipped dose responses, contradictory duplicate doses, missing ranks, exact
view/time/target indexing, nuisance-coordinate specificity and matched controls.
The first two failed checks are preserved; both were repaired before registration
or generated outcomes.
