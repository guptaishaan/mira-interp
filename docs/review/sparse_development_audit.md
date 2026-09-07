# Independent sparse dictionary audit

The registered experiment compares 30 final checkpoints: three seeds, active-coordinate budgets 32/64, and ReLU Top-K, signed Top-K, block Top-K, temporal block, and shuffled-time temporal block variants. Each has 3,072 latent coordinates, 1,536 input coordinates, 9,440,256 trainable parameters, and 1,000 updates. Decoder rows have unit norm. The input remains a whole-view descriptor; temporal pairs establish the same match, clip, and camera view, not a localized ball track.

Run the audit after the training suite passes:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/audit_sparse_development.py
```

`scripts/audit_sparse_development.py` reads the registered development input and safely loads every checkpoint on CPU with `weights_only=True`. It checks exact state keys, shapes, FP32 parameters, finite values, source/report hashes, training budgets, and the exact independently recomputed discovery mean/global RMS. An independent NumPy implementation performs affine encoding, selection by scalar magnitude or block norm, and linear synthesis. It recomputes equal-match reconstruction losses, each match's raw loss, activity, dead coordinates/blocks, and true-adjacent support Jaccard. The fixed numerical agreement criterion is relative tolerance `1e-4` plus absolute tolerance `1e-6`, allowing FP32 CPU/GPU arithmetic and top-group ranking differences. This is an engineering agreement tolerance, not a research success threshold. Signed, ReLU, and grouped selection/synthesis were checked on small hand-computed examples before the real audit.

The audit then measures retention for every dictionary without fitting or selecting new probes or maps:

- The previously frozen 15-position probe is applied to native and reconstructed descriptors. Prediction drift is normalized by its discovery target scales. Ball XYZ errors use the already verified own-view labels. These are observational probe measurements, not generated-video or physical-control results.
- The original four geometry comparisons retain their exact discovery bins, eligible rows, physical normalizers, and held intervals. The audit recomputes original bin means and compares reconstructed means in all 1,536 coordinates. It reports centered variation, centered Gram matrices, and pairwise-distance changes, alongside fixed-PCA coordinates for display.
- Both frozen physical-to-descriptor maps per variable are evaluated against reconstructed selection rows. **A lower map error after reconstruction can result from losing variation.** It does not establish retention by itself; native errors and raw reconstruction/conditional-variation changes accompany it. No new geometry is fitted.

The immutable output `results/feature_development_v1/sparse_audit.json` binds all three feature reports, all 30 checkpoints, source/probe/causal gates, and geometry outputs. `retention_arrays.npz` stores derived conditional means, display projections, original aggregate physical-bin coordinates, and native probe predictions; it contains no individual source state labels or video. A passed audit establishes reproducible computation, not successful feature recovery. Causal feature fidelity is a subsequent experiment and must also distinguish preserved native residual complements from effects carried by reconstructed descriptors.

## Completed results

The audit passed all 30 checkpoints in 38.082 seconds. The largest saved-versus-recomputed metric difference was `6.52e-6`, within the numerical criterion. Exact discovery normalizers and all checkpoint schemas passed. The main report SHA256 is `8f0c7fd1ce3c02503dff4f214c86da7be30b5022897a850a1e0c58604a5b2e54`.

The following values average the three seeds. Selection uses 11 held-out matches; these are development comparisons, with no dictionary chosen for a new confirmation test.

| Variant | Selection normalized reconstruction MSE, 32 / 64 active | Selection adjacent-support Jaccard, 32 / 64 active |
|---|---:|---:|
| ReLU Top-K | 0.6413 / 0.6027 | 0.2717 / 0.2456 |
| Signed Top-K | 0.6479 / 0.6068 | 0.2319 / 0.1979 |
| Block Top-K | 0.7158 / 0.6874 | 0.4124 / 0.3338 |
| Temporal block | 0.7080 / 0.6787 | 0.4343 / 0.3562 |
| Shuffled-time temporal block | 0.7036 / 0.6765 | 0.4466 / 0.3574 |

All variants used their requested active scalar budget on these inputs. ReLU has the lowest held-match reconstruction error at both budgets. Block selection has greater support overlap, but its groups and support universe differ from scalar Top-K; those absolute overlaps alone are not a matched temporal-effect comparison. Within the block family, the shuffled-time control matches or slightly exceeds the true-adjacent temporal variant in these averages. This does not establish a benefit from correct temporal correspondence.

Reconstruction also changes the frozen ball-height probe: selection prediction-drift MSE ranges from approximately 0.129 to 0.186 after normalization by the original probe's discovery target scale. Height-bin centered variation retains roughly 86–91% of its native value across variant/budget averages. Neither retained variation nor reduced physical-map error establishes a physical manifold, especially since the preceding geometry comparisons did not establish a nonlinear advantage.

The [six-panel figure](../../results/feature_development_v1/sparse_comparison.png) and [PDF](../../results/feature_development_v1/sparse_comparison.pdf) show all variants and budgets. Whiskers are the range of three seeds, not confidence intervals. `plot_audit.json` records every displayed value and both figure hashes. Figures were visually inspected for readable labels and clipping.
