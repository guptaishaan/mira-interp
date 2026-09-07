# Geometry development input

The exporter uses the original first attribution-ranked site,
`spatial/block_15_output`, after the exact causal selection audit passed. This
choice preserves the discovery ranking. It does not choose a new site from the
fresh confirmation results or the exact-intervention effect sizes.

All336 discovery/selection captures contribute view-major latent indices2..7:
31 discovery matches give5,952 rows and11 selection matches give2,112, totaling
8,064 rows. `X[8064,1536]` contains the unchanged FP16 spatial descriptor: twelve
3x4 bins projected to128 channels each. No further pooling, normalization, fitted
projection, or matched-mean descriptor is introduced during export.

`position[8064,3]` and `velocity[8064,3]` are the global ball XYZ source labels,
in world coordinates, copied from columns0..5 of each row's own-view-aligned
source state. Metadata preserves match, split, clip, view, canonical player IDs,
source frame and timestamp. `frame_index` is the consecutive latent index2..7,
while `source_frame_index` separately records the final video frame of its pair.
Adjacent latent rows within a clip and view are0.1seconds apart.

The descriptor summarizes the **whole camera view**. `entity_ids='ball'`
identifies the annotation target; it does not identify localized ball tokens or
establish visibility. The export explicitly sets `entity_mapping_verified=false`,
`descriptor_entity_localized=false`, and `ball_visibility_verified=false`.
Only same-match/clip/view temporal correspondence is verified. A temporal sparse
extension can test view-level support consistency with this data; it cannot be
described as verified tracked-ball feature recovery.

The exporter refuses confirmation before reading NPZ arrays. It binds the
original protocol and split, passed capture/probe audits, original site ranking,
and exact causal audit by hash. Every label, frame, timestamp and player identity
is rejoined to the original source NPZ. Exported arrays are read back and compared
with their exact source fingerprints. Original causal evidence concerns an
internal frozen output proxy, with no independently measured physical-control
claim. The full donor result is heterogeneous and the component effect is small;
an engineering PASS does not remove those scientific limitations.
The exact interventions tested view0 at latent7. Exporting the same site's
features from all views and latent2..7 does not establish causal validation at
every exported view or time; the geometry comparison remains observational.

The input audit also reports discovery-only support and row/match counts for
four intervals supplied before export: height[400,800], vertical velocity
[-250,250], horizontal speed[800,1200], and heading[-pi/8,pi/8] at horizontal
speed at least200. Heading uses `atan2(vy,vx)` in[-pi,pi). These are descriptive
support checks with inclusive bounds, not fitted geometry results; selection
support is not used to choose the intervals.

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -u \
  scripts/prepare_geometry_development.py --site block_15_output
```

Outputs are the local source-label-bearing file
`/data2/ishaangp/mira-interp/features/geometry_development_v1.npz`, plus
`results/geometry_development_v1/selected_probe.json` and `input_audit.json`.
The NPZ must not be included in public derived-only archives without removing
its position/velocity labels. Export success uses `passed_feature_input_export`;
it does not write the separate reviewed `passed_feature_development_prerequisites`
gate or run any geometry or sparse analysis.
