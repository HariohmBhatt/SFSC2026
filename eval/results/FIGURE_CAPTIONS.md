# Suggested report captions

**morphology_metrics.** Native-resolution OOF comparison of frozen V4 detector
evidence and rescored V3; human-to-human results use all directed annotator pairs
on the multi-record subset. Bars/bands are 95% date-clustered bootstrap intervals
(2,000 draws). Spine coverage includes zero for unmatched GT objects. IoU and
skeleton/spine length ratios condition on a strict IoU >0.5 match; unmatched
counts accompany the IoU distribution. Skeleton length includes branches.

**morphology_strata.** GT-spine coverage at 3 native pixels, including zero for
missed objects, stratified by projected width, centroid radius, other-record
support, and annotator count. Width is GT area / supplied spine length; radius
uses the frozen nominal preprocessing disk. Support is the fraction of other
records with an IoU >0.5 counterpart. Points show object-macro means and 95%
date-clustered intervals. Support is used for evaluation only.

**oof_examples_crops.** Deterministic poor/median/good OOF cases chosen nearest
the 10th/50th/90th percentiles of V4 image PQ, with filename tie-breaks. Each row
shows separate annotator polygons (cyan), supplied spines (yellow), V4 masks
(orange), and rescored V3 masks (blue) on the same H-alpha crop. Crops centre on
the largest first-record GT object; they are fixed independently of predictions.
The row PQ is the full-image score. Native-pixel scale bars are shown.

**oof_examples_full.** Full-disk context for the same frozen percentile-selected
images and annotations as the crop panel. Masks are unmodified native-resolution
instances. Missing annotator columns indicate absent records, not empty labels.
These examples illustrate the score range; they are not random population samples.

**oof_multi_annotator_crops.** Additional qualitative reference using the same
V4 image-PQ percentile rule **within the 296 multi-record images**. Separate
columns preserve every annotator record rather than merging their outlines.
This explicitly restricted selection illustrates label variation; its scores
and image IDs are recorded in `multi_annotator_examples_manifest.json`.

For a four-page report, prioritize `morphology_metrics.pdf` and
either `oof_examples_crops.pdf` or `oof_multi_annotator_crops.pdf`, retaining the other panels as
reproducibility/supplementary evidence. PDFs retain vector text and mask contours;
PNG copies are for preview.
