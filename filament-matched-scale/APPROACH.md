# Matched-scale hard-negative detector

## Decision

Retire inference-only slicing. Keep native-scale views and train the detector on
the exact views it will see at inference.

The sliced diagnostic recovered 84 additional validation true positives,
including nine tiny instances, but added 410 false positives. PQ fell from
0.43518 to 0.39923; the assessment delta was -0.03531 with a 95% interval of
[-0.04872, -0.02178]. Every individual view reduced PQ. Score thresholds,
cross-view support, border rejection, novelty filtering, and a regularized
quality model all failed to produce a positive assessment delta. The learned
gate selected no crop candidates.

This identifies covariate shift as the actionable defect. The original detector
was trained on complete 2048 images resized to 1280, then asked to interpret
native 1280 crops containing larger structures, partial instances, and different
background statistics. Its confidence is not calibrated for that input policy.

## Pilot

Start from the verified whole-image detector that produced internal PQ 0.43518.
Fine-tune it on 495 training images using five matched views per image:

- one complete 2048 image, resized by the detector to 1280;
- four overlapping native 1280 corner views;
- every visible annotated instance with at least 64 pixels, clipped exactly as
  it will appear during crop inference;
- empty and sparse crop views retained as genuine detector negatives.

The 2,475-view dataset is trained for eight fixed epochs with YOLO11l-seg,
batch two, AdamW at 2e-4, no mosaic, modest geometry and brightness augmentation,
and the verified P100-compatible PyTorch build. This is approximately the same
optimizer-batch budget as the original 36-epoch whole-image detector.

At validation, run the trained model on the whole image and the same four crop
views. Map crop masks to original coordinates, refine them with the frozen
verified refiner, preserve whole-image instances first, and add nonduplicate crop
instances afterward. Choose between whole-only and multiscale inference plus one
acceptance threshold on the frozen 24-image calibration subset. Report the
remaining 82 images once with the exact control candidates and metric.

## Issues addressed

| Observed failure | Matched-scale response |
|---|---|
| 256 score rejections | Relearn confidence at native object scale instead of applying a whole-image threshold to shifted crops. |
| 100 absent or merged proposals | Larger native-scale objects and overlapping crop views improve proposal recall. |
| 131 mask/refinement failures | Retrain the segmentation head on native masks and retain the proven refiner. |
| 27 assembly or identity losses | Preserve whole instances before additive crop masks and suppress cross-view duplicates by mask IoU. |
| 410 sliced false positives | Train all crop backgrounds as negatives; calibrate on predictions from the same view distribution. |

## Evidence and gates

The first run is a historical screening pilot because earlier experiments have
examined the 24/82 validation set. It cannot authorize a submission. It proceeds
only if artifact hashes, membership, GPU identity, finite loss, checkpoint
change, candidate coverage, and exact control-PQ checks pass.

The screening signal is useful only if assessment PQ improves by at least 0.015
and the paired 60-day block-bootstrap lower bound is positive. A pass triggers a
fresh time-blocked replicate with training R, quality-fit Q, threshold-calibration
Q-cal, and blocked assessment H. The replicate must be positive and the pooled
interval must remain above zero before full 601-image fitting and test inference.

A failed screening result retires this detector family. The next architecture is
a query-based instance segmenter with Hungarian matching and explicit foreground,
boundary, and centerline losses. That removes detector NMS and independent
proposal-score rejection, but costs substantially more implementation and P100
time, so it follows only after matched-scale training is measured.

## Runtime and artifacts

Expected P100 time is four to seven hours: dataset materialization, roughly
9,900 optimizer batches, five-view inference on 106 images, metric evaluation,
and artifact export. The controller enforces Kaggle's nine-hour ceiling and
persists the trained detector, validation candidate cache, report, source hash,
membership hash, and exact configuration. It never submits automatically.
