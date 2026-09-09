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

This identifies crop covariate shift as one actionable defect. The original detector
was trained on complete 2048 images resized to 1280, then asked to interpret
native 1280 crops containing larger structures, partial instances, and different
background statistics. Its confidence is not calibrated for that input policy.

## Pilot

Start from the verified whole-image detector that produced internal PQ 0.43518.
Preserve its candidates as the immutable control and fine-tune a separate crop
specialist on 495 training images using four matched views per image:

- four overlapping native 1280 corner views;
- every polygonizable visible component with at least four pixels, clipped exactly as
  it will appear during crop inference;
- empty and sparse crop views retained as genuine detector negatives.

The 1,980-view dataset is trained for eight fixed epochs with YOLO11l-seg,
batch two, AdamW at 2e-4, no mosaic, modest geometry and brightness augmentation,
lossless PNG crops, and the verified P100-compatible PyTorch build. Its roughly
7,920 optimizer batches are close to the original detector's 8,910 batches.

At validation, run the specialist on the same four crop views. Map crop masks to
original coordinates and refine them with the frozen verified refiner. The
calibration comparison always starts from the exact cached control masks. It may
add new crop instances, require cross-view support, or use crop agreement to
rescue existing whole-image candidates whose combined score was below 0.30.
Select one policy and its independent crop threshold, including 0.03–0.15, on
the frozen 24-image calibration subset. Report the remaining 82 images once.

## Issues addressed

| Observed failure | Matched-scale response |
|---|---|
| 256 score rejections | Retain low-score whole candidates and rescue them only when an independently trained crop specialist confirms the same mask. |
| 100 absent or merged proposals | Larger native-scale objects and overlapping crop views improve proposal recall. |
| 131 mask/refinement failures | Retrain the segmentation head on native masks and retain the proven refiner. |
| 27 assembly or identity losses | Preserve exact whole instances; cluster same-view and cross-view crop proposals before painting. This reduces duplicate fragments but does not claim to repair every immutable-control assembly loss. |
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
7,920 optimizer batches, four-view inference on 106 images, metric evaluation,
and artifact export. The controller enforces Kaggle's nine-hour ceiling and
persists the trained detector, validation candidate cache, report, source hash,
membership hash, and exact configuration. It never submits automatically.
