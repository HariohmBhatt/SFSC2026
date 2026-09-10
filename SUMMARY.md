# SFSC2026: problem, evidence, failures, and direction

Snapshot: **September 9, 2026**. This is the handoff for the Kaggle Solar
Filament Segmentation Challenge 2026; linked reports are authoritative.

## Problem and metric

Given a 2048×2048 grayscale H-alpha image, return one complete, non-overlapping
mask per solar filament as COCO RLE. This is **instance segmentation**: touching
filaments must remain separate, while disconnected-looking pieces may be one
filament. There are 180 unlabeled competition images.

The metric is global Panoptic Quality (PQ), with strict mask IoU > 0.5:

`PQ = 2 × sum(matched IoU) / (truth count + prediction count) = SQ × RQ`

Extra fragments increase the denominator; splits and merges near IoU 0.5 erase
matches. High semantic Dice is insufficient: predict the right **object set**,
then improve boundaries. See [EVAL.md](EVAL.md).

## Essential data facts

- 707 labeled images, 1,154 annotation-image records, 8,199 instances; historical
  split 495 train / 106 validation / 106 internal test. Internal test is consumed.
- 296 images have multiple annotation sets and 256 disagree on instance count.
  Train on one coherent annotation record at a time; never union annotators.
- Median instance area is 1,228 pixels; first percentile 209. Full-frame resize
  to 512 can erase thin branches.
- 99/106 validation images are within seven days of training. New claims require
  cross-site date blocks, a seven-day purge, and fourteen-day sensitivity.
- Organizer replies permit supplied `spine`/box/area/category metadata and extra
  **unlabeled** GONG/FITS imagery. Avoid public MAGFiLO labels or weights of
  unknown provenance because they may overlap competition test imagery.

See [DATA.md](DATA.md), [data assessment](reports/data_assessment.md), and
[generalization audit](reports/generalization_gap_analysis.md).

## Kaggle scan

We scanned 58 public notebooks spanning U-Nets, Mask R-CNN/YOLO,
detector→refiner cascades, tiling, ensembles/TTA, topology, and morphology. The
best transferable pattern was **global proposals + prompted high-resolution
refinement + conservative instance selection**. Headline scores were often
stale or unauditable due to private weights, contaminated folds, broken lookup,
empty fallbacks, or replayed predictions. Trust executable lineage and reproduced
current-metric PQ. The dated public maximum was about **0.56**, versus our 0.38;
judging also values qualitative evidence and reproducibility, with a November
15, 2026 report deadline. See the
[public-method audit](reports/competition_strategy_research.md).

## Approaches and failures

| Approach | Result | Conclusion |
|---|---:|---|
| Native tiled ResUNet with foreground/boundary/interior heads | Val **0.28775**; 766 TP / 1,590 FP / 471 FN | Severe fragmentation and FP load; larger U-Net/more watershed was not the fix. |
| Precision-v2: lighter positive loss, sampling and extraction | **0.38117** vs recalibrated **0.37112** | Removed 127 FP but lost 16 TP; +0.010 was borderline. Public ≈**0.32**; later 601-image fit **0.30**. |
| YOLO11l-seg 1280 + prompted ResNet18 refiner 384 | Val **0.43518**, SQ 0.68437, RQ 0.63588; 723 TP / 314 FP / 514 FN; public **0.38** | Current champion; explicit instances work, but validation→public transfer remains weak. |
| Consensus v3: three OOF cascades + learned utility | Val **0.43272**; assessment **0.43237** vs **0.43486**; public **0.37** after override | Added 16 TP and 47 FP; ranking cannot create missing/bad masks, and stacked exclusion was incomplete. |
| Retrospective contextual-quality rerank | Full val **0.43739**; assessment +0.00277 | Historical diagnostic only; no prospective R/Q/H exclusion. |
| Native global/local reconstruction, first run | **0.000**; 1,723 empty masks | AMP-path corruption made trainable tensors non-finite. Kaggle `COMPLETE` is not success. |
| Repaired FP32 native reconstruction | **0.43555**, +0.00037; uncertainty crossed zero | Stable but indistinguishable from control; refinement cannot recover absent proposals. |
| Four native 1280 sliced detector views | **0.39923**, −0.03595 | Recovered 84 TP but added 410 FP: crop covariate shift. |
| Crop score/support probe | Assessment **0.43417**, −0.00069 | Threshold, support, border and novelty policies never beat control. [Results](reports/crop_threshold_probe.json) |
| Learned crop-quality probe | Fell back to **0.43486**, delta 0 | No threshold admitted crop candidates profitably. [Results](reports/crop_quality_probe.json) |
| Matched-scale crop specialist on exact inference views/hard negatives | **Running at snapshot** | Must gain ≥0.015 with positive blocked interval or this detector family is retired. |

Sources: [precision](reports/precision_training.json),
[YOLO/refiner](reports/yolo_refiner_run.json),
[Consensus](reports/consensus_training.json),
[native review](reports/native_approach_deep_review.md),
[sliced recovery](reports/sliced_recovery_run.json), and
[matched scale](filament-matched-scale/APPROACH.md).

## Bottleneck and next move

Of 1,237 truths, the champion misses 514: 256 score rejections, 103 insufficient
masks, 60 ambiguous/merged proposals, 40 absent proposals, 28 refinement damage,
and 27 assembly losses. We need **calibrated recognition under annotation
uncertainty**: recover faint, small, long, low-score objects without a larger FP
wave. Truth-selected candidate recall 0.836 is not attainable PQ because it
ignores conflicts and FP cost. See [error audit](reports/proposal_error_audit.md).

1. Finish matched-scale training while freezing the 0.38 champion. If it fails,
   test a P100-sized **query instance segmenter** at the measured-safe full-frame
   scale with Hungarian matching and foreground/boundary/low-weight centerline
   supervision; see the complete
   [PQ-SetFormer-P100 implementation and training plan](filament-pq-setformer/IMPLEMENTATION_PLAN.md).
2. Train refiners on excluded real detector errors—partial masks, duplicates,
   false proposals, annotator disagreement—and retain seed/global/native choices.
3. Learn separate presence and expected matched-IoU scores on excluded data;
   deduplicate by mask geometry, resolve overlaps, then score the complete set.
4. Use honest R/Q/H lineage: train upstream on R, fit quality on Q-fit, calibrate
   on Q-cal, open date-blocked H once, then replicate. Require ≥0.015 PQ, positive
   block-bootstrap lower bound, and acceptable GPU cost.
5. Treat full-601 refitting as a challenger. Add TTA/ensembles only after one
   model wins independently.

## Non-negotiables

Select by full-frame additive PQ—not Dice, AP, per-image PQ, or oracle metrics—and
report PQ/SQ/RQ, counts, size/length, fragmentation, site/time, runtime and memory.
Use train-side labels only; check hard negatives against all annotations. Every
component and feature generator must exclude assessment images. Verify metric
parity for IoU 0.5, empty/duplicate cases, non-overlap, annotator weighting and
RLE. Hash code, membership, weights, config and outputs.

Compute is one Kaggle P100 16 GB with a nine-hour ceiling. Verified PyTorch is
2.7.1+cu126/torchvision 0.22.1+cu126; the then-default build omitted P100 `sm_60`.
Smoke-test real kernels, persist `/kaggle/working` artifacts, and audit outputs
before accepting success. See [GPU.md](GPU.md).

**Win by improving instance recall and selection together at native detail under
honest temporal exclusion—not by polishing semantic masks or adding candidates
faster than evidence can reject them.**
