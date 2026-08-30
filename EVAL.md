# Evaluation

## Authoritative Objective

The active Kaggle metric is **Panoptic Quality (PQ)** and it is maximized. An
older submission-format passage mentions mean Dice, but the updated competition
evaluation page, live metric configuration, and organizer self-evaluation
notebook use PQ. Recheck Kaggle before final submissions in case the organizers
change the evaluator.

For one-to-one matched ground-truth and predicted instances:

\[
PQ = \frac{\sum_{(y, \hat{y}) \in TP} IoU(y, \hat{y})}
{|TP| + 0.5|FP| + 0.5|FN|}
\]

A pair matches only when `IoU > 0.5`; equality is not a match. PQ combines
segmentation quality with detection quality, so false fragments and merged
filaments are penalized even when a union-like semantic mask looks reasonable.

The implementation in `src/sfsc/evaluation.py` performs maximum-IoU one-to-one
matching and exposes additive totals. Dataset PQ is computed from global totals,
not as an unweighted mean of image PQ values.

## Fixed Evaluation Process

1. Train only on manifest rows marked `train`.
2. Select checkpoints and all post-processing parameters only on `validation`.
3. For each annotation image-record, produce non-overlapping filament instances
   at the original 2048 x 2048 resolution.
4. Compute one-to-one matches at strict `IoU > 0.5`.
5. Sum matched IoU, TP, FP, and FN over all records before computing PQ.
6. Freeze the complete pipeline, then evaluate the `test` split once.
7. Log the split-manifest hash, commit, seed, GPU, runtime, and peak VRAM.

Because multiple annotation sets can point to one JPEG, inference may be cached
by filename, but that prediction must be scored independently against every
annotation set assigned to that split.

## Required Diagnostics

Report at least:

- global PQ, matched-IoU sum, TP, FP, and FN;
- recognition quality `TP / (TP + 0.5 FP + 0.5 FN)`;
- segmentation quality `matched IoU sum / TP`;
- matched IoU and Dice distributions;
- ground-truth versus predicted instance counts;
- fragmentation (one truth overlaps many predictions);
- merging (one prediction overlaps many truths);
- PQ by site, year, object-size band, and annotation multiplicity;
- inference time and peak allocated/reserved VRAM.

Do not select a model by Dice alone. A semantic baseline can have strong Dice
while merging nearby filaments or breaking one filament into tile fragments.

## Submission Contract

The submission CSV has one row per predicted instance:

```text
filament_id,segmentation_rle
20150125172714Mh_1,<compressed COCO RLE counts>
```

Preserve the exact test image ID and append a unique suffix. Encode each binary
mask as Fortran-contiguous `uint8` using `pycocotools`; submit the compressed
ASCII `counts` string without the `size` field or manual quoting. Round-trip
decode every RLE and enforce non-empty, in-bounds, non-overlapping instances.

## Sources

- [Official evaluation page](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/evaluation)
- [Official data page](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
- [Organizer self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- [Official competition rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules)
