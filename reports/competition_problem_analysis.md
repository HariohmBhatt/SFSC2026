# Solar Filament Segmentation 2026: problem analysis

Analysis date: 2026-09-06. Sources are the repository's measured data audit,
completed experiment artifacts, refreshed authenticated Kaggle dashboard, and
the companion [primary-source research](competition_strategy_research.md).
The first intervention is a hypothesis under test, not a winning result.

## What has to be predicted

Each input is a 2048×2048 grayscale H-alpha solar observation. The output is a
set of individual filament masks at original resolution, encoded as compressed
COCO RLE. A good union of foreground pixels is insufficient: one filament split
into several predictions, two filaments merged together, a missed filament, or
an unannotated dark feature can all damage instance recognition.

The local data audit found 707 unique annotated JPEGs, 180 competition-test
JPEGs, 1154 annotation-image records and 8199 instance annotations. Median
instance area is 1228 pixels, with 187 instances below 256 pixels. Median box
width/height are 74.5/69.6 pixels, with much longer examples. Resizing an entire
frame to 512 pixels reduces area by 16: the median mask becomes about 77 pixels,
and its narrow branches can disappear. Native crops preserve those branches;
global context and consistent probability stitching are still needed to avoid
tile seams and disjoint predictions for long structures. Source: [DATA.md](../DATA.md).

## The objective creates a recognition bottleneck

Our fixed development metric uses maximum-IoU one-to-one matching with strict
IoU > 0.5, evaluated against each annotation record independently. Sufficient
statistics are aggregated globally:

`PQ = matched_IoU / (TP + 0.5 FP + 0.5 FN) = SQ × RQ`.

Since the denominator also equals `(number_of_truths + number_of_predictions)/2`,
extra instances have a direct cost. More foreground recall is useful only when
it produces coherent instances with adequate IoU. Tiny false predictions are
not cheap simply because their pixel areas are small. The official shared
notebook's pair-counting behavior differs for overlapping masks; its current
live implementation is not fully public. Keep the conservative one-to-one
development evaluator and audit ordinary predictions for parity, not duplicate
predictions or empty-submission loopholes. Sources: [EVAL.md](../EVAL.md),
[official self-evaluation](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).

The first model's actual completed result supersedes the stale 'pending'
description in the original launch report:

| Quantity | First native ResUNet, validation |
|---|---:|
| Unique images / annotation records | 106 / 169 |
| PQ / SQ / RQ | 0.287749 / 0.674858 / 0.426385 |
| TP / FP / FN | 766 / 1590 / 471 |
| FP below 1024 pixels | 1271 (79.9% of all FP) |
| Fragmented truth / merged prediction diagnostics | 322 / 9 |
| Selected epoch | 60 |
| Peak TensorFlow allocator | 11.51 GB |
| Whole evaluation job | 6268 seconds |

Source: `outputs/first-model/experiment/report.json`, its training log and recipe;
the same report was accepted under `.sfsc/runs/cycle-001-evaluate-eaae4548/`.
Fragmentation and merging are the existing diagnostic overlap counts, summed
over annotation records; these are not separate mutually exclusive error totals.

The high-level priority follows from those counts: reduce unsupported detections
and avoid over-splitting while maintaining the faint ends of real filaments.
Boundary accuracy remains material, especially for small masks near the 0.5
match cutoff. The old extraction search ended at minimum area 64, far below
most ground-truth objects, and allowed four-pixel interior seeds to trigger
watershed splits. Both warrant controlled comparisons.

An instructive upper bound: removing **every** current FP while leaving matches
unchanged would raise PQ only to about 0.516. Removing just the 1271 tiny FP
with no other changes would give about 0.445. These are counterfactual arithmetic,
not obtainable or measured scores. Precision cleanup alone cannot establish a
leaderboard-leading system: recognition recall and/or mask IoU must improve too.
For example SQ 0.75 × RQ 0.75 = PQ 0.5625, but reaching those values is unproven.

## Annotator uncertainty is part of the learning problem

There are 296 JPEGs with multiple annotation sets; 256 disagree in instance count.
The same dark feature can be omitted, outlined differently, or split differently
by independent annotators. Unioning all their polygons creates targets no single
annotator supplied and may duplicate objects. Sample a complete annotation set
per training example; retain all independent sets for evaluation. A calibrated
object-confidence score should approximate repeatable evidence across annotators,
rather than force every weak structure into the output. Do not infer a test image's
labels from external copies of its annotations.

Measure human-to-human instance agreement on training images before treating an
architecture's apparent ceiling as a representation failure. A later ablation can
use annotation-consensus weights or multiple instance hypotheses, but should keep
individual annotator masks and demonstrate gains in PQ. Sources: [data audit](data_assessment.md),
[organizer metadata clarification](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/736543).

## Validation is comparable but not proven independent

The preserved split is 495 train / 106 validation / 106 internal-test filenames.
Exact duplicate filenames cannot cross splits, and the original audit found no
exact duplicate file hashes. However, that does not remove temporal correlation:
solar structures persist across observations and sites.

A new timestamp-only audit found 22 observing dates spanning partitions (45
images). Of 106 validation images, 26 are within one day of a training observation,
69 within three days, and 99 within seven days. This is evidence of a possible
correlation risk, not proof that every nearby image is a visual duplicate.
The counts use the first 14 filename characters as observation timestamps.

Keep the existing validation split for historical comparisons. Before trusting
small improvements, add event/time-blocked cross-validation within train and
validation, with purged temporal gaps and a sensitivity analysis of grouping
width (initial candidates seven and fourteen days). Group sites observing the
same event. Bootstrap uncertainty by source image or event, not annotation row.
Report site, date, multiplicity and size strata. The old internal test has already
been evaluated; its contents and score must not guide this experiment or become
a repeatedly consulted holdout. No state/history reset is appropriate.

## What the public evidence does and does not establish

The refreshed September 6 dashboard lists 58 public competition notebooks and a
top public leaderboard score of 0.56. That is a current leaderboard observation,
not validation evidence for a known reproducible training recipe. Some notebook
headlines predate the scoring fix. A public '0.70' detector/refiner handoff now
shows a current score of 0.34; an advertised U-Net notebook's submission path
replays an embedded CSV. Architecture labels and vote counts are therefore weak
evidence. Read the executable training and submission paths, training membership,
weight provenance and current score together.

Public filament weights with unknown source-image membership cannot be fairly
compared as held-out models on our validation set. Their outputs are useful for
understanding engineering choices, but a clean comparison requires training on
our allowed fold or documenting disjointness. Full source citations and a ranked
method table are in [the research report](competition_strategy_research.md).

## Recommended path and the run launched now

The immediate run is [Filament Precision v2](../filament-precision/APPROACH.md):
warm-start the verified train-only native checkpoint, reduce positive BCE weights
and foreground oversampling, compare confidence-gated connected components with
more conservative watershed, and broaden minimum-area calibration. Its unchanged
baseline gets the same improved post-processing search, separating calibration
benefit from fine-tuning benefit. Initial weights remain eligible if training
fails to improve selection PQ. Only the validation partition is scored.

The higher-upside system is a complementary pair: a generic-pretrained proposal
detector identifies whole instances and a conditioned crop model refines their
masks; the native dense branch recovers thin or missed structures. Merge candidate
masks and calibrate object confidence on clean validation predictions, with explicit
duplicate suppression and tests on nearby filament pairs. Train a clean detector
before reusing any public filament weights. On this P100 the known default
TensorFlow runtime works; any PyTorch detector requires a separately verified
Pascal-compatible build and a real optimizer-step smoke test.

Subsequent trials should isolate loss/sampling, train-only hard-negative mining,
topology continuity losses, detector/refiner benefit and ensemble diversity.
Generic pretraining, supplied spines or external unlabeled H-alpha pretraining
are candidates supported by organizer clarifications, not automatic improvements.
Large external-data collection and expensive multi-fold ensembles come after a
clean single-fold gain. Advance based on reproducible PQ and error changes,
not a promised rank. The live first-run identity is in
[precision_training.json](precision_training.json).

## First live evidence, September 6

The private P100 run loaded the exact train-only checkpoint and completed finite
optimizer steps, with observed peak allocator memory 13.06 GB. Before fine-tuning,
recalibration of unchanged weights improved the **same 24-image calibration
subset** from PQ 0.289299 to 0.367828. Filename lists were checked for equality.
The selected recipe uses hysteresis components, grow threshold .65, confidence
.85 and minimum area 256; its counts are TP139 / FP155 / FN74.

This is a tuned calibration result and does not establish the full 106-image
validation score, a fine-tuning improvement, or leaderboard rank. Those results
remain pending. The first measured gain does support treating extraction and
object acceptance as substantive parts of the model rather than fixed defaults.
