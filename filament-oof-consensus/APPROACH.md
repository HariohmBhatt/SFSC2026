# Filament Consensus v3 — executable quality cascade

Revised September 7, 2026 after [competitive review](REVIEW.md). This document
specifies the pipeline implemented in `core.py`, `remote.py`, and `run.py`.
The first release tests whether learned instance quality and choosing between
coarse and refined masks improve the 0.38 public control. Native crop refinement,
Mask2Former, extra losses, and dense recovery are subsequent experiments, not
unimplemented prerequisites hidden inside this release.

## Evidence and objective

Control submission 56072440 scored 0.38 public PQ. Its 495-image trained model
scores 0.435179 PQ on 106 validation JPEGs (723 TP, 314 FP, 514 FN; SQ 0.684369,
RQ 0.635884). The 82 images outside calibration score 0.434862. The new CPU
metric reproduces the control to floating-point precision; see
[control_reproduction.json](control_reproduction.json).

The candidate pool's 0.836 loose best-of-seed/refined recall is an optimistic
per-truth diagnostic, not achievable recall or a promised gain. Refinement both
rescues and damages masks. Mean foreground probability is a weak quality proxy.
The concrete experiment is therefore to predict expected matched-IoU contribution
from observable candidate features and select one coarse/refined alternative per
proposal. The source rationale is [Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241);
its COCO result is not evidence of an SFSC score.

## Model and controlled changes

Retain the proven generic-pretrained YOLO11l-seg at 1280 and prompted ResNet18
U-Net at 384. Hold their optimization and geometry fixed. The only new learned
component is a regularized quality model, fitted to predictions from complete
held-out detector/refiner pipelines.

For every proposal, generate both the coarse seed and refined mask. Features:
detector log-odds, refiner foreground confidence, mask and seed area, area ratio,
seed/mask IoU, aspect ratio, box occupancy, component count, intensity inside the
mask, local ring contrast, and a coarse/refined indicator. Everything is available
from predictions and the same image at inference. No filename, date, site, test
prevalence, or annotation metadata is a model feature.

The target for each candidate is the mean over its JPEG's annotation records of
`best_IoU * 1(best_IoU > 0.5)`. This is an unconditional quality contribution;
do not multiply it by a second presence probability. Candidate-local targets do
not mark valid duplicates as empty merely because another proposal exists.
Set-level matching still determines measured PQ. Annotation-record count weights
the quality loss to reflect repeated-record evaluation; additional agreement
weighting is not applied.

Fit fractional logistic regression with L2 coefficient 0.02 and standardized
features. The regularizer and feature list are frozen before validation. No
isotonic layer or second learned mask head is added in this release. Persist the
feature schema, scaling, coefficients, objective, membership and hashes.

## Complete out-of-fold training, without stacked assessment leakage

Use only the preserved 495 training JPEGs for quality supervision. Assign fixed
60-day date blocks to three folds using a deterministic SHA-256 rule. A block
includes all sites. Purge training observations within seven days of any held
observation. Membership is frozen before GPU launch:

| Fold | Train | Held | Purged |
|---|---:|---:|---:|
| 0 | 264 | 216 | 15 |
| 1 | 348 | 136 | 11 |
| 2 | 340 | 143 | 12 |

For each fold train BOTH the detector and refiner from generic initialization
using only its training images. Held images cannot select epochs or supply
refiner examples. Evaluate each held image only after the fixed fit finishes.
Across folds every one of the 495 JPEGs has exactly one held prediction set.
Quality targets use each annotation set independently, without unioning masks.
The inherited segmentation-training control uses one medoid annotation record;
changing that sampler is a later isolated ablation.

All 495 OOF prediction sets train the quality learner. No accuracy computed on
those same quality-training examples is advertised as held-out pipeline quality.
This distinction avoids an expensive nested scheme while preserving exclusion
of the actual validation images from every trained component. Training-side
quality loss is optimization evidence only.

The 106 historical validation images remain the comparison set. Their existing
24-image calibration subset selects one acceptance threshold. The remaining 82
are assessed after that choice is frozen. They have been used in earlier model
development and are explicitly not a newly untouched test. Never reuse the 106
internal-test images. Competition-test images supply inference pixels only.

## Frozen training schedule

Each OOF detector runs 36 epochs at batch two, AdamW lr 0.001, cosine final factor
0.05, two warmup epochs, weight decay 0.0005, and the verified control's
augmentation. No mosaic, mixup, new P2 head, resolution change, or early stopping.
Use the last checkpoint. Any Ultralytics diagnostic validation uses training
images and cannot select the saved checkpoint. Preserve the inherited nominal
batch/accumulation behavior and capture actual runtime arguments in artifacts.

Each refiner runs nine epochs, batch eight, AdamW lr 0.0003, weight decay 0.0001,
loss `0.45 BCE + 0.55 Dice`. Step the original TEN-epoch cosine schedule for nine
epochs, preserving the control's selected schedule position. No crop-Dice
checkpoint selection. Keep square-ROI geometry, context 1.6, minimum crop 128,
mask threshold 0.55, and no TTA for this experiment.

Use PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126, Ultralytics 8.4.103 and
pycocotools 2.0.11. Run on a Kaggle P100, check compute capability sm_60, and
record a real backward operation followed by finite-loss training batches.
Kaggle CPU fallback is an error. Release detector memory before refiner training.

## Candidate extraction and set assembly

Candidate generation reproduces the control: detector confidence 0.03,
Ultralytics box-NMS IoU 0.6, class-agnostic suppression, retina masks, max 100
candidates, followed by explicit seed-mask duplicate suppression at mask IoU 0.6.
Record these two suppression stages separately. The scorer cannot restore a
proposal removed before export; changing early suppression requires a subsequent
experiment. Ignore alternatives below 32 native pixels in the cache.

For the final set, score both alternatives, retain the highest-scoring eligible
alternative per prompt, and require area at least 64. Process prompts by quality,
suppress mask duplicates at IoU 0.6 against accepted masks, then remove pixels
already assigned to higher-ranked instances. Discard remnants below 64 pixels.
Score only these final masks with the exact additive PQ evaluator. Coarse and
refined masks from the same prompt can never both become predictions. Empty
images are allowed; no forced fallback or per-image count target is used.

Choose the acceptance threshold from the frozen grid
`0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40` on the 24 calibration JPEGs.
Choose by global PQ, with the higher threshold breaking an exact tie. Minimum
area, duplicate rule, features and regularization are not searched simultaneously.

## Validation and full-fit decision

The assessment kernel attaches the three completed OOF outputs plus the previous
validation kernel. Verify run/source identity, train/held manifests, OOF coverage,
and candidate hashes. Fit quality on the OOF pool. Read the frozen control's
106-image candidate cache, reproduce 0.435179 PQ, and construct identical image
features for quality inference. No new GPU inference or retraining of the
495-image control is needed for this comparison.

Report baseline and candidate PQ, SQ, RQ, TP, FP, FN; additive statistics for every
JPEG; calibration grid; all-106 and assessment-82 scores; and paired bootstrap
differences over shared 60-day observation blocks (2,000 replicates). This
interval is descriptive and conditional on historical model/threshold selection;
it does not include all training uncertainty or predict the hidden leaderboard.

The automatic full-fit gate requires all of:

- all-106 PQ at least 0.460;
- assessment-82 PQ at least 0.455 and gain at least 0.020 over its control;
- positive lower endpoint of the paired assessment-block 95% interval;
- official pair-counting and one-to-one PQ agreement within 1e-8 for this output.

SQ, RQ and size-specific aspirations are diagnostic rather than arbitrary
additional vetoes. A metric disagreement stops promotion for investigation;
non-overlapping predictions do not guarantee parity with overlapping truths.
A failed gate stops the controller with a complete report and spends no further
full-training compute. An explicit, user-authorized `run --override-gate` may
continue final fitting while preserving the failed assessment and override
provenance; ordinary resumes then honor that durable authorization. There is no
automatic fallback submission.

## Full training and inference

If the gate passes, or the failed gate is explicitly overridden as described
above, the next kernel fits the same detector and refiner recipe on
all 601 train-plus-validation JPEGs, retaining the fixed 36/9 durations and
original schedules. The quality coefficients remain frozen from clean OOF
training. No quality targets are fitted from in-sample full-model predictions.
Smaller-fold versus full-model score distributions may differ; keep that transfer
risk visible, and preserve the validation-trained and full-trained artifacts.
The previous full-refit regression means this stage is not guaranteed to improve
public PQ.

Run the 601-image models on all 180 competition images. Apply the exact shared
feature extraction and assembly implementation with the frozen quality model
and threshold. Write COCO RLE submission rows plus a manifest that contains ALL
180 filenames and their counts, including zeros. Verify IDs, nonempty instances,
non-overlap, RLE round trips, counts, source/weight/quality hashes and CSV hash.
The controller independently rechecks the downloaded CSV and records
`inference_ready`. This training launch does not upload a leaderboard submission.

## Operations and estimates

Local preparation and orchestration:

```bash
uv run python filament-oof-consensus/run.py prepare
uv run python filament-oof-consensus/run.py run
uv run python filament-oof-consensus/run.py run --override-gate
uv run python filament-oof-consensus/run.py status
```

The controller runs in the background, serially pushes private P100 kernels,
waits for completion, downloads reports, and advances only on verified lineage.
The final download includes both trained checkpoints and independently verifies
their hashes against the final report.
It records push intent before upload and reuses kernel IDs on resume. A local
lock prevents two controllers from running simultaneously. An ambiguous or
failed stage stops with an error instead of blindly restarting training.

Frozen source copies and packed stage scripts live in `outputs/consensus-<run>/`.
The shared `reports/consensus_training.json` records stage IDs and progress.
Each remote stage writes `consensus/progress.json`, optimizer-batch evidence,
trained checkpoints and compressed candidate masks. The fold models are complete
independent jobs so a later failure does not erase previously completed outputs.

Estimate 1.5–2.5 hours for the first fold, 6–8 hours for all OOF training and
candidate generation, then assessment. Gated full training/inference adds about
4 hours. These are estimates from the previous P100 run, not measured new-run
ETAs. Each kernel has a nine-hour ceiling. Profile actual epoch times and storage;
no automatic schedule changes are allowed after seeing held performance.

## Subsequent experiments

If quality selection succeeds, the next isolated experiment changes prompted
training to realistic detector mistakes, then adds explicit global/native local
refinement with identical training/inference geometry. It must generate quality
features from refiners that exclude those feature images. After that, test
standard native-crop proposals; Mask2Former is a challenger requiring its own
compatibility and incremental-PQ evidence. Do not bundle those interventions or
claim that the 0.49 validation milestone establishes a winning public score.
