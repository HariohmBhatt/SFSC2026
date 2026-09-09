# Research for the model after YOLO11l + ResNet18

Research date: **2026-09-07 UTC**.

## Decision

Build a **dual-scale proposal, native-resolution refinement, learned-quality
cascade**. Keep the proven whole-disk YOLO11l branch, add a small query-based
instance model on overlapping native-resolution crops, and replace the current
synthetic-prompt ResNet18 refiner with a proposal-trained local/global refiner
that also predicts object presence and mask IoU. Train the second stage only on
out-of-fold (OOF) detector predictions. Select and fuse candidates with those
learned quality estimates under the exact PQ evaluator.

This is the best-supported route to a material improvement on one Kaggle P100.
It attacks measured losses in the current system rather than relying on a larger
backbone alone. It cannot guarantee a leaderboard score; a reasonable promotion
target is **at least 0.46 validation PQ on all 106 validation images and at least
0.455 on the 82 images outside calibration**, before paying for full-data refit.

## What the completed run establishes

The current model is already a strong control. YOLO11l-seg at 1280 plus the
prompted ResNet18 refiner reaches validation PQ **0.435179**, SQ **0.684369** and
RQ **0.635884**, with TP/FP/FN **723/314/514** over 169 annotation records. The
82 images outside calibration score **0.434862**. Its full-data fit produced
1,151 test instances in 12,939 seconds with 13.12 GiB peak allocated VRAM and
scored **0.38 public**. Sources: [validation report](../outputs/yolo-refiner-validation/sfsc-yolo-refiner-validation/validation-report.json),
[final report](../outputs/yolo-refiner-full/results/sfsc-yolo-refiner-final/final-report.json),
[submission record](yolo_refiner_run.json).

The remaining limitation is mainly recognition quality. Raising SQ alone from
0.684 to 0.72 at unchanged RQ would yield only 0.458 PQ. Raising RQ to 0.68 at
unchanged SQ would yield 0.465. The next system therefore needs more correct
instances and better selection, with boundary refinement as a supporting gain.

The current implementation leaves unusually actionable headroom:

- It trains YOLO from one fixed medoid annotation record per filename and trains
  the refiner from jittered ground-truth masks plus one heuristic dark negative
  per image. It never trains the refiner on the actual detector errors it must
  process at inference. See [training implementation](../kaggle/yolo-refiner/train_validation.py).
- The refiner forces every proposal through a square crop resized to 384. A
  1,000-pixel elongated object can therefore be reduced several-fold even
  though the supplied image is 2048 square. The long-object path merely keeps
  the seed in a limited hybrid recipe; the selected recipe refines every mask.
- Candidate ranking is `detector_score * sqrt(mean foreground probability)`.
  It has no learned presence or mask-quality target. Overlaps are settled by
  painting higher-ranked masks first, so a mis-ranked mask can damage several
  instances.
- There is one detector/refiner model, no native crop proposal branch, and no
  TTA in the selected run.

I decoded the saved low-threshold validation candidates and compared them with
all 1,237 ground-truth instances across the 169 validation records. This is a
proposal diagnostic, not an attainable PQ oracle: it lets each truth choose its
best candidate independently and thus ignores one-to-one conflicts and the
requirement that one prediction set serve all annotators.

| Truth area | Truths | best seed recall at IoU>0.5 | best refined recall | best of either |
|---|---:|---:|---:|---:|
| <1,024 px | 508 | 0.713 | 0.799 | 0.809 |
| 1,024–4,095 px | 581 | 0.838 | 0.828 | 0.861 |
| >=4,096 px | 148 | 0.797 | 0.804 | 0.831 |
| all | 1,237 | 0.782 | 0.813 | 0.836 |

Across candidate/annotation-record pairs, detector confidence has Spearman
correlation about **0.578** with refined-mask IoU. The refiner's mean foreground
probability has correlation only **0.175**, and multiplying it into detector
confidence leaves correlation essentially unchanged (**0.578**). Refinement
rescues 67 truths across the 0.5 boundary but loses 28 that the seed matched.
These calculations use [saved candidates](../outputs/yolo-refiner-validation/sfsc-yolo-refiner-validation/validation-candidates.json)
and the [official training annotations](../SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json).
They show that better proposal scoring and choosing between seed and refinement
can recover value before an expensive new detector is considered.

## Proposed architecture

### 1. Two complementary proposal views

Retain the current YOLO11l-seg whole-disk model at 1280 and infer at a low
proposal threshold such as 0.03. It is a proven source of global context and
long-object hypotheses.

Add **Mask2Former with a Swin-T backbone** on overlapping 768 or 1024 native
crops. Train it as a single-class instance model with 50–100 queries. Crops must
be sampled so complete small/medium objects occur away from an edge; inference
uses overlap and rejects tile-edge masks unless another tile or the global
branch supports them. Mask2Former's masked attention localizes each query within
its predicted mask, and the architecture was designed for instance and panoptic
segmentation. A query branch is particularly relevant here because it supplies
set-level instance hypotheses without depending on the same YOLO mask prototype
and NMS path. The paper's results are on standard vision datasets and do not
prove an SFSC gain. Source: [Mask2Former paper](https://arxiv.org/abs/2112.01527),
[official source](https://github.com/facebookresearch/Mask2Former).

Use Swin-T, batch one, AMP, gradient accumulation four, and cropped input for the
first P100 experiment. Do not begin with Mask DINO or a large Swin backbone.
Mask DINO is a strong query-based detection/segmentation framework, but the
additional engineering and memory risk are unjustified until the small
Mask2Former challenger demonstrates complementary proposal recall. Source:
[Mask DINO paper](https://arxiv.org/abs/2206.02777).

If the Mask2Former optimizer-step smoke test exceeds 15 GiB allocated memory or
cannot sustain useful throughput at 768, substitute a YOLO11m-seg crop model
with a stride-4/P2 head. Feature pyramids were explicitly designed to reuse
multi-scale feature hierarchies for recognition at different scales. Source:
[FPN paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html).
This fallback preserves the important independent variable—native crop
proposals—even if the query architecture is infeasible on the P100.

### 2. Train on the proposal distribution seen at inference

Create three grouped folds within the 495 training filenames. Every annotation
record for a filename remains in one fold. For each fold, train proposal models
on the other two folds and save low-threshold predictions for the excluded fold.
Match those OOF candidates to each complete annotation record and construct:

- positive prompts, including partial masks, split fragments and duplicate
  candidates;
- hard negatives from detector false positives;
- a segmentation target sampled from one coherent annotation record;
- `presence = fraction of that filename's annotation records with a one-to-one
  match at IoU>0.5`;
- `quality = mean matched IoU contribution across its annotation records`, with
  zero for unmatched records.

This uses all available annotation opinions without unioning conflicting
instances. It also makes the quality target reflect the organizer's repeated
annotation records. Stacked generalization trains a later learner from the
base learners' predictions on held-out portions of the learning set; using
in-sample detector predictions would produce an unrealistically clean second-
stage distribution. Source: [Wolpert, *Stacked Generalization*](https://doi.org/10.1016/S0893-6080(05)80023-1).

### 3. Replace square downsampling with local/global native refinement

Use a ResNet34 U-Net-sized refiner rather than a foundation model for the first
run. Inputs are raw grayscale, CLAHE, coarse mask probability, a signed distance
transform of the prompt, and a box channel. Initialize image channels from
ImageNet and new prompt channels to zero.

For each proposal, make a low-resolution global crop for context and one or more
512-pixel local native-scale windows. A small object uses one local window. An
elongated or >512-pixel object uses overlapping windows following its coarse
mask, then blends soft logits in native coordinates before thresholding. Feed
the same full-object prompt into every window so the windows refine one instance
rather than independently create instances. This follows the tested principle
in CascadePSP: combine global context with local recursive refinement instead of
upsampling a coarse mask and expecting high-resolution boundaries to reappear.
Source: [CascadePSP paper](https://arxiv.org/abs/2005.02551),
[official implementation](https://github.com/hkchengrex/CascadePSP).

The refiner has four outputs:

1. native mask logits;
2. an auxiliary boundary map;
3. probability that the proposal corresponds to an annotation-supported
   filament;
4. predicted IoU for both the coarse seed and the refined mask.

Train mask logits initially with `0.35 BCE + 0.35 soft Dice + 0.20 Lovasz +
0.10 boundary`, then test a **small** 0.05 soft-clDice term by taking weight from
BCE. Lovasz supplies a tractable surrogate for IoU, while soft-clDice emphasizes
connectivity in tubular structures. Both must earn promotion under PQ because a
strong topology term can also preserve an erroneous bridge between neighboring
filaments. Sources: [Lovasz-Softmax](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html),
[clDice](https://arxiv.org/abs/2003.07311).

Train the presence head with binary cross-entropy and the two IoU heads against
measured mask IoU. Mask Scoring R-CNN directly documents the underlying failure:
classification confidence is not reliably aligned with mask IoU, and an
explicit MaskIoU head improves mask ranking. Source: [Mask Scoring R-CNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html).

### 4. Assemble instances with a metric-aligned candidate graph

For every coarse/refined candidate pair, estimate expected matched-IoU
contribution as `presence_probability * predicted_IoU`. Calibrate this value on
the fixed 24 calibration images with isotonic regression or a monotone bin map;
fit no calibrator on the other 82 validation images.

Cluster candidates by mask IoU, containment and prompt identity. Within a
cluster, let the quality heads select the coarse mask, refined mask, or a
probability average. Preserve candidates supported by both global and crop
models at a lower acceptance threshold; require stronger evidence from a
singleton. Resolve final pixel overlaps by calibrated per-pixel logits and then
discard any remainder below its calibrated area threshold. Search a small grid
over acceptance, model-support and overlap parameters with the exact additive
PQ evaluator.

This is aligned with the metric: under one-to-one PQ,
`PQ = 2 * matched_IoU_sum / (N_truth + N_prediction)`. Adding an unsupported
candidate always increases the denominator, while a supported high-IoU
candidate adds numerator. A learned expected-IoU score is consequently more
appropriate than mean foreground probability. Keep both official pair-counting
and conservative Hungarian one-to-one diagnostics, and require identical
results for the promoted non-overlapping prediction sets. Source:
[repository metric contract](../EVAL.md),
[organizer self-evaluation](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).

### 5. Add ensemble and TTA only after the single model wins

If the cascade clears the gate, train a second proposal seed and use horizontal,
vertical and 180-degree transforms. Invert masks before matching, require
agreement in the candidate graph, and keep the single-view mask when transformed
predictions fragment it. Do not average unmatched instance arrays by index.
Two complementary proposal models plus two seeds are enough for the first final
fit; a five-model ensemble is a poor first use of this P100.

## Validation and ablation order

Keep the historical 495/106 split for comparison with 0.435179. Keep the same 24
filenames for bounded calibration and report the other 82 separately. Resample
paired additive statistics by filename, keeping all annotator records together.
Also run the already specified seven-day time-blocked sensitivity split before
claiming generalization. The local audit shows that exact filename grouping does
not remove temporal proximity. Sources: [problem analysis](competition_problem_analysis.md),
[generalization audit](generalization_gap_analysis.md).

Run these experiments in order, preserving every candidate file so later stages
can be evaluated without repeating inference:

| ID | Change from current control | Decision evidence |
|---|---|---|
| B0 | Re-evaluate frozen current candidates | Must reproduce 0.435179 exactly |
| B1 | OOF quality/presence heads and seed-vs-refined choice | PQ, score-IoU calibration, TP/FP/FN |
| B2 | Native local/global refiner | PQ/SQ plus tiny, medium, large and elongation recall |
| B3 | Add low-weight Lovasz/boundary, then clDice | PQ and split/merge counts; keep only additive terms |
| B4 | Add native crop proposal model | Proposal recall and final marginal PQ after quality gate |
| B5 | Second seed plus guarded dihedral TTA | Paired PQ delta versus inference cost |

Advance to full refit only if full validation PQ is at least **0.460**, the
non-calibration 82 score at least **0.455**, and the paired filename bootstrap
interval is materially positive. Also require no catastrophic site or large-
object regression, peak allocation below 15 GiB, valid non-overlapping RLEs, and
one-to-one/official evaluator parity. These thresholds are engineering gates,
not a claimed score forecast.

## P100 execution plan

The known good environment is PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126 and
Ultralytics 8.4.103. Reuse it and install only the smallest additional query-model
dependencies. The completed full run already reached 13.12 GiB, so proposal
models, the refiner and inference must run sequentially. Source:
[completed run](../outputs/yolo-refiner-full/results/sfsc-yolo-refiner-final/final-report.json),
[P100 notes](../GPU.md).

1. Run a real Mask2Former forward/backward optimizer step at 768, batch one,
   AMP, and record allocated/reserved VRAM and seconds/step. Stop or use the P2
   fallback at 15 GiB.
2. Generate OOF proposals once and store masks as compressed RLE plus scores;
   do not save dense 2048 float maps for every candidate.
3. Train the refiner at 512 local resolution, batch two to four as measured,
   with gradient accumulation to effective batch eight. Cache image features
   only if cache size and exact augmentation semantics are verified.
4. During inference, run whole-disk proposals, crop proposals, and refiner
   sequentially. Decode and refine one image at a time. Release each proposal
   model before loading the refiner if memory is tight.
5. Keep the current YOLO submission artifact as the fallback. Full-data training
   begins only after the validation gate, using the selected epochs and frozen
   selection logic.

Self-supervised H-alpha pretraining, HQ-SAM, Swin-L/Mask DINO and a five-fold
ensemble remain later experiments. The organizer permits public additional
unlabeled GONG observations, but the existing evidence does not show that an
expensive pretraining phase beats correcting proposal mismatch and score
calibration. Source: [organizer clarification and public-method audit](competition_strategy_research.md).

## Expected source of gain

The target improvement is not assigned to a speculative larger backbone. The
measured candidate pool already exposes an 0.836 loose recall ceiling when the
better of seed and refined mask is chosen, versus 0.584 realized TP/GT in the
final aggregate. OOF quality learning should capture part of that selection
gap. Native local/global refinement targets the 147 candidate/record cases in
which refinement reduces IoU by more than 0.05, while retaining the 400 cases
where it increases IoU by more than 0.05. The crop query branch then adds
resolution and model diversity for genuine misses. The architecture earns a
full run only when each source of gain is visible under the exact competition
metric.
