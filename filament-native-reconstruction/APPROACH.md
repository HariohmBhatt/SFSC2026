# Filament Native Reconstruction v4

September 7, 2026. An exploratory native-refinement diagnostic was launched on
September 8. It reuses the three excluded-prediction caches from Consensus v3
for honest training prompts and reports against the historical 106-image
validation. It is cheaper than the blocked pilot specified below and cannot
promote the approach. The matched realistic-prompt experiment, R/Q/H pilot,
sliced detector, and full fit remain unimplemented.

## Decision

Build a **proposal-conditioned global/local instance refiner**, then test
native-scale detector views for objects the global detector misses. Retain the
proven YOLO11l-seg and compact ResNet18 components initially. Learn from realistic
held-out proposals, preserve whole-instance identity through a global mask, and
reconstruct each instance from overlapping native-resolution windows.

The central hypothesis is that recovering structure and missing instances will
produce more PQ improvement than ranking the existing candidate masks alone.
Keep quality selection, but treat it as the final selection layer. It is not a
substitute for generating a better mask.

This is a stronger experimental strategy, not evidence of a higher score. The
historical leader's 0.56 is a useful ambition, not a current verified leaderboard
or a result this architecture can promise.

## 1. What the evidence actually says

The best recorded public artifact scored **0.38**, submission 56072440. Its
validation-trained recipe is YOLO11l-seg at 1280 followed by a prompted ResNet18
refiner at 384. On 106 validation JPEGs, evaluated over 169 annotation records:

| Quantity | Control |
|---|---:|
| PQ | 0.435179 |
| SQ | 0.684369 |
| RQ | 0.635884 |
| TP / FP / FN | 723 / 314 / 514 |
| Non-calibration 82-image PQ | 0.434862 |

Source: [saved validation report](../outputs/yolo-refiner-validation/sfsc-yolo-refiner-validation/validation-report.json).
These validation numbers belong to the validation-trained recipe, not a claim
that the submitted full-fit weights were independently validated on those images.

When one-to-one matching and official counting agree, let S be summed matched
IoU, G ground-truth count and P prediction count. Then `PQ = 2S / (G + P)`.
Here S = 494.798556, G = 1,237 and P = 1,037. Illustrative changes on exactly
this population show why both recognition and geometry matter:

| Hypothetical isolated change | Resulting PQ |
|---|---:|
| Remove 50 false predictions without losing a TP | 0.444963 |
| Recover 50 missed truths at IoU 0.65, with 50 additional predictions and no FP | 0.453785 |
| Improve each existing TP's IoU by 0.05, preserving all matches and counts | 0.466973 |

These are arithmetic sensitivity examples, not measured interventions. Real
refinement can damage matches; recovery can introduce false predictions. At
SQ 0.75, PQ 0.56 would still require RQ about 0.747 on the same population.
Winning needs substantial, simultaneous progress, not threshold polishing alone.

The loose 0.836 candidate-recall diagnostic uses truth to pick masks independently.
It is not an achievable set-level ceiling. Before architecture training, classify
errors into proposal absence, insufficient seed coverage, refinement damage,
rejection, duplication and overlap assignment. Use maximum-cardinality matching
with IoU as a secondary objective for recall diagnostics; never count one
candidate as recovering several truths. Report oracle diagnostics separately
for each annotation record and never deploy truth-selected alternatives.

## 2. Architecture: identity globally, geometry locally

```text
H-alpha image
  -> global YOLO11l-seg proposals
  -> [later ablation: overlapping native-scale detector views]
  -> proposal seed + expanded instance ROI
  -> whole-ROI 384 global refiner
  -> 512 native windows conditioned on that proposal and global prediction
  -> blend windows into one mask in original-image coordinates
  -> compare seed / global / native alternatives using learned quality
  -> duplicate suppression and overlap assignment
  -> exact PQ / COCO RLE
```

### Global instance path

Start from the existing prompted ResNet18 refiner architecture. For each seed,
form an ROI centered on its bounding box with side
`max(128, ceil(1.6 * max(box_width, box_height)))`. Clip to the image; retain the
transform and valid-pixel mask. Letterbox the clipped ROI to 384 rather than
distorting its aspect ratio. The global output supplies whole-instance context
to all local windows. Verify the current geometry and retain it in the matched
baseline: any letterboxing difference must be an explicit ablation.

Use the entire expanded ROI for local processing. A 400-pixel seed box with
context 1.6 creates a 640-pixel ROI and therefore needs multiple native windows.
The local path must not depend on the seed's bounding-box size alone.

### Native local path

Use a second compact ResNet18 U-Net, initialized from generic image pretraining,
with a small input adapter for six channels: normalized image intensity, seed
mask, resampled global probability, normalized x/y coordinates within the ROI,
and valid-pixel mask. Global probabilities are detached. The seed and global
channels identify which filament to segment when several filaments share a tile.
Also pool the frozen global encoder's last feature map, project it to 64 channels,
and broadcast that vector into the local decoder bottleneck. This explicit
whole-ROI feature path carries context beyond the cropped probability map. Train
the projection with the local decoder; cache only detached global features.
Predict a residual over the softened global-refiner mask and initialize the
residual head at zero. Keep the detector seed as a conditioning channel, but do
not union it into the prior: doing so would restore pixels the global refiner
already rejected at a large positive logit. This initialization preserves the
current refined alternative while allowing the model to add and remove pixels.
Retain seed, global and native alternatives for the later quality stage; the
residual prior is not a deployment fallback rule.

Windows are **512 original-image pixels**, stride 384; no downsampling to fit a
large ROI into one window. Cover the full ROI, including global-confident
background, since an omitted thin branch can be confidently wrong. An ROI below
512 is padded, not enlarged. Use reflection padding for image pixels and zero
padding for prompts/validity; exclude invalid pixels from losses and assembly.
Anchor the last window at the ROI edge so there are no uncovered strips.

Blend overlapping probability predictions with a positive, tapered spatial
weight, divide by accumulated weights, then threshold the reconstructed mask.
Do not threshold each tile separately or use connected components as separate
instances. All tiles from a proposal remain one proposal, even if its annotation
contains disconnected pieces. Calibrate a compact predeclared threshold grid for
every new output head. The inherited 0.55 threshold is a control value, not
transferable calibration for a residual native decoder.

An optional, separately measured expansion rule can rerun a proposal once with
ROI context 2.4 if global foreground reaches the original ROI boundary. Apply
the same rule in training; cap expansion at the image. Do not use it in the first
geometry comparison. It cannot recover a branch entirely absent from the seed
and global prediction; wider proposals address that failure separately.

This borrows global/local refinement from
[CascadePSP](https://arxiv.org/abs/2005.02551), but is a new compact design, not a
claim to reproduce CascadePSP. [PointRend](https://arxiv.org/abs/1912.08193)
supports adaptive fine-detail refinement; uncertainty-only sampling is deferred
because confidently missing filament structure is a central risk here.

## 3. Teach the model the errors it will encounter

First compare realistic-prompt training with the current synthetic-prompt
refiner, holding backbone, loss and geometry constant. Then add native windows.

Generate training prompts with detectors that did not train on those images.
Keep partial objects, displaced masks and genuine background proposals, not just
high-IoU detections. For local-refiner training, generate global probabilities
with an excluded global refiner too. Training on perfect masks as both prompt
and target is not representative of inference.

Match each prompt to a target using overlap and ambiguity checks, independently
within each annotation record. Initial recoverability rule: at least 50% of seed
pixels overlap one truth and seed coverage of every other truth is below 10%.
This allows a small fragment of a long filament to request the complete object.
Log the distribution of IoU, truth coverage and seed purity, and freeze the rule
before assessment. Ambiguous multi-object seeds do not receive arbitrary merged
targets; exclude them from positive segmentation loss in the first release.

Train multiple valid duplicate prompts against the same instance when appropriate.
An otherwise valid prompt must not become an empty target because another
proposal would win duplicate suppression. Confirm background across all available
annotation records before using a proposal as a hard negative. Unmatched but
ambiguous proposals are excluded from empty-mask loss, not declared false objects.

For the first controlled comparison retain the existing medoid annotation
record for positive masks. Later test sampling complete annotator records, using
their evaluator weight; never union disagreeing records into one target. Do not
multiply segmentation losses by an invented consensus-confidence weight.

Start with `0.45 BCE + 0.55 Dice`, averaging per proposal so long objects do not
dominate solely because they generate more tiles. Sample a bounded number of
tiles per proposal per epoch and cycle coverage across epochs; include both
foreground and empty windows. Reconstruct complete masks at evaluation.
Topology losses and boundary losses are subsequent ablations only: a thin-looking
mask or an improved skeleton is not necessarily a better instance at IoU > 0.5.

## 4. Recover objects outside the global candidate pool

Once native refinement earns its compute, test sliced inference with the same
YOLO detector before training another architecture. Initial view policy for a
2048-square image: four 1280-square crops at offsets 0 and 768 along each axis,
plus the whole-image view. Generalize offsets from actual image dimensions and
record every transform. This keeps the crop view at native scale and gives
objects more detector pixels than whole-image downsampling to 1280.

Use the same initial confidence 0.03 and per-view cap 100; measure saturation.
Map masks into native coordinates and send them through the same global/local
refiner on the original image. A crop-border fragment is a prompt, not a complete
final instance. Keep global proposals for long-object identity; do not blindly
union neighboring fragments or discard every border-touching proposal.

Trace detector box NMS separately from later mask suppression. Initially keep
box NMS unchanged within each view, delay cross-view mask suppression until masks
have been refined, and record additional matches and FP from each source. A
separate relaxation of upstream box NMS is warranted only if suppression audits
show real losses. [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/)
defines these inference controls; [SAHI](https://arxiv.org/abs/2202.06934)
motivates testing sliced views, without establishing a solar-filament gain.

Promote crop views by complete PQ and runtime after reconciliation, not raw
proposal recall. If they help but generalize poorly, fine-tune the same detector
on a mix of full and cropped views as a distinct experiment. Train the refiner
and quality model on excluded predictions from the enabled view policy.

## 5. Quality and set assembly

Keep seed, global and native masks as alternatives for each prompt. Fit expected
matched-IoU contribution
`u = E[best_IoU * 1(best_IoU > 0.5)]`, averaging annotation records as the
evaluator does. This is unconditional utility; do not multiply it by another
presence probability. Start with v3's regularized feature-based model, extending
the alternative-type schema explicitly. Add border contact and cross-view support
only as predeclared feature ablations.

**Refit quality for the new mask distribution.** v3's coefficients and threshold
are useful controls, not calibrated defaults for a new refiner. Quality-training
images must be excluded from detector, global refiner and local refiner fitting.

Initially use v3's reproducible assembly: one eligible alternative per prompt,
quality ordering, mask-IoU duplicate threshold 0.6, minimum area 64, and overlap
painting. Evaluate the resulting masks, not their pre-painting versions. Log
instances damaged or removed by painting and record suppression provenance.
Use the same small acceptance-threshold grid as v3, calibrated on a separate
partition. Do not jointly sweep area, mask threshold, NMS and acceptance.

Keep lower-area candidates in diagnostic caches to quantify the current area
floor. Lowering the final area floor is a separate PQ experiment, not an assumed
tiny-object improvement. Empty image outputs remain valid. There is no count
quota and no forced instance fallback.

## 6. A validation design that can reject this proposal

### Affordable first pilot

Before any new fit, partition the original 495 training JPEGs into fitting R,
quality Q and assessment H using shared 60-day date blocks across all sites.
The frozen filename-only assignment in `core.make_pilot_split` produces 289 R,
69 Q-fit, 34 Q-calibration, 87 H and 16 purged JPEGs. Record the membership hash
with every stage. Q has separate block-disjoint quality-fit and
threshold-calibration portions. Purge
fitting observations within seven days of H, and R within seven days of Q.
Purge quality-fit observations near its calibration blocks as well. If these
counts are too small, redesign membership before training or reading outcomes.

| Component | Supervision / selection allowed | Predictions used for |
|---|---|---|
| Two inner detector/global fits inside R | Their own R subfold only, purged against its held subfold | Realistic excluded prompts and global maps for local training on R |
| Deployed detector + global refiner | R only; global refiner uses inner detector prompts | Inference on Q and H |
| Native local refiner | R only, with inner excluded upstream predictions | Inference on Q and H |
| Quality model | Q-fit, predicted by R-trained components | Candidate scores on Q-calibration and H |
| Acceptance threshold | Q-calibration only | Frozen assembly on H |
| H assessment | No component fitting or selection | One predeclared comparison |

Inner global refiners use the fixed synthetic-prompt recipe initially; both they
and their detectors exclude the inner held images. This gives local training
honest upstream errors without recursive unlimited nesting. The deployed global
refiner's realistic-prompt change is evaluated as its own stage; log the remaining
inner/deployed prompt-distribution difference.

Retrain the old recipe on exactly R and score it on the same H. Comparing a
reduced-data challenger against a control that trained on H would be invalid.
Both baseline and challenger get the same permitted calibration budget. Record
every component's train, checkpoint-selection, feature-generation and assessment
membership. Existing v3 OOF caches may diagnose errors, but cannot automatically
be reused inside these folds: some upstream models may have trained on H.

H has historical development exposure; this is prospective exclusion from new
fits, not a newly untouched test. Freeze one second blocked pilot with a different
H, block-disjoint from the first H, before opening the first result. Run that replication only for the selected
configuration. Do not tune through repeated readings of either H. Keep the old
106-image validation as secondary historical evidence, and never reuse the
consumed 106-image internal test for selection.

### Required reporting and promotion

Use the repository's exact additive official evaluator, **strict IoU > 0.5**,
plus one-to-one parity diagnostics. Aggregate sufficient statistics across
annotation records; do not average JPEG PQ as the primary score. Explain and
resolve parity differences before promotion rather than optimizing an overlap
counting loophole.

Report paired PQ delta, SQ/RQ/counts, size/width/length bins, site/year strata,
proposal recall before and after each stage, refinement damage, and time/memory.
Bootstrap shared 60-day blocks with 2,000 replicates. Sparse subgroup results are
diagnostic. Intervals remain conditional on model selection, not leaderboard
forecasts. Repeat the selected design with a 14-day purge if seven-day temporal
dependence appears material.

Predeclare a practical target of at least +0.015 PQ over the matched control on
the first pilot. Require positive delta on the second blocked pilot and a
positive lower bound of the pooled, paired block-bootstrap 95% interval before
an expensive final campaign. Keep pilot-specific results visible; pooled evidence
does not erase a regression. If the interval remains inconclusive, retain the
control and report uncertainty rather than buying submissions to resolve it.

## 7. Experiment order and P100 budget

| Order | Change under test | Proceed only if |
|---|---|---|
| 0 | Numerical/integrity diagnostic on historical validation | Finite weights and non-collapsed masks; no promotion claim |
| A | Freeze the 0.38 control and audit error stages | Inputs, metric and model lineage agree |
| B | Realistic prompts, same 384 architecture | Complete-mask PQ improves under matched membership |
| C | Add native local reconstruction, fixed proposals | Gains survive complete-mask assembly and runtime measurement |
| D | Add four crop views | Incremental PQ justifies additional FP and latency |
| E | Second blocked pilot of selected B/C/D recipe | Gain replicates under the frozen promotion rule |
| F | Optional record sampling or alternate detector | A measured remaining failure mode warrants one more intervention |
| G | Frozen final fitting, inference and submission | Artifact, metric and transfer checks pass |

Run one P100 job at a time, after the current v3 run. Reuse verified outputs only
where membership and source hashes permit it. No Mask2Former, larger encoder,
custom CUDA extension or foundation-model fine-tuning is a prerequisite.

Start the stable local path at batch 2, accumulation 4, FP32, AdamW lr 1e-4,
weight decay 1e-4 and gradient clipping at 2. Treat FP16 AMP as a controlled
training-only profile after the FP32 path is verified; do not assume a Tensor Core
speedup or use BF16. Freeze pretrained encoder BatchNorm running statistics and
use GroupNorm in newly added decoder layers; accumulation does not fix noisy
batch-two normalization statistics. Preserve the proven detector/global schedules in controls.
For the new local path, use a predeclared 10,000-optimizer-update pilot with a
500-update warmup and cosine decay; select the last checkpoint, not H's best
epoch. Freeze this budget after a 100-update training-only profile, before any
held-out metric is read. Adjust the proposed budget before launch if measured
cost makes it infeasible, recording the change.

Profile full-ROI inference on the largest ROIs as well as typical cases. A
2048-square ROI can require 25 windows at these settings. Batch windows, retain
only necessary tensors, and release detector memory before refinement. Bound
wall-clock runtime rather than silently truncating hard examples. Persist
checkpoints and candidate caches across independently completed kernels; verify
resume from downloaded output before relying on recovery.

Planning envelope, **not a measured ETA**: budget roughly 8–16 P100 hours for a
complete first pilot including its inner fits and cached alternatives; a second
pilot and final artifacts can bring the campaign to roughly 24–48 GPU hours.
Replace these ranges after profiling using
`updates * measured_seconds_per_update + preprocessing + validation + export`.
Split work into jobs below the current nine-hour pipeline ceiling; verify actual
Kaggle quota and session limits at launch. Stop ineffective branches early at
their predeclared comparison, not by peeking at H every epoch.

Every training stage must reject a non-finite loss or gradient before the
optimizer advances, verify parameters at checkpoints, retain the last finite
checkpoint, and reject non-finite or all-empty inference. A remote kernel marked
`COMPLETE` is infrastructure status only; artifact integrity decides whether the
experiment completed successfully.

## 8. Final fitting and submission policy

Select architecture, schedules and assembly before final fitting. Generate clean
quality-training outputs for the selected architecture on training-side excluded
images; a new detector/refiner requires new compatible quality evidence. Fit the
validation-trained candidate and preserve it as an artifact with measured PQ.

Before a 601-image refit, simulate adding held training blocks within development
data and check score calibration and schedule transfer. The earlier 0.32 to 0.30
public regression is evidence that more training data alone is insufficient.
Record both optimizer-update count and image exposure; preserve the chosen
schedule convention explicitly instead of compressing a selected epoch into a
different cosine schedule. Retain the validated model if refit transfer is poor.

Infer all 180 test images with the frozen recipe. Verify filename coverage,
including zero-instance images, original-coordinate geometry, non-overlap, RLE
round trips and deterministic ordering. Save source, weights, quality schema,
threshold, per-image counts, CSV hashes and run IDs. The eventual authorized
submission should record its returned reference and completion status. Check
current daily allowance immediately before upload; historical remaining counts
are not reliable. Test pixels never provide tuning labels or prevalence targets.

The first implementation milestone is **B followed by C on one blocked pilot**.
That directly tests whether realistic prompts and native reconstruction produce
better masks. Crop recovery earns its place afterward. This provides a path to
a substantial improvement while making it possible to stop when the evidence
does not support one.

## Research and implementation status

Primary-source comparison and limitations:
[native reconstruction research](../reports/native_reconstruction_research.md).
Earlier failure-mode review:
[competitive review](../filament-oof-consensus/REVIEW.md).

This directory implements the reusable native geometry, proposal matching, local
model, historical assessment, and Kaggle orchestration used by the exploratory
diagnostic. The diagnostic uses binary cached coarse masks and 4,000 optimizer
updates to fit within one P100 session; those are recorded simplifications, not
the 10,000-update blocked pilot. Split/lineage preparation for R/Q/H, matched
realistic-prompt controls, global probability generation, quality fitting,
subgroup reporting, and the second blocked assessment remain required before a
GPU campaign can claim to implement this full specification.
