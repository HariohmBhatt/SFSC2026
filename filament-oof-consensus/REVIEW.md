# Competitive review of the proposed cascade

Review date: September 7, 2026. Scope: the adjacent APPROACH.md, the completed
YOLO/refiner source and saved reports, and primary sources cited below. This
review proposes revisions; it does not launch or alter a training run.

## Verdict

The approach has a credible route past 0.38 public PQ, but is not ready to serve
as an execution specification or evidence of a winning model. Keep realistic
proposal training, native refinement, and learned mask quality. Correct the
evaluation and target definitions, then prioritize experiments by measured
incremental PQ per GPU hour. A larger combined architecture is not itself proof
of better generalization.

The 0.465 promotion target and 0.49 stretch target are intermediate milestones.
The previous leaderboard snapshot's leader was 0.56; that is historical, not a
fresh ranking. At the control SQ of 0.68437, achieving PQ 0.56 would require RQ
about 0.8183 on the same evaluation population. A fixed subtraction of 0.055 from
future validation results is not a valid score forecast: architecture, training
membership, selection history, and the unknown public subset all change it.

## Required corrections

### 1. Make exclusion apply to every stage

The proposal says to generate detector OOF predictions, fit a refiner on them,
and cross-fit a utility model. Detector OOF status alone does not make the
refiner's training-image masks or quality predictions held out. A scorer can
learn from artificially clean masks if the refiner has seen those targets.
Likewise, repeatedly choosing detector checkpoints against an OOF assessment
partition turns it into selection data.

For an outer assessment set H, all trained components, upstream generators of
training features, checkpoint selection, and calibrators must exclude H. Inside
the remaining pool, generate detector OOF prompts and fit refinement. Generate
quality-training features with refiners that exclude their feature images, or
reserve a separate quality-training partition. A raw OOF cache shared across
outer folds can violate this: models that generated other folds' training
features may have fitted H. Record component-level train/selection/assessment
membership, not only a single image manifest.

An affordable first pilot is a fixed train-side fit/quality/assessment separation
within the 495 images, with the assessment partition excluded from all upstream
training. Rotate only if it shows a useful gain. The 106 historical validation
images remain a comparison set, not a newly untouched test. Any new temporal
assessment is prospective training exclusion, not erasure of prior development
knowledge. Compare the old recipe retrained on the same reduced membership.

### 2. Specify temporal blocks, not an optional purge

“Grouped by observing date and site” can put the same solar event at different
sites into different folds. Use fixed contiguous date intervals shared across
sites, with a deterministic seven-day purge. Publish counts and hashes before
training. Do not drop the purge when remaining counts are inconvenient; change
the block design before inspecting performance. Use fourteen days as a
sensitivity experiment. Bootstrap blocks as well as JPEGs; a JPEG bootstrap
alone ignores dependence across nearby observations.

### 3. Separate duplicate suppression from object presence

The instruction to treat only the best duplicate assignment as the instance can
give a valid filament an empty target merely because a second proposal exists.
A refiner that sees one prompt cannot reliably infer that set-level competition.

Train each recoverable duplicate against its matched instance mask. Reserve
empty segmentation targets for verified background proposals; handle ambiguous
annotator omissions with explicit uncertainty. Train duplicate selection at the
candidate-set stage. Measure quality separately for seed and refined masks,
reconstructed at native resolution with a detached target. Recompute matches
after final overlap assignment, which can materially change a mask's IoU.

### 4. Define the quality objective mathematically

Choose either direct expected contribution
`u = E[IoU * 1(IoU > 0.5)]` or `p * q`, where p is match probability and q is
expected IoU conditional on matching. An IoU target already zeroed for unmatched
examples is unconditional contribution; multiplying it by p again double-counts
non-match risk. Define targets separately for the seed and refinement, and
average annotation records consistently with the official evaluator.

For one-to-one PQ, `PQ = 2*S/(G+P)`. For an isolated, nonconflicting extra
prediction with expected contribution u, the plug-in objective increases when
`u > S/(G+P) = PQ/2`. This is a useful diagnostic, not an independent selection
rule for interacting masks or the organizer's overlap edge cases. Candidate
conflicts, repeated annotation weighting, and pixel assignment require scoring
the complete set. Avoid arbitrary per-image instance quotas.

Explicit learned mask quality is well motivated by
[Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241), but its COCO AP gains do
not establish an SFSC PQ gain. Begin with one regularized quality model. Add
isotonic calibration only if independent image-level assessment supports it;
1,723 candidates are not 1,723 independent observations.

### 5. Audit the apparent selection ceiling before spending on architecture

The 0.836 number lets every truth choose a candidate independently and uses truth
to choose between seed and refinement. The 0.584 number is realized recall after
selection. Their difference includes low-confidence false-positive tradeoffs,
conflicts, annotation disagreement, and suppression. It is not 0.252 of freely
recoverable recall, nor a forecast of achievable PQ. Weak confidence/IoU
correlation alone does not establish that missing ranking information is
learnable from the image.

Reproduce 0.435179 from cached predictions, then separate losses from proposal
absence, refinement damage, score rejection, duplicate suppression, and overlap
painting. Report precision/recall and PQ versus confidence. Compute one-to-one
best-available-mask diagnostics per record, explicitly retaining their optimistic
status; also inspect which records demand incompatible predictions for one JPEG.
Use train-side held-out candidates for learned choices. Existing validation
caches support diagnostics, not new supervised quality training presented as an
independent validation result.

### 6. Test native crops before committing to Mask2Former

Native resolution is a plausible source of complementary recall. The document
has not shown that query-based attention is the necessary source. First test
native-crop inference with the existing train-only detector, then a standard
pretrained crop detector if needed. Measure incremental matches at the same
false-positive budget and complete PQ after reconciliation. Only then compare
Mask2Former under comparable training compute.

The [official Mask2Former repository](https://github.com/facebookresearch/Mask2Former)
is archived. Its [installation contract](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md)
needs a real compatibility test on the pinned P100 runtime. A custom P2
segmentation head is another architecture change, not an automatically supported
fallback with fully matching pretrained weights. State its configuration and
weight mapping before budgeting it.

The proposal also specifies mask NMS without distinguishing it from the
detector's earlier box NMS. Trace both: a whole filament suppressed upstream
cannot be recovered by a later mask scorer. Confirm the pinned implementation,
candidate cap and mask export before changing thresholds; the
[Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/)
describes separate NMS and mask-resolution controls.

### 7. Turn the refiner into a controlled experiment

Changing ResNet18 to ResNet34, 384 to 512, prompt channels, annotation sampling,
six losses, and scoring together hides which intervention works. First keep the
current refiner and BCE/Dice objective, changing only realistic prompts and
quality targets. Then add native geometry. Change backbone and auxiliary losses
only after those comparisons.

The [CascadePSP source](https://arxiv.org/abs/2005.02551) supports global/local
refinement, but the proposed five-channel local crop has no defined global
context path. Specify a whole-ROI context pass, how its prediction/features enter
each native tile, stride, padding, blending, and image-border handling. Apply the
native-size condition to the expanded ROI: a 400-pixel box at 1.75 context already
spans 700 pixels. Train on the same tiling geometry used at inference. TTA guards
can reduce damage; they do not guarantee protection against fragmentation.

### 8. Replace the fixed runtime and diagnostic gates

The old 384-pixel ResNet18 refiner takes roughly 313–322 seconds per epoch.
Eighteen epochs already approach 95 minutes before adding pixels, a deeper
encoder, more OOF proposals, or additional excluded-fold fits. A 512 crop has
1.78 times the pixel count. The proposed 90-minute allocation is therefore not
supported. The global detector OOF estimate of about 3.56 hours is reasonable
only under the old per-image throughput and excludes further nesting. Native
crop training, refiner cross-fitting, and final fits require their own budgets.

Record seconds per optimizer step, steps per epoch, candidate count, actual
batch/accumulation, peak allocated and reserved memory, and install/export time.
Budget the complete experiment across kernels after profiling. Save artifacts
between kernels; a local checkpoint inside a failed kernel is not by itself a
verified recovery mechanism.

Promote by exact PQ and reproducibility. Mandatory SQ/RQ minima and +8-point
large-object recall can reject a genuinely better PQ tradeoff. Retain those as
diagnostics. A bootstrap lower bound of -0.005 does not establish improvement.
Require either a positive paired block-level interval or a predeclared second
replication with consistent gains. Sparse site scores cannot support rigid
per-site thresholds. Explain official/one-to-one differences on ambiguous truth
overlaps rather than assuming non-overlapping predictions force parity.

## Revised order with the best chance of progress

| Order | Experiment | What earns the next GPU spend |
|---|---|---|
| 0 | Cache reproduction and error decomposition | Identify a loss source that can be corrected with available features |
| 1 | OOF-trained quality selection; keep existing mask models | Better complete PQ on excluded images; recover low-score truths at controlled FP cost |
| 2 | Realistic prompted training and explicit native/global refinement | Better seed/refined choice and mask IoU without added merges |
| 3 | Standard global plus native-crop proposals | Additional correct instances at useful marginal PQ per inference minute |
| 4 | Query model or alternate backbone challenger | Evidence of complementary errors under a matched compute budget |
| 5 | Two-model consensus and guarded TTA | Repeated paired improvement after overlap resolution |
| 6 | Frozen final fitting and one submission | Full artifacts verified, lineage and schedules preserved |

Sample complete annotator records during training as a distinct ablation against
the current fixed-medoid target. Do not additionally multiply by annotator
agreement without checking metric weighting: that can disproportionately suppress
faint real objects. Mine negatives across all available training records.

Before a final 601-image refit, compare the selected validation-trained artifact
with refitting policies on a training-side simulation. The earlier 0.32 to 0.30
full-refit regression is evidence that more training data does not guarantee
better public PQ. Preserve original schedule progress and optimizer-step counts;
do not recreate a shorter cosine schedule merely because an intermediate epoch
was selected. Final utility models also need held-out refiner features, not only
held-out detector proposals. Retain both validated and full-fit artifacts.

This revised order supports a credible next improvement and a longer winning
campaign. Winning remains an empirical objective: each additional component must
close a measured error category and survive a complete external submission.
