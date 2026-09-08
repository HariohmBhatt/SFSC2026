# Research: native reconstruction after Consensus v3

Date: September 7, 2026. Scope: a practical successor on one Kaggle P100,
using primary papers and author-maintained implementations. This is a design
recommendation, not evidence that a new model has already improved SFSC PQ.

## Recommendation

Keep the proven whole-disk YOLO proposal model and build a **proposal-conditioned
global/local refiner** with explicit full-object context and native-pixel local
inputs. First change the refiner's training prompts to held-out detector errors;
then add local reconstruction as an isolated experiment. Test sliced inference
with the existing detector for missing small objects before investing in another
backbone. Refit candidate quality from complete held-out pipelines whenever the
refiner or candidate sources change.

This choice follows the measured problems in [the previous research](next_model_research.md)
and the narrower scope of [the executable Consensus v3 approach](../filament-oof-consensus/APPROACH.md).
Consensus can improve selection among existing masks. It cannot produce a
filament branch that neither candidate contains, or recover an object for which
there is no usable prompt. Reconstruction and proposal coverage attack those
separate limitations. None of the papers below measures this dataset or its
multi-annotation PQ, so their gains must not be translated into SFSC forecasts.

## What the primary sources support

| Method | Useful mechanism | Decision for this project |
|---|---|---|
| CascadePSP | Coarse global structure followed by local high-resolution refinement | Adopt the explicit context-plus-detail principle in a small existing-stack model |
| PointRend | Predict selected uncertain points using coarse and fine features | Optional later optimization once dense local refinement establishes an accuracy target |
| RefineMask | Reintroduce fine spatial features during instance mask refinement | Supports preserving native detail; do not import its entire detector stack initially |
| SAHI | Overlapping sliced inference and slice-aware detector fine-tuning | Cheap proposal-complementarity experiment with existing weights before a second model |

**CascadePSP.** The paper addresses loss of boundary detail when low-resolution
masks are enlarged, using global and local refinement with an input mask as a
prompt. Its local processing supplies detail that interpolation cannot recover.
The author implementation distinguishes a global-only fast mode from full
refinement and exposes a resolution/memory control. It was originally tested on
PyTorch 1.0; that is historical compatibility information, not a failure claim
about today's environment. We should implement the relevant geometry with the
verified SFSC PyTorch stack and train on H-alpha examples, rather than assuming
natural-image pretrained refinement transfers unchanged. Sources:
[paper](https://arxiv.org/abs/2005.02551),
[author repository](https://github.com/hkchengrex/CascadePSP).

**PointRend.** The method adaptively predicts points using coarse predictions
and fine image features. Its released instance configurations produce 224-square
masks; using its name does not itself mean output is native to a large ROI.
For SFSC, uncertainty-focused refinement could eventually reduce local compute.
However, a confidently erased branch need not lie among the chosen uncertain
points. A recovery design therefore needs spatial coverage beyond uncertain
boundaries. This limitation is our inference from the selection mechanism, not a
negative result reported by the paper. Sources:
[paper](https://arxiv.org/abs/1912.08193),
[author project](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend),
[point-selection implementation](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_features.py).

**RefineMask.** The paper uses fine-grained semantic features and staged instance
refinement to reduce detail lost during pooling and downsampling. This supports
injecting image detail after the instance representation is formed. Its results
are standard-dataset mask AP, not SFSC PQ. The official repository lists PyTorch
1.5.0 and mmcv-full 1.0.5; it also advertises an improved implementation with
less than 11 GB for two training images. That memory figure is not a P100
measurement for our crop sizes or environment. This is a later architectural
challenger if a compact native refiner fails, rather than a required dependency.
Sources:
[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html),
[author repository](https://github.com/zhanggang001/RefineMask).

**SAHI.** Slicing gives an existing detector a larger effective view of small
objects, and slice-aware fine-tuning addresses the changed input distribution.
The paper evaluates object detection on aerial datasets. It does not establish
safe instance-mask merging for long solar filaments. Use slicing as a proposal
source, retain the global view, and evaluate the final mask set rather than
assuming higher proposal count improves PQ. Source:
[paper](https://arxiv.org/abs/2202.06934).

## Minimal architecture: proposed adaptation

The following is our engineering proposal, not a reproduction of any paper.
Start with the existing ResNet18 encoder and a small U-Net decoder; retain the
existing BCE/Dice loss until geometry is validated. A larger encoder, topology
loss, semantic recovery network and query detector are separate experiments.

1. For each seed, construct a padded, expanded ROI in native image coordinates.
   Preserve its aspect ratio when letterboxing to a 384-square global tensor.
   Inputs include the image and seed. Produce an object-specific global
   probability map. Record the exact affine transform and valid-pixel mask.
2. Use 512-square native-pixel local windows with overlap over that expanded
   ROI. Each window receives the native image, transformed seed, corresponding
   global probability crop, and its coordinates within the full ROI. Explicitly
   pool global encoder features into a context vector and condition the local
   decoder with it; a shared prompt alone is not an implemented global pathway.
3. The local decoder predicts an absolute object mask, so it can add pixels
   outside the seed. Blend logits with valid-pixel weights in native coordinates
   before thresholding. Preserve original seed and global-only alternatives for
   quality selection; use one resulting instance per prompt.
4. Cover the whole expanded ROI in the accuracy experiment. Do not restrict
   windows to seed foreground, coarse uncertainty or predicted boundaries. If
   the predicted object reaches the ROI edge, allow one predetermined expansion
   and rerun; record how often this happens. Expansion is bounded by the image.
5. Route small objects through one native window; large/elongated objects get
   multiple windows. A scheduling cap must be measured and declared. Silently
   downsampling large ROIs after calling the method native defeats the purpose.

The crop boundary is an explicit remaining recall limit. Even a native refiner
cannot discover unrelated missing instances without new prompts. Training must
use the same global/local coordinate transforms, padding and ROI expansion as
inference. A local target is the crop of the selected complete instance mask,
not the union of every dark feature in that tile.

## Training data and score lineage

Use actual detector outputs generated while their source JPEG/date block was
excluded from detector fitting. Include partial objects, overly broad prompts,
real false positives and valid duplicates. All annotation records for one image
stay together. Sample one coherent record for a mask target; do not union
incompatible annotations. Ambiguous low-overlap prompts need an association
rule established on training data: assigning every sub-0.5 seed to background
would train away exactly the partial objects the refiner should rescue.

The scorer introduces a second exclusion requirement. A quality-training
example must be generated by a refiner that also excluded that JPEG. Detector
OOF alone is insufficient once the refiner learns from real proposals. If an
outer assessment block exists, every upstream fit and prompt cache feeding its
assessed pipeline must exclude it. A shared cache made by models trained on the
outer block can leak indirectly even when those cache rows are removed.

Quality remains the direct expected matched-IoU contribution, including zero for
unsupported predictions; do not multiply it by a second presence probability.
Duplicate candidates may each have good local mask targets. Suppression and
pixel ownership determine set quality later. Existing v3 coefficients are an
ablation control, not automatically calibrated scores for a new mask generator.

## Missing-object recovery experiment

Before new detector training, evaluate the current detector on a fixed grid of
overlapping native crops plus the whole image. Export source tile, original
coordinates, confidence, full mask and whether the mask touches a tile boundary.
Run the same refiner on each prompt in original-image coordinates. Favor an
interior view when the same object is present in multiple crops; retain a global
candidate for long objects. Do not blindly union masks from intersecting boxes:
overlap alone does not establish that two filament fragments have one identity.

Measure per-record one-to-one coverage, candidates per image, true additional
matches, additional false positives, truncation frequency, and final PQ after
the same selection procedure. High loose best-of-candidate recall is only a
diagnostic. If slicing adds complementary matches with acceptable cost, train a
standard smaller YOLO crop branch with the same geometry. Its architecture can
stay standard; a custom P2 head is not a prerequisite for this experiment.

## P100 and promotion constraints

Use the verified current environment and profile real backward steps. Begin
local training at batch one or two, then accumulate to the control's effective
batch. Freeze pretrained encoder batch-normalization statistics or deliberately
choose a small-batch normalization policy; do not silently rely on noisy
per-batch estimates. Record allocated and reserved GPU memory, step time and
the distribution of windows per object. Process proposal and refinement models
sequentially. No paper above establishes the exact memory or runtime here.

Report mask-only change with fixed proposals first, then selection change, then
sliced proposals. Use exact additive official PQ plus the conservative one-to-one
audit; inspect any disagreement. Include size, elongation, site and time-block
statistics, and distinguish crossing IoU 0.5 from polishing already-good masks.
Use paired temporal-block uncertainty and a frozen historical comparison.
Neither historical validation82 nor training-side OOF scorer loss is an untouched
test. Predeclare an additional purged outer assessment before candidate-specific
development; report that it remains subject to historical reuse of this dataset.

Promote only after measured PQ improves reproducibly enough to justify the
additional training/inference cost. Keep the current 0.38 artifact and the
ongoing Consensus run available as controls. This proposal provides testable
routes to additional mask and proposal coverage; it does not establish a
leaderboard-winning score.
