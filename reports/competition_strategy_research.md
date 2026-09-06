# Solar Filament Segmentation Challenge 2026: evidence and strategy

Research date: **2026-09-06 (UTC)**. This report combines a source audit of the dashboard's public-notebook inventory, current first-party web results, the organizer's cached evaluator and discussion replies, original research papers, and this repository's completed first experiment. It proposes experiments; it does not claim an untrained method will win. Research performed by a dedicated background research agent, independently of the agent implementing and launching training.

## Decision

**Start a precision-focused continuation of our native-resolution model now. Build toward a clean pretrained detector plus conditioned native crop refiner, with the dense model supplying missed-instance proposals.** The immediate experiment addresses an observed failure, while the eventual cascade addresses the structural limitations shared by public solutions.

Our completed validation run has **PQ 0.28775, matched-mask SQ 0.67486, TP 766, FP 1,590, FN 471**. The primary problem is recognizing the right instances and avoiding fragments, rather than uniformly poor boundaries. The report contains 322 fragmented-truth cases versus nine merged-prediction cases. A larger network or more aggressive watershed is not the first remedy. [Completed experiment](../outputs/first-model/experiment/report.json)

There is no defensible promise of first place. Some apparent public breakthroughs are old metric scores, private-weight inference, or replayed predictions. A useful competitive target is independently reproducible **validation PQ above 0.35**, then above 0.40 with gains surviving grouped out-of-fold evaluation. These are engineering promotion targets, not predictions or guarantees. A leaderboard number cannot substitute for those experiments.

## 1. What problem are we actually solving?

### Physical signal and label geometry

The input is an H-alpha solar observation. A model must identify dark filament structures while distinguishing them from other dark features, uneven illumination, site-dependent imaging, and background artifacts. The scientific target includes thin branches and barbs, with one output mask per filament. The MAGFiLO authors describe annotations generated through human review; their reported inter-annotator Kappa of 0.66 measures annotation agreement, **not an achievable competition PQ ceiling**. Their original dataset's observation-record counts differ from unique image counts, which reinforces the need to distinguish records from source pixels. [MAGFiLO dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11437148/), [first-party dataset description](https://www.mlecofi.net/magfilo)

Our local audit establishes 707 unique labeled JPEGs, 180 competition-test JPEGs, and 1,154 annotation-image records containing 8,199 instances. All local JPEGs are grayscale 2048×2048. Median filament area is 1,228 pixels; the 1st percentile is 209 pixels. At 512×512 full-frame input, those become approximately **77 and 13 pixels**. The objects' narrow parts shrink by four in each direction, so an otherwise accurate resized semantic map can lose the geometry required to cross IoU 0.5. [Local data assessment](data_assessment.md), [dataset contract](../DATA.md)

Native resolution is therefore a strong starting constraint, but not a prohibition on a coarse detector. A detector can locate context at 1024–1280, while the refiner rereads original pixels. This arrangement spends high-resolution compute where objects occur. Long filaments need larger or elongated crops: blindly mapping every square crop to 256 pixels reintroduces the resolution loss.

### Annotation disagreement is central

296 source images have several annotation sets; 256 of those disagree on filament count. Sampling complete annotation sets avoids training on duplicated interpretations simultaneously, but it does not eliminate the contradiction between labels. One annotator can omit a real-looking structure or regard two connected structures as one. The deployed model must return a single interpretation, which is scored against several interpretations. [Local data assessment](data_assessment.md)

Recommended treatment:

1. Keep every version of a JPEG in the same split. The parent agent's September 6 filename/timestamp audit found **22 observing dates crossing partitions (45 images)**. Of 106 validation images, 26 have a training observation within one day, 69 within three days, and 99 within seven days. These are temporal-proximity counts, not proof of identical filament content. They nevertheless show that exact-file grouping alone does not establish independence for persistent solar structures. Preserve the historical split for like-for-like comparison, then add observation-event or 7–14-day blocked validation with a purge gap and sensitivity analysis inside the development pool; do not use the consumed internal test to choose the blocking rule.
2. Train on one complete annotator record at a time, or on separately sampled complete records. Preserve all categories as filament foreground; chirality ambiguity does not make an instance background.
3. Compare record-uniform sampling with filename-uniform sampling. The organizer scores annotation records, so a three-annotator source contributes three times to its aggregate. Uniform filenames are a reasonable first sampler but not exactly that weighting.
4. Use inter-annotator IoU correspondences on training data to characterize stable cores and uncertain boundaries. Auxiliary soft semantic targets may average opinions; the **instance target must remain a coherent annotation set**. Do not union every annotator into an instance ground truth.
5. Report pairwise human agreement as a reference, not as a hard upper bound. A consensus prediction may agree with several annotators differently from any one annotator.

Hard-negative mining needs particular care: an apparent false positive for one annotator may be labeled by another. Prefer patches negative across all available training annotation sets for the strongest negative mining, or reduce their mining weight when annotators disagree. Never mine validation masks into training examples.

### The metric rewards correct objects

The organizer's self-evaluation source uses IoU strictly greater than 0.5 and aggregates IoU sums, false positives and false negatives over every annotator-image. It does not average per-image PQ. Non-overlapping predictions are required by the organizer. [Official evaluator notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), [cached source](competition_watch/cache/azimahmadzadeh__self-evaluation-notebook/source.txt), [organizer overlap clarification](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735377)

For one-to-one matching, the denominator simplifies to `0.5 × (number of GT instances + number of predicted instances)`. Thus:

`PQ = 2 × sum(matched IoU) / (N_ground_truth + N_prediction)`.

This explains several otherwise surprising outcomes:

- Splitting a perfect instance into two equal halves produces no match: both IoUs are exactly 0.5.
- Splitting it into 60% and 40% gives one match of 0.6, plus a false positive; PQ becomes 0.4 for that example.
- Merging two equally sized disjoint objects produces IoU 0.5 to each, again giving no matches.
- Adding a fine branch can increase overlap, but also create an unwanted bridge and destroy instance recognition.
- Small instances count individually, so large-object semantic Dice can hide missed small targets.

These are algebraic deductions from PQ, not measured competition experiments. The underlying metric and its segmentation/recognition factorization are documented in the original panoptic segmentation paper. [Panoptic Segmentation](https://arxiv.org/abs/1801.00868)

The cached organizer implementation counts **all** pairs above 0.5 rather than explicitly running Hungarian assignment. Our evaluator enforces one-to-one assignment. They agree for non-overlapping instance partitions, but overlapping ground truths can break the uniqueness assumption. Before final submission, compare both evaluators on actual validation masks and document any difference. Do not alter labels to manufacture parity. Empty cases, strict equality, omitted image predictions, and RLE round trips also need explicit agreement. This is a source-level observation, not a claim that the current server remains defective.

## 2. Which public scores and permissions are current?

The official overview says the metric changed on August 7 and the leaderboard was rescored on **August 12, 2026**. Its August 20 announcement warns about attempts to manipulate the system and says scores above 0.35 are scientifically valuable. Final judging also considers qualitative evidence and reproducibility; the stated final report/solution deadline is November 15. [Official overview and announcements](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/citation), [final submission details](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/organizers)

The initial dashboard snapshot used here was fetched **2026-09-05T15:09:48Z**: 58 public notebooks, 20 discussion topics, one model entry, and a leaderboard headed by 0.56 with many 0.55 scores. The parent agent's fresh **September 6, approximately 05:20 UTC** refresh again found 58 notebooks and a top score of 0.56. These are dated snapshots, not predictions about future rank. [Dashboard store](competition_watch/store.json)

There is direct evidence of stale score labels: Fususu's notebook still has **“LB 0.70” in its title**, and its source dates that result to August 5. The current first-party notebook and log pages accessed September 6 display **0.34**, while an older comments crawl and model-card text retain 0.70. Consequently this method is evidence for a useful cascade, not for current PQ 0.70. [Current notebook](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff), [current log page](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff/log), [cached source](competition_watch/cache/phuongncn__lb-0-70-topology-safe-solar-filaments-handoff/source.txt)

The empty-mask 1.00 notebook is a historical responsible disclosure, with no trained segmentation model. Organizer replies dated August 20 explicitly say the issue was addressed. Its score must not enter an architecture ranking. [Disclosure notebook](https://www.kaggle.com/code/artkomissar/please-fix-the-leaderboard), [organizer response](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728474)

Some original MAGFiLO public labels overlap competition-test imagery according to participant disclosures. The organizer emphasizes reproducible real segmentation rather than such submissions. Do not import original/public test annotations or use published filament-specific weights without examining training provenance. The overlap allegations do **not** prove that any named top team used leakage. [First-party competition discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062)

### Updated organizer permissions

These statements were read in the dashboard's cached first-party replies fetched September 5. Direct browser rendering of some discussion pages was empty, so these are explicitly **cached organizer clarifications**, not newly retrieved complete rule text:

| Organizer reply | What it permits | Source |
|---|---|---|
| Azim Ahmadzadeh, 2026-08-21 14:23:55 UTC | Supplied `spine`, `bbox`, `area`, and `category_id`; forbids external information leaking test ground truth | [Topic 736543](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/736543) |
| Azim Ahmadzadeh, 2026-08-24 15:40:32 UTC | Public additional **unlabeled** GONG observations for self-supervised pretraining | [Topic 737215](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/737215) |
| Azim Ahmadzadeh, 2026-09-02 14:11:45 UTC | Original GONG FITS images | [Topic 738951](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/738951) |

The older cautious statements in `DATA.md` and `model_evidence.md` are therefore superseded on supplied metadata. This first continuation does not need new supervision. A later spine auxiliary head is permissible in principle, while inference still must not rely on unavailable ground-truth metadata. Generic ImageNet/COCO weights are a cleaner transfer-learning starting point than an undocumented model trained on the original MAGFiLO release.

## 3. Public approaches: what is real evidence?

The inventory was scanned and the important families read at source level. Scores below retain their evidence type. No public model was retrained or independently benchmarked by this research agent.

| Public family / representative | Mechanism visible in source | Evidence and limitations | What to retain |
|---|---|---|---|
| [Nomannic19 Mask R-CNN + refiner](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner) | COCO-pretrained Mask R-CNN/FPN, grouped source-image split, ResNet18 crop U-Net; threshold search | Current page reports **0.33**. Full detector/refiner code is useful, but crop Dice is not full-frame PQ and our manifest differs. | Separate detection from boundary recovery; evaluate the complete cascade. |
| [Fususu topology-safe handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff) | 1024 detector, 256 refiner, guarded four-flip TTA and conservative seed splitting | Historical **0.70 → current displayed 0.34**. Frozen model hashes ensure identity, not validation independence. | Guard TTA against fragmentation and retain exact provenance. |
| [Timgluhkikh YOLO11s → crop U-Net](https://www.kaggle.com/code/timgluhkikh/best-score-magfilo-yolo11s-detect-crop-u-net) | 1280 YOLO11s; 512 ResNet34 refiner with grayscale/CLAHE/box prompt | Visible run loads existing weights. Source mentions local PQ near 0.409 but no reproduction here. Its prose and `USE_BOX_JITTER_TTA=False` configuration disagree. | A spatial prompt tells the crop refiner which nearby object to segment; audit code rather than titles. |
| [Ektarr YOLO + U-Net](https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation) and [new metric pipeline](https://www.kaggle.com/code/ektarr/new-metric-solar-filament) | Coarse detector plus conditioned refiner; newer source has grouped five-fold training, medoid annotators and explicit PQ | Old search result 0.69 is stale-era evidence. New pipeline source exists, but cached notebook outputs contain no executed PQ result. | Grouped OOF, all-annotator scoring, detector/refiner training mismatch control. |
| [Sivapnlr80 five-fold Mask R-CNN](https://www.kaggle.com/code/sivapnlr80/filament-maskrcnn-kfold-v2) and [agreement inference](https://www.kaggle.com/code/sivapnlr80/solar-filament-segmentation-improved-pipeline) | 1024/1280 detectors; cross-fold mask-IoU clustering; require multiple-fold support; non-overlap painting | A retrieved old page shows 0.60, incompatible with the dated leaderboard and unsuitable as a current target. Validation sweeps using **all** fold models on a separate random holdout are train-contaminated for models that saw those files. Some versions filter categories 1/2 and omit valid category 3. | Match instance masks before averaging; every validating model must have excluded that image. |
| [HDJoJo inference](https://www.kaggle.com/code/hdjojo/solar-filament-seg-inference) | YOLOv8l segmentation at 2048, score-ordered overlap removal | Current page displays **0.55**, but training is in a private dataset. Visible code takes ordering values from `.boxes.cls`, not `.boxes.conf`; for one class that is not confidence ordering. | Native-resolution direct instance segmentation is promising, but this does not establish a reproducible training recipe. |
| [Lâm Huy “U-Net 0.55+”](https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55) | Source includes a small U-Net training loop, then decodes an embedded compressed `CHAMPION_PAYLOAD` CSV for final test RLEs | The final submission comes from the embedded prediction table, **not the visible trained U-Net**. No assertion is made about where those predictions originated. | Exclude its 0.55 headline from evidence that this U-Net training works. |
| [AnthonyTherrien native U-Net](https://www.kaggle.com/code/anthonytherrien/magfilo-blended-tile-u-net-for-solar-filaments) | Despite the tile title, cached latest source uses one 1888 crop, EfficientViT encoder, BCE+Dice, dihedral TTA | Page shows **0.33, best 0.36**. Source unions annotator masks and reports unsuccessful topology/instance-loss variants. Its bf16 execution is not a P100 recipe. | Preserve resolution, correct positive-weight calibration, inspect actual latest code. Do not assume topology loss automatically helps. |
| [Grouped-CV strategy writeup](https://www.kaggle.com/code/dattadhebe/solar-filament-grouped-cv-and-strategy-writeup) | Native crops, grouped folds, OOF blending and instance diagnostics | Claims penalized Dice 0.4958, not official PQ; historical LB 0.66 is not a present benchmark. A writeup alone is weaker than checkpoint/prediction artifacts. | Clear promotion gates, grouping, error accounting. |
| [RF-DETR-Seg + PTL](https://www.kaggle.com/code/jirkaborovec/filamentsegm-eda-rf-detr-seg-ptl) | GroupShuffleSplit, pretrained transformer instance model, learned crop refiner | Source describes expensive training and selection by mAP/Dice proxy rather than final PQ. No P100 reproduction here. | Useful alternate proposal generator after a cheaper detector establishes value. |
| [EdgeAttNet Kaggle adaptation](https://www.kaggle.com/code/koushikkumardinda/solar-filament-segmentation-2026-edgeattnet) | Attention/edge-informed U-Net | Current retrieved page shows **0.25**. Paper evidence concerns different masks/splits and mIoU; it does not establish leaderboard superiority. | Low-cost structural auxiliary cues, only if our ablation supports them. |
| [Early 0.9 U-Net](https://www.kaggle.com/code/pavloivanin/0-9-solar-filament-baseline-pipeline-u-net), [“LB 1st” U-Net](https://www.kaggle.com/code/rokaiyasomapti/lb-1st-solar-filament-segmentation-2026) | Resized semantic masks and simple postprocessing | Dashboard source audit flags annotation-lookup and empty-output paths. These need individual verification before running; their historical headline scores do not prove segmentation ability. | Basic architecture only; do not reproduce invalid parsing or old-metric selection. |
| [Classical dark-filament segmentation](https://www.kaggle.com/code/vmerckle/simple-dark-filament-segmentation-0-46) | Intensity-based computer vision and morphology | Title's 0.46 was not independently verified under current PQ. | Cheap darkness/local-contrast cues for proposal scoring and negative diagnostics, not an assumed champion. |

### Concrete validation-contamination checks

Reconstructing Timgluhkikh's **visible** Python `random.seed(42)`, shuffled sorted filename list and 15% validation recipe from our 707-file manifest places **88 of our 106 validation filenames in that recipe's training set**. This does not prove the reused checkpoints followed that recipe; it proves that simply attaching them cannot yield a defensible result on our holdout. The same concern applies to public “final” models trained on all competition training images. [Cached source](competition_watch/cache/timgluhkikh__best-score-magfilo-yolo11s-detect-crop-u-net/source.txt), [our manifest](../splits/split_manifest.csv)

The five-fold contamination issue is structural: each image in a five-fold training pool is in four models' training folds. An ensemble of all five cannot be called OOF for that image. Correct OOF uses its one excluded-fold model, or an ensemble whose **every member** excluded a common outer holdout. Likewise a full-data refiner can contaminate an otherwise held-out detector evaluation.

## 4. What the literature changes about the recommendation

| Primary source | Applicable lesson | Why it is not enough by itself |
|---|---|---|
| [DCAN](https://arxiv.org/abs/1604.02677) | Region and contour supervision can separate touching objects | The original task is glands; thin filament interiors can break into multiple seeds. Our present fragmentation evidence argues for restrained splitting. |
| [clDice](https://arxiv.org/abs/2003.07311) | Skeleton overlap can improve tubular connectivity | Foreground connectivity is not instance identity. False bridges can improve a connectivity surrogate while reducing PQ. Add only a small auxiliary term in a separate experiment. |
| [EdgeAttNet paper](https://arxiv.org/abs/2509.02964), [official code](https://github.com/dasjar/EdgeAttNet) | Learned image-edge information at the U-Net bottleneck is directly relevant to filament barbs | Authors report mIoU on MAGFiLO, not this competition's grouped PQ. Pretrained weights may have seen overlapping MAGFiLO imagery. Reimplement/train cleanly if tested. |
| [MORDEN, July 2026](https://arxiv.org/abs/2607.24525) | Multi-scale oriented features, DenseCRF, and clustering target filament morphology | Semantic results on MHAS/AHAS do not establish instance PQ. DenseCRF adds tuning/runtime, and clustering can join distinct instances. Treat as a later ablation, not a replacement for our measured reference. |
| [Ultralytics segmentation documentation](https://docs.ultralytics.com/tasks/segment) | Generic pretrained instance models support an efficient detector/segmenter starting point | Default natural-image augmentation, image scaling, masks, and confidence are not automatically suitable for thin solar structures. |

The best-supported principle is **coarse context plus fine spatial detail, with validated object selection**. There is no primary evidence in this review that a foundation model, larger transformer, or a particular fashionable loss will beat a clean detector/refiner on this P100 budget.

## 5. The experiment to start now

The completed model is a 4.66-million-parameter, width-24 residual U-Net with foreground/interior/boundary heads. It trained 60 epochs on 768 crops, batch two, 200 steps/epoch, with 70% foreground sampling and BCE positive weights `[4,4,8]`. Its best calibrated thresholds hit the **highest foreground threshold and largest minimum area in the offered grid**: foreground 0.65 and area 64. On the calibration subset, moving minimum area from 8 to 64 at fixed thresholds removed 116 FPs while retaining all 135 TPs. This is unusually direct evidence that the original search ended too early. [Recipe](../outputs/first-model/experiment/checkpoint/recipe.json), [calibration](../outputs/first-model/experiment/checkpoint/calibration.json), [loss implementation](../filament-resunet/pipeline.py)

The current FP/TP ratio is 2.08. Precision is approximately 0.325, recall 0.619 and recognition quality 0.4264. Holding TP, FN and matched IoU fixed, removing half the FPs would raise PQ from 0.2877 to roughly **0.370**. Removing all FPs would yield about **0.516**, still below the dated leaderboard's 0.56. These are counterfactual arithmetic illustrations; a real filter will also discard true detections. They show why reducing FPs is worthwhile but cannot by itself justify a first-place promise.

### Run specification

| Item | Proposed first continuation |
|---|---|
| Provenance | Initialize only from our existing train-only checkpoint; create a separate experiment and preserve the frozen first cycle |
| Data | Existing 495 train JPEGs; 106 validation JPEGs; sample complete annotation records; no internal-test reuse |
| Hardware | Single Kaggle P100 16GB; existing verified TensorFlow route; assert GPU identity and execute a gradient step |
| Geometry | Same width-24 model, native 768 crops, batch two, stride-512 blended inference |
| Duration | Start with 20 additional epochs × 200 steps, hard cap three GPU hours including validation; choose checkpoint by validation PQ, not the last epoch |
| Optimizer | AdamW; learning rate 1e-4 tapering toward 1e-5; weight decay 1e-4; gradient and loss finite checks |
| Loss | Preserve BCE+Dice/head structure for attribution; lower positive BCE weights toward `[1,1,2]` as an explicit precision hypothesis |
| Sampling | Reduce foreground-centered probability from 0.70 to about 0.50; retain random crops; after warmup devote a bounded fraction of background slots to high-scoring false-positive training crops |
| Hard-negative guard | Mine from train only; check all supplied annotator masks where possible; cap candidate-forward overhead and avoid selecting only the hardest samples |
| Postprocessing | Compare connected components, hysteresis growth from confident seeds, and current watershed with stronger seed/area constraints |
| Minimum area | At least `64,128,192,256`; do not permanently delete all sub-256 objects without validation, since real small labels exist |
| Thresholds | Initial foreground grid `0.50,0.65,0.75,0.85`; hysteresis low/high and interior acceptance must be calibrated separately after the probability distribution changes |
| Fair control | Retune the unchanged baseline over the same extraction grid; compare fine-tuned weights against that improved control |
| Artifacts | Checkpoint, manifest hash, source hash, config, seed, training log, probabilities or cached extraction summaries, full validation counts, runtime and VRAM |

The exact run configuration may be narrowed by the implementing agent for runtime; record the executed configuration as authoritative. Lower positive weights and mining together establish whether a useful continuation exists, but do not attribute the gain to either factor individually. Follow with matched single-change ablations.

**Executed first-run adaptation:** the implementing agent launched 40 additional
epochs × 200 steps, learning rate 1e-4 to 5e-6, BCE weights `[1,1,2]`, foreground
sampling 50%, and a three-hour training cap. Hard-negative mining is deferred to
an isolated follow-up. The 65-recipe extraction grid includes grow/confidence pairs
(.30,.55), (.45,.65), (.55,.75), (.65,.85), both extraction modes, four minimum
areas, and the exact old recipe as a control. See the immutable run identity in
[precision_training.json](precision_training.json) and the
[executed approach](../filament-precision/APPROACH.md). This is continuation of our
own trained model, not training of the proposed full detector/refiner cascade.

Keep hysteresis semantics clear: a low threshold grows a candidate's shape, while a high-confidence support criterion decides whether to keep it. Do not create one filament per disconnected high-threshold island. The same thin filament can have several bright/dark subparts. A candidate without convincing foreground support need not survive merely because a “fallback seed” was available.

The internal test was already consumed by cycle 001 according to current workspace state. It is not available for repeated decisions. Use validation for development and later nested/grouped evaluation within the permitted development pool; report the old internal-test result as historical only. Do not reset state to make it appear untouched.

## 6. Highest-upside complete approach

**Proposed architecture: contextual instance proposals + prompted high-resolution reconstruction + dense missed-instance recovery.** This is a ranked research hypothesis, not a reproduced system.

1. **Coarse proposal model.** Start from generic COCO-pretrained YOLO11s detection/segmentation, trained cleanly on our manifest. Use 1024–1280 full-disk input for context; inspect recall by small/long-object bins. Keep all filament categories as a single foreground class. Use boxes derived from allowed masks or the permitted supplied boxes. Avoid aggressive mosaic/mixup until a controlled ablation supports them.
2. **Prompted crop refiner.** Start with a compact ImageNet-pretrained encoder such as ResNet18/34. Input image plus detector box or coarse mask prompt; the prompt identifies the intended object among neighbors. Use 384–512 crops with aspect-preserving padding, native crops where possible, and a larger/tiled pathway for very long instances. Preserve detector/refiner probability maps until native-space reconstruction.
3. **Match training to inference.** Begin with jittered boxes and degraded masks derived from training annotations, then train on actual held-out detector predictions generated within the training pool. Include wrong proposals with empty targets and low-quality proposals with matched targets. A refiner trained only on exact GT crops is never asked to reject detector false positives and suffers a major deployment mismatch.
4. **Dense recovery proposals.** Use the precision-tuned native U-Net to find strong candidates not covered by the detector. Feed those through the same refiner/quality scorer. Add them only if validation shows positive marginal PQ; the dense model's historical FP burden forbids blindly unioning its output.
5. **Instance quality prediction.** Learn or calibrate a small candidate-quality score from training/OOF predictions: detector confidence, foreground/interior probability, shape support, local darkness, refiner agreement and estimated mask IoU. Use validation for final selection thresholds. Predict annotation-supported quality rather than merely object darkness or a fixed image count.
6. **Conservative mask combination.** Match duplicate proposals by mask IoU/containment before blending aligned probabilities. Box overlap alone is unreliable for long neighboring filaments. Avoid single-linkage chains that merge two objects through an intermediate candidate. Resolve residual overlap using confidence-quality painting; validate the resulting masks, not pre-paint masks.
7. **Topology-aware acceptance.** Optional flip TTA should improve a candidate without fragmenting it; keep the single-view result when topology/area checks fail. Splitting requires independent instance evidence, not just distance-transform peaks along one elongated filament.
8. **Small clean ensemble.** After one model is good, train 2–3 diverse seeds/folds with fixed outer-validation exclusion. Ensemble instances after matching. Only move to five folds if measured improvement justifies its P100 cost.

This proposal improves on public patterns in five concrete ways: honest common validation; native reconstruction of small/long objects; refiner training on realistic imperfect prompts; a controlled recovery route for detector misses; and object-level quality selection with calibrated fragmentation. Each is testable separately. The strategy is intended to improve recognition and shape together; it does not rely on unlabeled data being a guaranteed source of gain.

### P100 constraints

Our existing run verifies that 768/batch-two TensorFlow training fits, with approximately 11.5 GB allocator usage reported during the initial run. A 1024 crop has **1.78×** the pixels of a 768 crop and cannot be assumed to fit at the same batch. Batch one, smaller decoder width, or sequential branches should precede any reduction in native detail. The local smoke audit found Kaggle's then-default PyTorch binary lacked P100 `sm_60`, while a pinned PyTorch 2.7.1 CUDA 12.6 build worked. Recheck the actual image, pin a compatible stack, and run one real training/inference step before launching a new detector. [GPU verification](../GPU.md), [initial P100 training evidence](first_model_training.json), [official PyTorch version installation commands](https://pytorch.org/get-started/previous-versions/)

Do not copy T4/A100 bf16 or Tensor Core throughput expectations. Train detector, refiner and dense branch sequentially. Prototype a detector at batch one/two, profile an epoch, then determine the epoch budget from measured P100 runtime. The public YOLO recipe's batch eight and RF-DETR's reported T4 epoch duration do not establish P100 feasibility. Keep generic weights attached and dependencies pinned to avoid paying installation/download overhead repeatedly.

External unlabeled GONG pretraining and FITS normalization are **later** options. Self-supervised training can consume much of the available budget before producing any PQ result; alternate FITS preprocessing changes the deployment contract and needs paired image auditing. Neither should delay the first controlled continuation.

## 7. Ablations and promotion gates

| Order | Experiment | Main question | Required comparison |
|---|---|---|---|
| A0 | Same checkpoint, expanded area/threshold grid and CC versus restrained watershed | How much of the defect is extraction? | Full validation PQ/counts versus old 0.28775 |
| A1 | Warm-start precision continuation | Can better training reduce the FP burden without losing small-object recall? | Compare against **A0**, not only old extraction |
| A2 | Loss-weight-only and sampling/mining-only continuations | Which training intervention caused the gain? | Same training steps/seed/postprocessing search budget |
| A3 | Clean generic-pretrained coarse detector | Is proposal recall/precision better than dense candidates? | Per-size detection and full-mask PQ on same outer holdout |
| A4 | Prompted refiner with jittered GT then OOF predicted prompts | Does realistic refinement improve full-frame PQ? | Detector masks alone, GT-prompt refiner, predicted-prompt refiner |
| A5 | Dense recovery branch | Does it add true missed instances rather than fragments? | Marginal TP/FP/FN and net PQ of added proposals |
| A6 | Low-weight clDice or permitted spine auxiliary head | Does continuity improve without false bridges? | Split/merge counts and PQ; no promotion by skeleton Dice alone |
| A7 | TTA/2–3 model ensemble and annotation weighting | Are gains robust enough to pay their runtime cost? | Same outer holdout, instance matching, measured P100 time |
| A8 | Unlabeled GONG pretraining or alternate FITS preprocessing | Is domain transfer better than spending the same hours on supervised training? | Equal GPU-hour baseline and data-provenance exclusion audit |

Record **global** PQ, SQ/RQ, TP/FP/FN, small-instance recall, fragmentation/merging, site/multiplicity strata and runtime for every promoted candidate. Use paired bootstrap resampling by **filename**, keeping its annotator records together, to estimate uncertainty in score changes. Threshold selection itself uses validation and thus introduces optimism; predeclare modest search grids, retain the retuned baseline, and require later grouped replication before claiming broad superiority. Sparse yearly strata are diagnostics, not reliable standalone rankings.

The immediate deliverable is an auditable P100 training run and a fair validation comparison. The research program's outcome remains empirical: a new candidate earns promotion by better masks and better held-out PQ, then a reproducible competition submission establishes its actual leaderboard position.
