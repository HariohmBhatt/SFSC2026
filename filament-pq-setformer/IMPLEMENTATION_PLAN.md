# PQ-SetFormer-P100 implementation and training plan

Status: implementation plan only, September 10, 2026. This document does not
launch training or authorize a leaderboard submission.

## 1. Outcome

Build, validate, fully train, and package a single-class query-based instance
segmenter for SFSC2026. The model is a P100-sized MaskDINO-R50 adaptation with:

- global 2048-to-1280 image processing rather than inference crops;
- one-to-one Hungarian set prediction with 64 filament queries;
- filament/no-object, mask, auxiliary box, and predicted-mask-IoU heads;
- low-weight foreground, boundary, and soft-clDice supervision;
- optional MP-Former-style query piloting during training only;
- an excluded-data match calibrator and expected-PQ utility score;
- mask-set selection and utility-weighted pixel ownership without box NMS or
  connected-component splitting;
- optional dihedral TTA and a second seed, admitted only after the single model
  wins independently;
- required offline-first W&B metrics, Tables, and Artifacts alongside the local
  experiment record;
- immutable Kaggle stages, artifact hashes, resumable controllers, exact PQ
  evaluation, and verified COCO RLE generation.

The implementation is complete when a clean checkout can prepare the frozen
stages, train them on Kaggle P100s, reproduce their reports from downloaded
artifacts, and create a locally verified submission from the selected full-data
checkpoint. Training success and model promotion are separate outcomes: a
technically complete model may correctly finish as `gate_failed`.

## 2. Decisions inherited from the repository

These are requirements, not tuning options.

- The official objective is global additive PQ at strict mask IoU `> 0.5`.
  Model selection never uses Dice, AP, validation loss, or mean per-image PQ as
  a substitute.
- The immutable comparison champion is the YOLO11l-seg 1280 + ResNet18 refiner
  result at historical validation PQ `0.4351790286733054`.
- Predictions are one non-overlapping mask per physical filament. A query mask
  may remain disconnected. Touching masks are not merged, and one query is not
  split into connected components.
- One physical JPEG is one training item. Every access selects exactly one
  complete annotation record. Masks from different annotators are never mixed
  into an artificial target set.
- A hard negative is negative only after checking every annotation record for
  that JPEG.
- The consumed 106-image internal test is never used again.
- The historic 106-image validation is useful only as secondary comparison
  evidence. New promotion evidence comes from prospective site/date-blocked
  partitions of the original 495 training filenames.
- Every stage records source, membership, data, config, environment, weight,
  candidate-cache, and output hashes. A Kaggle `COMPLETE` state is not evidence
  of successful training.
- Training and inference use one Kaggle Tesla P100-PCIE-16GB per job. CPU
  fallback, a missing `sm_60` kernel, non-finite updates, or a collapsed mask
  set is a hard failure.
- No stage uploads a leaderboard submission automatically.

The local annotation audit resolves the query-count question from the attached
proposal: the maximum complete annotation record contains 26 instances. Sixty-
four queries therefore leave 38 spare slots. Forty-eight queries are the first
memory fallback and still leave 22 spare slots; this fact and the dataset hash
must be rechecked during preparation.

## 3. Corrections and clarifications to the attached proposal

The attachment supplies the architectural hypothesis. The following changes
make it consistent with how this repository actually runs experiments.

1. **H is one-shot assessment only.** `checkpoint_metric: global_PQ` is valid
   only on a declared development partition. H cannot select an epoch, stop
   training, tune augmentation, select loss weights, or calibrate a threshold.
   Pilot training uses a fixed optimizer-step budget and the final checkpoint.
2. **Annotation-record sampling is mandatory from baseline.** It is not a later
   optional ablation because unioning contradictory instance sets violates the
   data contract.
3. **W&B logging is mandatory and offline-first.** Every local and Kaggle stage
   creates a W&B run containing config, scalar history, system measurements,
   evaluation Tables, and versioned Artifacts. The same event is written to the
   repository's JSON/JSONL record, which remains the recovery authority when a
   network or W&B service is unavailable. A stage cannot start expensive work
   if its offline W&B run cannot be initialized.
4. **Mask quality has one precise meaning.** The neural quality head learns
   `E[mask IoU | query assigned to a truth]` on matched queries. The excluded
   calibrator learns `P(official one-to-one match)` after duplicates and
   interactions are considered. Final utility is their product. Unmatched
   queries do not receive a contradictory continuous-IoU regression target.
5. **The P100 fit is measured, not assumed.** The official MaskDINO R50 memory
   figure is already close to 16 GB at its reference setup. Input size 1280 is
   the target, but the frozen run size is whatever first satisfies the measured
   `<14.5 GB` peak-memory gate after the declared fallback sequence.
6. **Full-data score calibration uses OOF predictions.** A calibrator fitted to
   in-sample predictions is prohibited. The final campaign creates excluded
   predictions for all 601 train-plus-validation filenames before training the
   single 601-image deployment model.
7. **The complete model is configurable; the final recipe is evidence-based.**
   Quality, piloting, structural heads, TTA, and ensemble code are implemented,
   but only interventions that pass their declared comparison are enabled in
   the selected final recipe.

## 4. Architecture contract

### 4.1 Input and target construction

For the core experiment:

```text
2048x2048 uint8 grayscale JPEG
  -> float32 [0, 1]
  -> configured raw or p0.5-p99.5 robust normalization
  -> full-frame resize to measured safe size, target 1280x1280
  -> one-channel ResNet-50
```

Masks use nearest-neighbor resizing. Boxes, foreground union, one-pixel inner
boundaries, and skeleton targets are derived from the selected segmentation
masks, never read from supplied box/spine/category metadata. Empty target sets
must be supported even though the present labeled records are non-empty.

The first controlled preprocessing comparison is `raw / 255` versus robust
per-image clipping. It runs at proxy fidelity with otherwise identical source,
membership, seed, and training budget. The winner is frozen before full-size
training.

ImageNet initialization is a Phase 0 rules decision. The competition overview
allows external data but also says models may not use other ground-truth
metadata for training, so the exact permitted interpretation must be recorded.
If approved, convert the generic R50 stem to one channel and hash the source
weights. If not approved, use the identical architecture from random
initialization. No COCO segmentation, public MAGFiLO, or unknown-provenance
checkpoint is eligible.

### 4.2 Network

The base network is:

```text
one-channel ResNet-50
  -> C2/C3/C4/C5 multiscale features
  -> MaskDINO multiscale deformable pixel decoder
  -> stride-4, 256-channel mask feature map
  -> 64 learned object queries, dimension 256
  -> six transformer decoder layers, FFN dimension 1024
  -> per-query filament/no-object logits
  -> per-query auxiliary boxes
  -> per-query mask embeddings and mask logits
  -> per-query conditional mask-IoU estimate
  -> shared foreground and boundary logits
```

The MaskDINO and Detectron2 sources are pinned to exact commits and recorded in
`vendor-lock.json`. They remain implementation details behind the model
interface rather than leaking Detectron2 configuration throughout the runner.
All source licenses and local patches are recorded beside the commit hashes.

The quality head receives the final query feature and a detached, mask-weighted
pool of the pixel-decoder features. It is a two-layer 256-unit GELU MLP ending
in a scalar sigmoid. Detaching the mask weights prevents quality regression
from steering the primary mask logits through the pooling operation.

### 4.3 Hungarian matching and losses

Hungarian assignment uses the filament class, point-sampled mask BCE/Dice, and
low-weight derived-box localization costs. It never sees the external
calibrator. Assignment and training-loss coefficients are separate config
fields even when initialized to the same values.

Initial full-model coefficients:

| Loss | Weight | Target |
|---|---:|---|
| focal filament/no-object | 2.0 | matched versus unmatched queries |
| point-sampled mask BCE | 5.0 | matched query mask |
| point-sampled Dice | 5.0 | matched query mask |
| box L1 | 1.0 | box derived from selected mask |
| GIoU | 0.5 | box derived from selected mask |
| conditional mask IoU | 1.0 | detached realized IoU, matched queries only |
| shared foreground | 0.20 | union of selected-record masks |
| shared boundary | 0.10 | derived one-pixel inner boundary |
| soft-clDice | 0.05 | matched instance mask/skeleton |

Deep supervision applies the core set losses to intermediate decoder layers
with explicitly recorded weights. Quality supervision initially applies only
to the final decoder output. Skeleton and boundary targets are cached per
annotation record so morphology does not consume training wall time.

The no-object coefficient, point sampling counts, oversample ratio, importance
sampling ratio, and auxiliary-layer weights are explicit and frozen in every
run. No loss is silently inherited from an upstream COCO config.

### 4.4 MP-Former-style piloting

Piloting is implemented after the standard query path passes FP32 and AMP
smokes. When enabled, the first three decoder layers receive perturbed coherent
GT masks with probability 0.5. Perturbations are limited to 1-3 resized pixels
of erosion/dilation plus mild boundary dropout. They never cut one truth into
separate pseudo-instances. The final decoder layer is always prediction-driven.

The base implementation and pilot implementation share the same inference
path. A checkpoint trained with piloting must load and infer with the feature
disabled, proving that it adds no inference dependency.

### 4.5 Augmentation

Default identity-preserving augmentations are horizontal/vertical flips,
optional 180-degree rotation, gamma 0.85-1.15, contrast +/-15%, brightness
+/-5%, small Gaussian noise, mild blur, and very mild JPEG recompression.
Scale jitter is limited to the measured safe 1152-to-selected-resolution range.

Mosaic, CutMix, Copy-Paste, ordinary random crops, and aggressive morphology
are excluded from the core experiment because they alter global identity or
recreate the crop-distribution failure already observed.

### 4.6 Candidate output

Inference exports every query above a deliberately permissive presence floor,
before final set selection. Each cached candidate contains:

- filename, query index, source seed/view, and checkpoint hash;
- filament probability and logit;
- predicted conditional IoU;
- mask entropy, native area, component count, and bounding geometry;
- pairwise overlap/containment summaries against stronger candidates;
- compressed COCO RLE at native 2048 resolution;
- transform provenance for TTA candidates.

Dense native float maps are not cached for the full dataset. Logits needed for
ownership are processed per image; compact low-resolution logits may be cached
only for bounded calibration sets with a recorded format and hash.

## 5. Calibration and final set assembly

### 5.1 Official-match labels

For every excluded image and every complete annotation record:

1. binarize all permissive query masks with the declared provisional threshold;
2. run the repository's exact strict-`>0.5` one-to-one matching over the whole
   candidate set;
3. label a query positive only if it owns an official match in that set;
4. record its realized matched IoU, otherwise zero;
5. average sufficient statistics across complete annotation records without
   merging their truths.

This makes duplicates that lose the one-to-one assignment negative examples.
It fixes the single-candidate/oracle-target weakness of earlier quality work.

### 5.2 Calibrator

Fit an L2-regularized logistic model on Q-fit/OOF-fit only. Initial features are:

```text
presence_logit
predicted_conditional_iou
log_native_area
mask_entropy
maximum_iou_with_stronger_query
maximum_containment_with_stronger_query
component_count_capped
```

Feature count stays deliberately small. Standardization statistics, feature
order, coefficients, regularization, fitting membership, training objective,
and input-candidate hash are serialized. The module refuses schema or hash
mismatches.

Candidate utility is:

```text
U = calibrated_P_official_match * predicted_E_IoU_given_match
```

A direct regressor for `IoU * 1(IoU > 0.5)` is implemented only as a predeclared
ablation, not blended opportunistically after results are read.

### 5.3 Assembly

For a single model without TTA:

1. filter only candidates below the calibrated utility threshold or numerical
   minimum-area floor;
2. suppress near-identical masks by mask IoU/containment, retaining the higher
   utility query; do not use box NMS;
3. upsample retained logits to 2048x2048;
4. permit ownership only where mask probability exceeds the selected mask
   threshold;
5. assign overlapping pixels by `mask_logit + gamma * log(U + epsilon)`;
6. remove only post-ownership remnants below the selected minimum area;
7. keep disconnected pieces owned by the same query in one mask;
8. encode non-empty masks as COCO RLE and round-trip check them.

Q-cal searches a bounded CPU grid over mask thresholds `0.40-0.60`, utility
quantiles, minimum areas `{0, 32, 64, 96}`, and a small declared gamma grid.
Search dimensions are staged: select utility/mask thresholds first, then test
area/gamma around that frozen result. This avoids a large joint overfit.

No forced fallback, image-level count target, threshold chosen from H, or
leaderboard threshold adjustment is allowed.

### 5.4 TTA and ensemble

TTA is disabled for all single-model promotion decisions. If admitted, use
identity, horizontal flip, vertical flip, and 180-degree rotation. Invert masks
before clustering by mask IoU and containment. Average aligned logits and scores
inside a supported cluster. Single-view candidates require a stricter calibrated
threshold. Pixel contact alone never creates a cluster.

A second seed is trained only after the first seed passes the blocked model
gate. Each seed must pass independently. Cross-seed fusion uses the same mask-set
logic as TTA; unmatched singleton masks again require stronger evidence. Union
ensembling and array-index averaging are prohibited.

## 6. Validation design and model gates

### 6.1 Research pilot

Reuse and generalize the already-tested filename-only blocked split logic from
`filament-native-reconstruction/core.py`. Before training, freeze two distinct
site/date-blocked pilots over the original 495 training filenames. The existing
first assignment yields:

| Role | Images | Permitted use |
|---|---:|---|
| R | 289 | network fitting only |
| Q-fit | 69 | external match calibrator only |
| Q-cal | 34 | assembly and threshold selection only |
| H | 87 | one-shot challenger decision only |
| purged | 16 | no use in this pilot |

Shared 60-day date blocks span all sites. R is more than seven days from Q/H;
Q-fit is more than seven days from Q-cal. The split module verifies those facts
and writes membership hashes. A second deterministic salt must produce a
different H set that is block-disjoint from the first H; it is frozen before
the first H result is opened. A 14-day purge sensitivity split is also generated
before results, but run only for the selected configuration.

Because the existing champion was trained on images that include pilot H, its
0.435 result is not the causal pilot control. Retrain the frozen YOLO/refiner
recipe on each pilot's R membership, apply equivalent Q-fit/Q-cal handling, and
score it on the same H. The 0.435 champion remains a historical target and
submission fallback.

### 6.2 Required reports

Every comparison reports:

- global PQ, SQ, RQ, TP, FP, FN, and matched-IoU sum;
- paired PQ delta and 2,000-replicate shared-60-day-block bootstrap interval;
- matched-IoU and Boundary-IoU distributions;
- proposal recall before calibration and final recall after assembly;
- instance-count error, fragmentation, merge, duplicate suppression, and
  ownership damage counts;
- performance by native truth area, skeleton length, site, date block,
  annotation multiplicity, and annotation-count disagreement;
- candidate reliability and TP/FP utility distributions;
- seconds per optimizer step, inference seconds per image, peak allocated and
  reserved VRAM, and artifact sizes.

The official evaluator and the conservative one-to-one evaluator must agree
within `1e-8` on promoted prediction sets. A disagreement stops promotion for
investigation.

### 6.3 Gates

The first pilot passes only if all are true:

- query model PQ gain over its matched-R control is at least `+0.015` on H;
- the paired blocked-bootstrap 95% lower endpoint is positive;
- RQ improves without a material SQ collapse;
- no site has a catastrophic regression attributable to empty/collapsed output;
- finite training, valid candidate coverage, metric parity, runtime, and memory
  gates all pass.

The selected recipe is then retrained from initialization on the second pilot's
R and must produce a positive H delta. Promotion requires the pooled paired
blocked-bootstrap lower endpoint to remain positive. The 14-day sensitivity run
must not reverse the direction of effect.

The historic validation comparison is then run for continuity. A practical
target is all-106 PQ at least `0.460` and non-calibration-82 PQ at least `0.455`,
but those reused images cannot override failed blocked evidence.

An explicit operator override may continue implementation/full fitting after a
failed gate for diagnostic reasons, but the record retains `gate_failed`, the
override reason, source hash, and failed assessment. It does not become evidence
of promotion and never triggers automatic submission.

## 7. Experiment sequence

All capabilities are implemented before the final campaign, but evidence is
collected in this order.

| ID | Frozen change | Question | Promotion evidence |
|---|---|---|---|
| E0 | matched-R YOLO/refiner control | What is the fair pilot baseline? | complete artifacts and valid H metrics |
| E1 | MaskDINO-R50, standard set losses | Does set prediction improve the object set? | PQ/RQ versus E0 |
| E2 | neural conditional-IoU head | Does query quality rank masks better? | calibration error and marginal PQ |
| E3 | excluded logistic match calibrator | Does PQ-aligned selection remove FPs profitably? | Q-cal choice, frozen H delta |
| E4 | MP-style piloting | Does it improve query use/faint recall? | matched pilot with fixed assembly |
| E5a | shared foreground + boundary heads | Do auxiliary maps rescue near-threshold masks? | matched IoU/Boundary-IoU and PQ |
| E5b | soft-clDice `{0,.025,.05,.10}` | Does topology help without merges? | PQ plus split/merge counts |
| E6 | robust normalization/augmentation winner | Does it improve site/date transfer? | proxy ranking, then one promoted full run |
| E7 | four-view TTA | Does support improve PQ without an FP wave? | post-win Q-cal/H comparison |
| E8 | second successful seed | Does fusion beat both seeds? | independent seed gates plus fused delta |

E1 is a valid complete inference model by itself. E2-E6 are isolated changes,
not one bundled run. An intervention that fails its declared comparison stays
implemented but disabled in the selected recipe.

Hyperparameter work uses successive fidelity:

1. run 8-12 sequential 1024-pixel proxies for 800-1,200 microbatches on R;
2. choose at most three configs from Q-cal global PQ and training stability;
3. promote those configs to measured-safe full resolution and 6,000
   microbatches;
4. open H once after every choice is frozen.

Search only the bounded ranges declared in the attachment: decoder LR, backbone
LR ratio, weight decay, no-object weight, mask/Dice/quality weights, clDice,
query count, pilot probability, gamma range, and gradient clip. Sequential local
run manifests are the execution authority; every proxy is also grouped in W&B
as one sweep-compatible study. A W&B Sweep controller may schedule the same
frozen trials when authenticated connectivity is available, but it may not
change the search space or launch concurrent P100 work.

## 8. Full 601-image training campaign

This phase starts only after the two blocked pilots select one immutable config.

### 8.1 OOF calibration campaign

Create three deterministic 60-day-block folds across all 601 train-plus-
historical-validation filenames, with a seven-day purge around every held fold.
For each fold:

1. train the selected PQ-SetFormer from the declared initialization using only
   that fold's allowed training membership;
2. use the fixed optimizer schedule and final checkpoint;
3. infer permissive query candidates for its held filenames exactly once;
4. verify every held filename is covered by exactly one model that did not train
   on it;
5. save compressed candidates, per-file sufficient statistics, progress, and
   checkpoint/config/source hashes.

Partition the complete OOF candidate pool by date blocks into `OOF-fit` and
`OOF-cal`. Fit the logistic match calibrator on OOF-fit and choose the bounded
assembly recipe on OOF-cal. There is no advertised pipeline accuracy on the
same OOF rows used to fit the calibrator; pilot H remains the promotion evidence.

Quality labels check every complete annotation record independently. Feature
creation, standardization, calibrator fitting, and threshold selection each
record their exact membership and candidate hashes.

### 8.2 Deployment fit

Train one fresh model on all 601 filenames with the selected architecture,
resolution, initialization, augmentations, losses, seed, and fixed 6,000-
microbatch schedule. Save at least restartable, last, and final checkpoints, but
do not choose among them using competition-test pixels or labels. The final
checkpoint is the deployment checkpoint unless a predeclared training-only
integrity failure requires the last known finite restart checkpoint.

The reference optimizer configuration is AdamW with decoder/head LR `1e-4`,
backbone LR `1e-5`, weight decay `0.05`, 100 optimizer-step warmup, cosine decay,
physical batch 1, accumulation 4, and gradient clip 0.5. Proxy results may
replace these values only before the final config is frozen.

The OOF calibrator and assembly recipe remain frozen. They are not refitted on
in-sample predictions from the 601-image deployment model. Report score and
instance-count distribution drift between OOF and deployment outputs as a
transfer risk; do not silently shift thresholds to make counts look familiar.

### 8.3 Competition inference

Run the deployment model on all 180 competition JPEGs. For each image:

1. verify grayscale shape and preprocessing identity;
2. produce query logits and conditional quality;
3. calculate the frozen calibrator features;
4. apply the frozen single-model/TTA/ensemble recipe;
5. resolve pixel ownership at 2048x2048;
6. assert every retained mask is non-empty, permit an image-level empty result,
   enforce no pairwise pixel overlap, and use deterministic ordering;
7. encode and immediately decode every RLE;
8. record count, area, checksum, runtime, and any empty-image result.

The final manifest includes all 180 source filenames, including those with zero
predictions. The controller independently downloads and verifies the CSV,
filename coverage, RLEs, reported per-image counts, weight/calibrator/config
hashes, and final CSV hash. Terminal state is `inference_ready`, not
`leaderboard_submitted`.

## 9. P100 runtime and failure policy

Pin PyTorch `2.7.1+cu126` and torchvision `0.22.1+cu126`, the pair already used
successfully in this repository. Pin exact MaskDINO/Detectron2 commits and build
their custom CUDA extensions with `TORCH_CUDA_ARCH_LIST=6.0`. The preflight must
run after the pinned PyTorch install because Detectron2 custom operators are
tied to the PyTorch/CUDA build.

To avoid recompiling mutable sources during every nine-hour job, the preflight
publishes a private, hash-addressed Kaggle runtime dataset containing the pinned
source snapshots, built wheels/extensions, dependency lock, licenses, and a
verification report. Training kernels consume that dataset and refuse an
unexpected hash. Internet-enabled source compilation remains a recovery path,
not the normal training path.

Mandatory preflight:

1. assert a Tesla P100, compute capability `(6, 0)`, and compiled `sm_60`;
2. import every Detectron2/MaskDINO custom operator;
3. run custom deformable-attention forward/backward in FP32;
4. run the full E1 model for 20-50 FP32 updates with finite tensor checks;
5. run 200 AMP updates with scaler, gradient, parameter, and logits checks;
6. save/reload a checkpoint and compare deterministic inference;
7. benchmark 1024, 1152, 1216, and 1280 where safe;
8. record allocated/reserved peak VRAM and stabilized step timing.

AMP is enabled only if the smoke is finite and materially useful on Pascal.
Every optimizer step unscales before clipping, records skipped scaler steps,
and checks gradients/parameters after the update. A non-finite event writes the
offending filename, annotation record, augmentation seed, scaler state, config,
and emergency checkpoint before failing the stage.

The resolution/memory fallback sequence is fixed:

1. 1280, 64 queries, decoder activation checkpointing;
2. 1280, 48 queries if the repeated max-count audit remains <=26;
3. 1216, then 1152 with 48-64 queries as measured;
4. checkpoint selected pixel-decoder stages;
5. if none stays below 14.5 GB with useful throughput, mark MaskDINO/P100
   infeasible and stop rather than disguising the experiment at 512.

Each training job reserves at least 60 minutes for checkpoint/report flushing or
runs inference as a separate dependent kernel. The controller never assumes a
6,000-microbatch run fits a session until the measured preflight extrapolation
leaves this margin.

Initial DataLoader settings are batch 1, workers 4, pinned memory, persistent
workers, and prefetch factor 2. Benchmark workers `{2,4,6}`. Precompute masks,
derived boxes, boundaries, skeletons, native areas, and membership metadata.
Profile data wait, transfer, backbone, pixel decoder, transformer, matching,
mask loss, backward, optimizer, validation, ownership, and RLE encoding.

## 10. Repository design

Create one experiment package and keep the reusable competition contracts in
`src/sfsc`:

```text
filament-pq-setformer/
|-- APPROACH.md                 # frozen selected experiment after planning
|-- IMPLEMENTATION_PLAN.md      # this document
|-- README.md                   # operator commands and terminal states
|-- config.py                   # typed config, validation, canonical hashing
|-- core.py                     # CPU split/calibration/assembly/report logic
|-- experiment_logging.py       # dual JSONL and offline W&B experiment logger
|-- model.py                    # Kaggle-only PQSetFormer implementation
|-- remote.py                   # stage executor
|-- run.py                      # local prepare/run/status controller
|-- audit.py                    # downloaded-artifact integrity audit
`-- vendor-lock.json            # upstream commits, patches, licenses, hashes

tests/
|-- test_pq_setformer_core.py
|-- test_pq_setformer_controller.py
|-- test_pq_setformer_contract.py
|-- test_pq_setformer_logging.py
`-- test_pq_setformer_audit.py

kaggle/pq-setformer-preflight/
|-- run.py
`-- kernel-metadata.json
```

Generated source snapshots and stage folders live under
`outputs/pq-setformer-<run-id>/`; active state lives in
`reports/pq_setformer_run.json`. Weights and dense predictions remain ignored.

### 10.1 Deep module interfaces

The CPU module exposes a small interface:

```python
split = make_experiment_split(filenames, split_spec)
calibrator = fit_calibrator(excluded_candidates, truths, calibration_spec)
prediction = assemble_instances(candidates, calibrator, assembly_spec)
report = evaluate_run(predictions, truths, comparison, report_spec)
```

It hides annotation-record weighting, official matching, feature schemas,
duplicate handling, pixel ownership, bootstrap grouping, and diagnostics. This
is the primary local test seam.

The model module exposes:

```python
losses = model.train_batch(images, targets)
candidates = model.predict(images, native_shapes)
```

Backbone conversion, pixel decoding, matching, deep supervision, quality
pooling, MP piloting, and AMP-safe internals stay behind that interface.

The remote runner exposes one stage interface:

```python
result = execute_stage(stage_config, kaggle_paths)
```

The logging module exposes one dual-sink interface:

```python
with ExperimentLogger.start(run_context) as logger:
    logger.log(event, step=optimizer_step)
    logger.log_table(name, rows)
    logger.log_artifact(kind, paths, metadata)
```

It writes the canonical event locally and mirrors it to the stage's offline W&B
run. Callers do not contain W&B retry, media, artifact, or directory logic.

The local controller exposes `prepare`, `run`, `status`, and an explicit
`run --override-gate`. It packages immutable sources, records push intent before
upload, adopts the returned Kaggle slug, reuses known kernels after interruption,
downloads only declared artifacts, and advances after verification.

## 11. Kaggle stage graph

```text
runtime-preflight
      |
      +--> pilot-1-control-train --> pilot-1-control-infer --+
      |                                                     |
      +--> pilot-1-query-train   --> pilot-1-query-infer ----+--> pilot-1-assess
                                                               |
                     selected recipe frozen <------------------+
                                                               |
      +--> pilot-2-control-train --> pilot-2-control-infer --+ |
      +--> pilot-2-query-train   --> pilot-2-query-infer ----+-+--> pilot-2-assess
                                                                    |
                                             promotion gate <--------+
                                                                    |
      +--> oof-fold-0-train --> oof-fold-0-infer --+                |
      +--> oof-fold-1-train --> oof-fold-1-infer --+--> final-calibrate
      +--> oof-fold-2-train --> oof-fold-2-infer --+          |
                                                               +--> full-train
                                                                        |
                                                               single-model-infer
                                                                        |
                                                    optional TTA/seed-2 stages
                                                                        |
                                                               verify-submission
```

Proxy and ablation stages use the same train/infer/assess templates with
different frozen configs. No remote stage modifies a previous stage's artifact.

## 12. Test plan

### CPU behavior tests

- maximum query requirement is derived per coherent record, not per filename
  union;
- split membership is deterministic, filename-grouped, block-disjoint, purged,
  hash-stable, and different across the two predeclared pilots;
- one complete annotation record is sampled deterministically from
  `(seed, epoch, filename)` and workers cannot duplicate RNG streams;
- derived boxes/boundaries/skeletons remain aligned through resize and flips;
- strict IoU equality at 0.5 is not a match;
- official pair-counting and one-to-one matching parity covers empty, duplicate,
  overlap, multiple-annotator, split, and merge cases;
- calibrator labels duplicates that lose assignment as negatives;
- feature ordering and standardization are schema-checked and finite;
- utility thresholding, mask deduplication, disconnected masks, pixel ownership,
  minimum area, and deterministic ties behave as specified;
- TTA inversion and geometry clustering preserve native coordinates;
- RLE encode/decode, all-180 filename coverage, zeros, and non-overlap are exact;
- bootstrap resamples date blocks while keeping all annotation records for a
  JPEG together.

### P100 integration tests

- custom deformable attention forward/backward on `sm_60`;
- one-channel stem loads the declared initialization and changes after an
  optimizer step;
- Hungarian matching handles zero, one, 26, 48, and 64 targets/queries safely;
- every enabled loss is finite and backpropagates to its intended parameters;
- detached quality pooling does not backpropagate through mask weights;
- piloted checkpoints infer with piloting disabled;
- accumulation matches the declared optimizer-step count, including final
  partial accumulation handling;
- FP32/AMP checkpoint save/reload produces stable candidate counts;
- 1280 inference upsampling and ownership remain below the memory limit;
- a short real-data smoke produces non-empty, non-collapsed masks on a fixed
  positive case set and empty output is allowed on negative-looking cases.

### Controller and artifact tests

- source changes after `prepare` are rejected;
- run and membership hashes propagate through every dependent stage;
- ambiguous pushes are resumed without duplicate training;
- a status/fetch timeout does not relabel a successful remote kernel as a
  training failure;
- `COMPLETE` with non-finite training, unchanged weights, missing coverage,
  corrupt candidates, or all-empty predictions is rejected;
- a failed gate prevents automatic downstream training;
- an explicit override is stage-limited and permanently recorded;
- JSONL and W&B histories contain the same terminal status and optimizer-step
  sequence;
- a missing, corrupt, or unhashed offline W&B run bundle fails artifact audit;
- final checkpoint, calibrator, assembly recipe, count manifest, and CSV hashes
  are independently checked after download.

## 13. W&B logging contract

W&B is required for all preflight, training, inference, calibration, assessment,
and submission-verification stages. Pin the W&B SDK version in the private
runtime bundle and record it in every report.

### 13.1 Run identity and operating mode

Remote stages set:

```text
WANDB_MODE=offline
WANDB_DIR=/kaggle/working/wandb
WANDB_PROJECT=sfsc2026-pq-setformer
```

Each immutable stage has a deterministic W&B identity:

```text
run_id:    sha256(experiment_run_id + stage_name + execution_source_sha256)[:16]
name:      <experiment-run-id>-<stage-name>
group:     <experiment-run-id>
job_type:  preflight | control | train | infer | calibrate | assess | full-fit | submit-verify
tags:      model family, pilot/fold, resolution, seed, AMP mode, gate status
```

`wandb.init` receives the complete frozen configuration before GPU allocation.
The local logger writes `wandb_run_id`, offline directory, SDK version, mode,
project, group, job type, and tags into the stage manifest. Failure to initialize
the offline run stops the stage. Failure to synchronize it later does not alter
the model result; it records `wandb_sync_failed` for operator recovery.

The controller downloads and hashes the complete offline run directory. Sync is
an explicit post-download operation, for example:

```bash
wandb sync outputs/pq-setformer-<run-id>/<stage>/results/wandb/offline-run-*
```

No Kaggle credential or network connection is required during training. Media
or source-image synchronization occurs only after confirming project visibility,
competition rules, and dataset licensing. Scalar and artifact metadata remain
available locally even if media sync is withheld.

### 13.2 Configuration logging

Every W&B run config includes:

- git SHA; source, execution-source, dataset, annotation, metric, membership,
  runtime-bundle, upstream-checkpoint, and candidate-cache hashes;
- stage, pilot/fold, role memberships and counts, seed, query count, resolution,
  model/backbone initialization and provenance;
- matching costs, every loss coefficient, point-sampling settings, MP pilot
  settings, optimizer groups, learning rates, schedule, accumulation, clipping,
  AMP/scaler settings, and augmentation probabilities;
- calibrator feature schema/regularization and complete assembly/TTA/ensemble
  configuration;
- Python, PyTorch, torchvision, CUDA, Detectron2, MaskDINO, pycocotools, and W&B
  versions plus GPU name, compute capability, and compiled architectures.

The serialized W&B config hash must equal the config hash in the local manifest.

### 13.3 Training history

Use logical `optimizer_step` as the W&B step axis. Log `microbatch` and
`epoch_equivalent` as values; never let gradient accumulation create four
different meanings for one step. At every optimizer update log:

```text
train/loss_total
train/loss_cls
train/loss_mask_bce
train/loss_dice
train/loss_box_l1
train/loss_giou
train/loss_quality
train/loss_foreground
train/loss_boundary
train/loss_cldice
train/matched_queries
train/no_object_queries
optim/lr_backbone
optim/lr_decoder
optim/grad_norm
amp/enabled
amp/scale
amp/skipped_step
perf/data_ms
perf/h2d_ms
perf/forward_ms
perf/matching_loss_ms
perf/backward_ms
perf/optimizer_ms
perf/step_ms
perf/images_per_second
memory/allocated_gb
memory/reserved_gb
```

The logger also retains W&B system metrics so CPU saturation, GPU utilization,
host RAM, disk use, and loader stalls can be diagnosed. High-frequency timings
may be accumulated locally and emitted as stable-window summaries every fixed
number of steps, but loss, LR, gradient, AMP, and memory evidence is emitted at
every optimizer update.

### 13.4 Evaluation and calibration history

Every validation, Q-cal grid, H assessment, OOF fold, and final inference pass
logs:

```text
eval/global_pq
eval/sq
eval/rq
eval/tp
eval/fp
eval/fn
eval/sum_matched_iou
eval/prediction_count
eval/truth_count
eval/count_mae
eval/fragmentation
eval/merges
eval/duplicate_suppressions
eval/ownership_damaged
eval/boundary_iou
eval/pq_small
eval/pq_medium
eval/pq_large
eval/pq_short
eval/pq_long
eval/runtime_seconds_per_image
eval/peak_vram_gb
calibration/brier
calibration/log_loss
calibration/expected_calibration_error
```

Calibration trials log mask threshold, utility threshold/quantile, minimum area,
ownership gamma, TTA/ensemble support rule, PQ, counts, and selection status. A
W&B summary stores the selected recipe and its exact Q-cal/H metrics; H values
are logged only after the one-shot assessment is complete.

### 13.5 Tables and charts

Create versioned W&B Tables for:

- a fixed 40-80 image error-analysis set with native image, GT/prediction
  overlays, site/date, annotation record, TP/FP/FN state, utility, presence,
  predicted/realized IoU, areas, and split/merge flags;
- per-image additive PQ sufficient statistics for the matched control and
  challenger;
- candidate reliability rows and TP/FP utility distributions;
- complete Q-cal threshold-grid results;
- ablation comparisons including delta PQ/SQ/RQ/TP/FP/FN, P100 time, and peak
  memory;
- OOF coverage/exclusion lineage and final 180-image prediction counts.

The resulting project workspace must support charts for PQ/SQ/RQ versus training
steps, TP/FP/FN versus utility threshold, PQ versus mask/utility thresholds,
predicted versus realized IoU, P(match) reliability, utility distributions, PQ
by area/length/site/date, step-time decomposition, memory, GPU utilization, and
AMP scale/skips.

Tables use fixed case membership so runs compare the same images. Large full-
dataset rows live as artifact Tables instead of being repeatedly appended to
run history.

### 13.6 W&B Artifacts

Log versioned artifacts for:

- runtime lock and custom-operator verification;
- source/config/metric snapshots and split/training memberships;
- final and last finite checkpoints for promoted training stages;
- Q-fit, Q-cal, H, and OOF candidate caches;
- calibrator coefficients, feature schema, and selected assembly recipe;
- stage reports, profiling summaries, and qualitative Tables;
- final inference count manifest, verified RLE CSV, and verification report.

Artifact names include content-hash prefixes rather than mutable `latest` being
the only reference. Each artifact's W&B metadata repeats its local SHA-256, and
the controller checks the local file before logging and after download. W&B is
never the sole copy of a checkpoint, candidate cache, split, or submission.

### 13.7 Sweeps and comparisons

The proxy-search specification is stored in a committed/frozen W&B sweep YAML
with objective `eval/global_pq` and the bounded ranges in Section 7. The local
controller remains responsible for one-at-a-time P100 scheduling and the fixed
trial count. When online sweep control is unavailable, it executes the same
configs deterministically and logs `sweep_name`, `trial_index`, and search-space
hash so the runs can be compared after sync.

No sweep reads H, alters its search space after results, promotes by validation
loss, or launches simultaneous Kaggle P100 sessions.

## 14. Artifacts and schemas

Each stage writes `progress.json` atomically plus append-only `events.jsonl`.
Training progress includes microbatch, optimizer step, epoch-equivalent, every
loss term, both learning rates, gradient norm, AMP scale/skips, timings, and
memory. The dual-sink logger mirrors the same logical events to the required
offline W&B run. Partial candidate caches are checkpointed by filename and
resumed only after config/checkpoint hash validation.

Required final reports:

- `runtime-report.json` and private runtime bundle hash;
- split manifests and membership audit;
- per-experiment training and candidate reports;
- calibration fit report, reliability bins, and selected assembly grid;
- pilot-1, pilot-2, 14-day sensitivity, and historical-validation comparisons;
- OOF coverage and exclusion report;
- full-training report and checkpoint hash;
- 180-image inference/count manifest;
- submission verification report and CSV hash;
- a qualitative fixed-case table for the competition report.
- hashed offline W&B run bundles plus per-stage synchronization status.

Every report contains status, run/stage IDs, timestamps, git SHA, source hash,
dataset/annotation/split hashes, environment and upstream hashes, config, seed,
membership counts, GPU identity, elapsed time, peak memory, and artifact hashes.

## 15. Work breakdown and completion criteria

### Phase 0: decisions and freeze

- Resolve the current matched-scale run's fetch-only failure and preserve its
  actual model result; it does not block coding but informs the final baseline
  narrative.
- Record the external-pretraining rules decision.
- Pin upstream commits and licenses.
- Freeze pilot 1, pilot 2, 14-day sensitivity, and planned 601 OOF split specs.

Done when all decisions and memberships are hashed before query-model outcomes
exist.

### Phase 1: local implementation

- Implement typed configuration, split logic, candidate/calibrator schema,
  assembly, evaluation/reporting, dual JSONL/W&B logging, artifact audit, and
  controller.
- Add the frozen `APPROACH.md`, operator `README.md`, and all CPU/controller
  tests.

Done when `uv run pytest`, `uv run ruff check .`, and
`uv run ruff format --check .` pass without installing local CUDA packages.

### Phase 2: P100 model implementation and preflight

- Package pinned runtime sources.
- Pin the W&B SDK and verify offline run, scalar, Table, and Artifact creation.
- Implement the one-channel model, matching, heads, losses, and prediction
  export.
- Run the complete FP32/AMP/custom-op/memory/reload preflight.
- Publish the private hash-addressed runtime input.

Done when a real optimizer updates finite weights, a reload reproduces
inference, the selected size stays below 14.5 GB, timing fits a Kaggle job, and
the downloaded offline W&B run passes the local parity audit.

### Phase 3: pilot and ablations

- Train the matched-R control and E1-E6 experiments in declared order.
- Fit/select on Q-fit/Q-cal only.
- Open pilot-1 H once, select the recipe, then run pilot 2 and 14-day
  sensitivity.

Done when the promotion gate produces a durable pass/fail decision with complete
metrics and artifacts.

### Phase 4: final OOF and 601 fit

- Train/infer all three excluded folds.
- Verify coverage and fit the final OOF calibrator/assembly recipe.
- Train the fresh 601-image deployment model.

Done when the final model, calibrator, config, and every lineage hash have been
downloaded and independently verified.

### Phase 5: inference, optional enhancements, and handoff

- Run single-model inference first.
- Admit TTA and seed 2 only through their gates.
- Generate and verify the 180-image CSV without uploading it.
- Update `SUMMARY.md`, the model evidence report, and the final competition
  method documentation with measured rather than forecast results.

Done at `inference_ready`, with a reproducible submission artifact and a written
operator decision on whether to upload it.

## 16. Compute budget

Planning range, replaced by preflight measurements:

| Work | P100-hours |
|---|---:|
| runtime build and stability/memory preflight | 1-2 |
| fair matched-R controls | 6-12 |
| E1 base model and full-resolution reruns | 8-12 |
| proxy search and targeted E2-E6 ablations | 15-25 |
| second blocked replication and sensitivity | 8-14 |
| three-fold 601 OOF calibration campaign | 15-24 |
| fresh 601 deployment fit and inference | 6-9 |
| optional TTA and second seed | 6-10 |
| **expected total before optional work** | **53-86** |

This is higher than the attachment's 40-70 hour range because it includes a
fair matched-membership control, a second blocked replication, and clean OOF
calibration for the full 601-image model. Compute stops early when E1 is clearly
noncompetitive or a later intervention fails its own marginal gate.

## 17. Primary risks and mitigations

| Risk | Detection | Response |
|---|---|---|
| MaskDINO/Detectron2 cannot build for P100 | preflight custom-op tests | pin/build private runtime; stop if `sm_60` remains invalid |
| 1280 exceeds 16 GB | measured peak/reserved VRAM | checkpoint activations, 48 queries, then 1216/1152 |
| AMP corrupts training | per-update finite/scaler checks and reload smoke | use FP32 and revise runtime estimate, not the architecture |
| query model predicts too few faint filaments | permissive recall and query-use diagnostics | test MP piloting/quality separately; stop if E1 RQ is not competitive |
| quality model repeats prior leakage | membership and candidate-hash audit | Q-fit/Q-cal/H separation and 601 OOF predictions |
| added recall recreates FP wave | final-set incremental PQ and utility histograms | conservative calibration; no union or forced fallback |
| topology loss bridges instances | merge diagnostics and clDice sweep | default low weight; disable unless marginal PQ improves |
| full-fit score distribution shifts | OOF versus deployment score/count audit | retain frozen calibration and report risk; no test-set retuning |
| historical validation overstates transfer | two prospective blocked pilots | require replicated direction and pooled positive interval |
| remote completion hides bad outputs | artifact audit and non-collapse checks | terminal failure despite Kaggle `COMPLETE` |
| W&B/network interruption loses visibility | mandatory offline run plus JSONL parity | download/hash locally; sync explicitly after completion |
| W&B media exposes restricted data | offline media bundle and explicit sync review | sync scalars/metadata first; withhold image media until approved |

## 18. Immediate implementation order

The first coding increment should be a vertical tracer through the complete
workflow:

1. scaffold config/core/experiment_logging/run/remote/audit and frozen stage manifests;
2. implement E1 only, with one coherent annotation sampler and 64-query output;
3. run a 20-step P100 job that trains, saves, reloads, predicts, assembles, and
   scores a tiny declared train-only sample;
4. download and verify every artifact locally;
5. verify JSONL/W&B event parity and sync the offline smoke run;
6. only then add the quality head/calibrator, piloting, structural losses, TTA,
   and OOF stage graph one at a time.

That tracer proves the deepest interfaces—model, remote stage, controller,
assembly, metric, and artifact verification—before the project spends a full
P100 session on architecture tuning.

## References used for implementation choices

- [Repository summary](../SUMMARY.md)
- [Evaluation contract](../EVAL.md)
- [GPU contract](../GPU.md)
- [Native blocked-pilot design](../filament-native-reconstruction/APPROACH.md)
- [OOF controller and calibration pattern](../filament-oof-consensus/APPROACH.md)
- [Official MaskDINO repository](https://github.com/IDEA-Research/MaskDINO)
- [Official Detectron2 installation guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)
- [Official PyTorch 2.7.1 installation matrix](https://pytorch.org/get-started/previous-versions/)
- [Current SFSC2026 overview and evaluation policy](https://www.kaggle.com/competitions/filament-segmentation-2026/overview)
- [W&B offline logging and sync](https://docs.wandb.ai/models/ref/cli/wandb-offline)
- [W&B Tables](https://docs.wandb.ai/guides/track/log/log-tables)
- [W&B Artifacts](https://docs.wandb.ai/models/artifacts)
- [W&B Sweeps](https://docs.wandb.ai/models/sweeps/walkthrough)
