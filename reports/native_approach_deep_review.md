# Native reconstruction v4: deep review and repair report

Review date: **September 8, 2026 (UTC)**. Reviewed implementation: commits
`58651c6`, `249400a`, and `1dec90b`; failed run `20260908-104235` and its
downloaded artifacts; the Consensus v3 artifacts; the current competition rules
and leaderboard; and two independent reviews of the revised implementation.

## Verdict

The global/local reconstruction idea is technically reasonable, but the first
implementation was neither a valid trained model nor the complete experiment
described by `APPROACH.md`. The immediate failure was numerical corruption: all
trainable tensors in the saved checkpoint became non-finite. The kernel still
reported `COMPLETE`, inference silently converted non-finite probabilities into
1,723 empty masks, and the resulting PQ was 0.000.

Native refinement alone is also unlikely to close the whole gap from the best
honest artifact's 0.38 public score to the current 0.56 leaderboard maximum. On
the frozen validation set the control has SQ 0.6844 and RQ 0.6359, with 723 TP,
314 FP and 514 FN. It therefore needs better proposal recognition and selection
as well as finer masks. The repaired strategy keeps native reconstruction as an
isolated mask-quality experiment, then requires sliced proposal recovery and a
new quality model before any full fit or submission.

The official task requires complete instance masks and uses PQ, which penalizes
FPs, FNs, fragmentation and merging. The organizers also consider efficiency and
have warned against metric manipulation. The review therefore excludes the
public-label overlap and empty-mask loophole discussed on Kaggle from the model
strategy. Sources: [official competition announcement](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements),
[official metric description](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/citation),
[organizer response to the empty-mask disclosure](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728474).

## Measured evidence

| Evidence | Result | Implication |
|---|---:|---|
| Best submission | 0.38 public | Retain as the deployment control |
| Full-601 version of that family | 0.30 public | More labeled images did not guarantee transfer |
| Consensus v3 | 0.43272 validation, 0.37 public | Quality reranking added TP but added more FP and failed its gate |
| Native run control | 0.435179 validation | Exact evaluator/control parity was reproduced |
| Native run output | 0.000 PQ; 0/1,723 non-empty candidates | Artifact is invalid, not negative model evidence |
| Native training | 4,000 updates in 2,315 s; 4.61 GiB peak | The architecture fits comfortably on P100 |
| Failure onset | logged loss first non-finite by update 2,600 | Training needed fail-fast numerical checks |
| Checkpoint probe | every trainable tensor non-finite; frozen global encoder finite | Corruption occurred in the optimized path before thresholding |
| Candidate supervision | 4,523 positives + 1,507 hard negatives over 491 images | Honest OOF prompt volume is adequate for a diagnostic |
| Current leaderboard pull | top 0.56 on September 8 | A first-place result remains unverified and cannot be promised |

The deterministic failure command is:

```bash
python filament-native-reconstruction/audit.py \
  outputs/native-pilot-20260908-104235/results/native-pilot/pilot-report.json \
  outputs/native-pilot-20260908-104235/results/native-pilot/native-validation-candidates.json
```

It exits nonzero with `all reconstructed candidate masks are empty` and `run was
marked complete despite integrity failures`.

## Root-cause analysis

Three explanations were separated with a P100 checkpoint probe:

1. **Numerical corruption** predicted non-finite model tensors. Confirmed: every
   trainable encoder, context and decoder tensor contained non-finite values.
2. **Threshold miscalibration** predicted finite weights and probabilities below
   0.55. Rejected for this artifact because the weights themselves were corrupt.
3. **Stitching/assembly loss** predicted finite local outputs that became empty
   later. Rejected for this artifact for the same reason.

The old instrumentation cannot identify the first corrupt operation. It recorded
only a cumulative loss, did not abort on non-finite loss or gradients, did not
check parameters, and retained only the final checkpoint. AMP/fp16 is the leading
suspect, but the evidence does not prove a specific CUDA operation caused the
first non-finite value. PyTorch explicitly notes that AMP/fp16 is not compatible
with every model and recommends checking NaNs/gradients; gradient inspection must
occur after unscaling. Sources: [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp),
[PyTorch AMP examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html).

## Issue register

### Critical issues fixed in implementation

| ID | Problem | Repair |
|---|---|---|
| C1 | Kaggle `COMPLETE` was treated as model success | Added artifact audit plus remote/controller integrity gates |
| C2 | Non-finite loss and parameters were allowed through training and inference | Added fail-fast finite loss, finite gradient norm, checkpoint parameter and probability checks |
| C3 | Unverified fp16 was used on P100 | Stable baseline uses FP32; AMP is now only a later controlled profile, because the stabilization bundle does not isolate AMP as the cause |
| C4 | Learning rate was aggressive for a new two-path decoder | Reduced AdamW LR from 3e-4 to 1e-4 and gradient clip from 5 to 2 |
| C5 | Only the corrupt final checkpoint survived | Writes a last-finite weight snapshot every 500 updates for diagnosis; full resumable state remains a blocked-pilot requirement |
| C6 | Cumulative loss hid the failing batch | Reports rolling/last/min/max loss, gradient norm and AMP scale |
| C7 | A randomly initialized absolute-mask head could discard a good prompt immediately | Decoder now predicts a zero-initialized residual over the softened coarse mask; seed remains conditioning instead of being forcibly restored |
| C8 | The inherited 0.55 threshold was applied to a new output head | Selects from a predeclared grid on the 24 calibration images and reports the 82-image result separately |
| C9 | All-empty inference still produced a normal report | All-empty or non-finite inference now raises and cannot pass artifact verification |
| C10 | Batch-wide BCE underweighted 128-pixel padded ROIs by as much as 16x versus 512-pixel ROIs | BCE is normalized per proposal before the batch mean, matching the Dice term and tiny-object objective |
| C11 | Seed/coarse union prior reinstated detector pixels rejected by the global refiner at logit +2.94 | Prior now starts from the coarse refined mask only; seed and coarse/native alternatives remain separately measurable |
| C12 | Seeded proposal shuffle defeated a single-image decode cache | Cache each grayscale training image after first use; the 495-image ceiling is about 2 GiB and fits the Kaggle host-memory envelope |
| C13 | The corruption auditor could itself crash on null or malformed numbers | Invalid numeric fields and cache structure now become explicit audit findings |
| C14 | Calibration membership was trusted implicitly and a Boolean diagnostic decision used consumed validation | Added uniqueness/subset/disjointness assertions and membership hashes; historical results are descriptive only and produce no continue/stop decision |
| C15 | Native threshold-key construction was duplicated four times | Centralized key construction to prevent calibration/cache drift |

### High-impact scientific issues corrected or explicitly blocked

| ID | Problem | Resolution |
|---|---|---|
| H1 | The old run was called a controlled pilot although it reused historical validation | It is now labeled a historical geometry diagnostic and cannot promote the approach |
| H2 | The implementation skipped experiment B: realistic prompts in the matched 384 model | B is again a required isolated experiment before attributing gains to native geometry |
| H3 | There was no R/Q/H implementation | Prepared and tested a deterministic 60-day-block split helper: R 289, Q-fit 69, Q-calibration 34, H 87, purged 16; the actual R-only training orchestration is still blocked |
| H4 | Existing caches could indirectly leak H if their upstream model trained on H | Full pilot must regenerate detector/global outputs inside R; existing OOF caches remain diagnostic-only |
| H5 | Binary RLE masks were described as probability maps | Code/docs call them coarse masks; full pilot must export actual global probabilities/features |
| H6 | The “global encoder” was a frozen generic ImageNet encoder, not the trained global refiner | Accepted only as a diagnostic context substitute; full pilot must reuse the excluded global refiner encoder |
| H7 | Seed-only ROI could omit extent already recovered by the global mask | Future ROI covers the union of seed and coarse mask before applying context |
| H8 | Horizontal/vertical flips reversed coordinate-channel meaning | Flip augmentation now restores canonical x/y signs |
| H9 | Chronologically sorted samples were fed without shuffling | DataLoader now uses deterministic seeded shuffling |
| H10 | The new mask distribution reused the old refiner score | Allowed only to isolate geometry; deployment requires newly excluded quality targets and refitting |
| H11 | Native-only assembly discarded seed/global alternatives | Revised plan retains all alternatives and selects one with the refitted quality model |
| H12 | Reporting lacked domain diagnostics | Added site and time-period comparisons; size/width/length and proposal-stage reports remain mandatory for the blocked pilot |

### Strategic issues in the original order

1. **Refinement cannot recover an unproposed instance.** The control's 514 FN and
   RQ 0.6359 make proposal recovery a required path. After B/C, run whole-image
   plus four 1,280-pixel detector views and measure marginal TP/FP before any
   fine-tuning. SAHI supports sliced inference as a testable proposal mechanism,
   but its aerial-detection results do not establish an SFSC gain.
   [SAHI paper](https://arxiv.org/abs/2202.06934)
2. **Fine detail is useful only when instance identity survives.** The residual
   prompt prior and global/local windows now make that invariant explicit.
   CascadePSP supports coarse-to-fine refinement, but does not validate this
   compact architecture on H-alpha data.
   [CascadePSP paper](https://arxiv.org/abs/2005.02551)
3. **Topology loss is not a first repair.** clDice is relevant to tubular
   connectivity, but a bridge between neighboring filaments can improve a
   topology surrogate while damaging PQ. It remains a small, isolated ablation
   only after the stable BCE/Dice baseline.
   [clDice paper](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)
4. **Full-601 fitting is not automatically the final model.** The 0.38-to-0.30
   regression requires a learning-curve transfer check. Preserve the validated
   R-trained champion; treat a 601-image fit as a challenger with matched update
   exposure, not an assumed upgrade.
5. **The 0.56 public maximum is weak architecture evidence.** Public discussions
   include metric loopholes and overlap with a public MAGFiLO release. Those
   disclosures do not prove wrongdoing by any named team, but they make headline
   scores unsuitable labels for copying an architecture. The solution remains
   train-only, reproducible and eligible for qualitative review.

## Repaired experiment sequence

| Stage | Data and change | Decision |
|---|---|---|
| 0 | Historical numerical/integrity diagnostic | Must produce finite weights and non-collapsed masks; never promotes |
| A | Freeze 0.38 control and exact evaluator | Stop if artifact/source/control parity fails |
| B | R-only matched 384 refiner: synthetic prompts vs excluded realistic prompts | Continue only on complete-mask PQ improvement |
| C | Add native residual reconstruction with fixed proposals | Require +0.015 H PQ, positive assessment delta and acceptable runtime |
| D | Add four native detector views | Keep only if marginal TP gain exceeds FP/fragmentation cost |
| Q | Fit alternative quality on Q-fit; choose acceptance/mask threshold on Q-cal | Never train or choose on H |
| E | Open H once, then repeat selected recipe on a second blocked assignment | Require positive replication and pooled block-bootstrap lower bound above zero |
| F | Compare validated champion with matched-exposure 601 challenger | Preserve champion when transfer evidence is weak |
| G | Infer, verify and submit one frozen candidate | Submission cannot tune the pipeline retroactively |

All upstream components used to produce Q/H predictions must be trained on R.
The two inner R fits generate realistic prompts for native training without
letting detector/global-refiner supervision cross into H. Existing Consensus OOF
caches are useful for error analysis and the historical diagnostic, but cannot
substitute for that lineage.

## Implemented repairs and verification

Files changed:

- `filament-native-reconstruction/audit.py`: deterministic artifact integrity check.
- `filament-native-reconstruction/core.py`: prepared R/Q/H split assignment,
  threshold selection, complete conditioning bounds, tiling and blending primitives.
- `filament-native-reconstruction/remote.py`: residual refiner, FP32 stable path,
  per-proposal loss, finite guards, forensic weight snapshots, isolated threshold
  calibration and stratified report.
- `filament-native-reconstruction/run.py`: stable configuration and controller-side
  rejection of corrupt artifacts.
- `filament-native-reconstruction/APPROACH.md`: corrected evidence scope, model
  initialization, threshold and numerical-integrity requirements.

Local verification passes Ruff, Python compilation, the focused native tests and
the complete repository suite. The failed run is preserved as
`reports/native_training_20260908-104235.json` and its immutable output directory.

The obsolete numerical-only run `20260908-152408` was stopped at the user's
request. The reviewed P100 diagnostic then ran as `20260908-162230`, kernel
`hariohmbhatt/sfsc-native-pilot-20260908-162230`, from source hash
`dd073bd7322ac6fca197d48206cb43ea4f68bde3ef6c20bbe2a1ff389c09b4fb`.
All artifact hashes and integrity checks passed: 4,000 FP32 updates completed with
zero non-finite batches, and 1,713/1,723 candidate masks were non-empty. Historical
PQ changed from 0.435179 to 0.435553 (+0.000374); on the non-calibration 82 images
it changed from 0.434862 to 0.435309 (+0.000447), with paired block-bootstrap 95%
interval [-0.000246, 0.001787]. This is indistinguishable from the control and far
below the +0.015 pilot target, so it does not justify submission or full fitting.

## Remaining limits

The numerical diagnostic can show whether the stabilization bundle produces
finite output and describe retrospective behavior on historical validation. It
cannot establish prospective signal, generalization, a winning score, or justify
full fitting. Global probability
export, letterboxed global geometry, the matched experiment B, actual R-only
orchestration and upstream fits, fully resumable training state, quality refitting,
detailed size/length diagnostics, sliced proposal recovery and the second blocked
pilot remain promotion requirements. The revised approach makes these hard gates,
so a partial implementation can no longer be mistaken for the full solution.
