# Filament Precision v2

This is the first experiment from the September 6 public-method research,
not a claim of leaderboard superiority. The baseline's completed validation PQ
is 0.287749 (766 TP, 1590 FP, 471 FN); 1271 FP are below 1024 pixels.
Its diagnostics count 322 fragmented truths and nine merged predictions across
169 validation annotation records. The immediate hypothesis is that precision
and fragmentation are more urgent than a larger backbone.

## Experiment

Warm-start the 4.66M-parameter native-resolution TensorFlow ResUNet from the
original **train-only evaluation** kernel, checked against its checkpoint hash.
Do not use the later full-training checkpoint, which saw validation images.
Keep the 495 training filenames, 106 validation filenames, and original source
grouping. Sample one complete annotation set at a time.

Fine-tune for up to 40 epochs of 200 steps, batch 2, crops 768, float32 on one
P100. AdamW learning rate 1e-4 decays to 5e-6; positive BCE weights fall from
4/4/8 to 1/1/2 and foreground-directed sampling falls from 70% to 50%.
These are a combined precision intervention, not a claim to isolate each cause.
Use the existing grayscale/local-contrast channels and geometric/photometric
augmentation. A three-hour training cap leaves time for validation within a
nine-hour job. Expected training is approximately one hour from the previous
measured throughput, with additional calibration/inference overhead; verify live.

Compare two extraction modes: hysteresis connected components (one confident
patch retains the complete weakly connected filament) and watershed with a
larger minimum interior seed of 32 pixels. Calibrate grow/confidence pairs
(.30,.55), (.45,.65), (.55,.75), (.65,.85) and minimum areas 64/128/192/256.
Also include the exact previous recipe (.65 foreground, .5 interior, area64,
legacy four-pixel seeds). The 65 configurations are selected on the fixed
24-image validation subset. Small-object recall must be reported because larger
area cutoffs can discard real annotations.

Calibrate the unchanged baseline first. Score the initial weights and every fifth
epoch on the same 24 images using that fixed recipe. Retain the best weights,
then independently calibrate them and the final epoch, selecting the better of
the two. Initial weights remain eligible, so failed fine-tuning need not replace
the baseline. Final `best.json` and `recipe.json` identify the selected checkpoint.

Report full 106-image validation PQ and diagnostics for both the selected model
and unchanged baseline with its recalibrated extraction. Save selected-model
float16 probabilities for later CPU geometry experiments; official metrics use
the original float32 probabilities before this storage conversion. All reported
validation scores are development estimates, not independent holdout scores.
Do not access the internal test: the previous experiment already consumed it.

## Execution and artifacts

`uv run python filament-precision/prepare.py` creates a self-contained private
Kaggle kernel in `outputs/precision-v2-*` and records its identity in
`reports/precision_training.json`. Submit that directory with the Kaggle CLI,
explicit P100 and timeout 32400. The package uses the existing verified runtime
for GPU checks, source/annotation/checkpoint hashes and full validation scoring.
It leaves the frozen cycle history intact and performs no leaderboard upload.
This experimental runner accepts only the evaluate phase.

The kernel writes training steps, calibration grids, checkpoints, selected recipe,
baseline metrics, full validation report, predictions and probability maps under
`/kaggle/working/experiment`. A recorded optimizer step with finite loss is the
evidence that training actually started; a successful push alone is insufficient.

`uv run python filament-precision/status.py` refreshes the saved run status and
latest optimizer evidence. After remote completion it fetches only small JSON
reports and checks their run/source identity. Large checkpoints and probability
maps can be downloaded separately when needed. This local status helper was added
after launch and does not change the immutable training snapshot.

## Next evidence gates

Compare recalibration alone against fine-tuning. Then isolate loss/sampling,
train-only hard-negative mining, and topology loss if fragmentation persists.
The higher-upside research track is a generic-pretrained proposal detector with
conditioned crop refinement and native-resolution recovery, evaluated with clean
training manifests. Public filament checkpoints with unknown training membership
cannot establish held-out performance on our validation split. See
`reports/competition_strategy_research.md` for public-source evidence and ranking.
