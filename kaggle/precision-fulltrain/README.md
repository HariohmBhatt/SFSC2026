# Frozen full training and leaderboard submission

Authorized by the user on September 6, 2026 after precision-v2 scored 0.32 public.
The development choice is fixed: precision epoch15, hysteresis foreground .55,
confidence .75, minimum area256, native768 tiles/stride512 and horizontal TTA.

The 601-image baseline stage is already completed by
`hariohmbhatt/sfsc-cycle-001-full-train-e0b5f3ed`: 60 epochs, original schedule,
fresh initialization on train+validation. Its complete checkpoint directory is
identified by SHA256 `8b574576663abb60a6ad2039f3d19b22de803898a2172fb09180a1db574992c9`.
Reusing that exact stage avoids repeating 4489 seconds of work.

The new runner executes 15 precision epochs ×200 steps on all601 development
images with a fresh optimizer and the original 8000-step cosine schedule. It
does not compress the40-epoch schedule into15 epochs, tune new thresholds, or
access internal-test masks. Its frozen recipe has no validation PQ because
the training set now includes the original validation images.

After training, the same kernel predicts all180 competition images and uses
the shared encoder to round-trip every COCO RLE. Checkpoints, predictions,
recipe, logs, `submission.csv` and reports are retained under
`/kaggle/working/experiment`.

```bash
uv run python kaggle/precision-fulltrain/prepare.py
# Push the generated directory from reports/full_training.json, explicitly on P100.
uv run python kaggle/precision-fulltrain/finish.py
# After remote completion and explicit user authorization:
uv run python kaggle/precision-fulltrain/finish.py --submit
```

The local finish command downloads only checkpoint/report/CSV artifacts, checks
run/source/weight/submission hashes and the frozen recipe, independently decodes
all RLEs, checks counts and overlap, and respects the five-submission daily cap.
It records an upload attempt before calling Kaggle. An ambiguous upload is never
retried automatically; matching submission history is checked first.

The accepted baseline orchestration state and earlier model source snapshots are
unchanged. `reports/full_training.json` is the durable record for this experiment.
