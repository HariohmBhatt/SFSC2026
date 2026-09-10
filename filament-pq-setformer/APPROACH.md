# PQ-SetFormer E1 vertical tracer

Status: **private-data P100 smoke passed on 2026-09-10**.

This package is the first executable increment of the broader
[implementation and training plan](IMPLEMENTATION_PLAN.md). It is deliberately
limited to E1: a complete train-save-reload-predict-assemble-score path that
must pass on the actual Kaggle P100 before longer experiments are permitted.
It does not claim that the full E2-E8 campaign or final 601-image deployment
training has run.

## Frozen E1 decision

- Input: one 2048×2048 grayscale image, resized to 1024 for the first smoke.
- Target: one complete annotator record selected deterministically per image
  presentation. Records are never unioned.
- Model: one-channel ResNet-50, MaskDINO deformable pixel decoder, six encoder
  and six decoder layers, and 64 object queries.
- Initialization: random; external pretraining is disabled until its rules
  status is explicitly approved.
- Losses: weighted filament classification, point-sampled mask BCE/Dice, and
  mask-derived box L1/GIoU with deep supervision. Quality, foreground,
  boundary, clDice, MP piloting, TTA, and ensembling remain disabled so their
  later effects can be measured independently.
- Optimizer: AdamW, decoder/head LR `1e-4`, backbone LR `1e-5`, weight decay
  `0.05`, warmup plus cosine decay, physical batch 1, accumulation 4, and
  gradient clip `0.5`.
- Candidate policy: preserve raw query identities and disconnected masks, use
  permissive presence export, remove geometric duplicates, then assign each
  pixel to at most one retained query.

The query count is bound to a repeated coherent-record audit whose observed
maximum is 26, rather than to an invalid union of all annotations for a JPEG.

## Evidence separation

Pilot 1 uses the already frozen 495-image historical training membership and a
deterministic 60-day blocked split with a seven-day purge:

| Role | Images | Use |
|---|---:|---|
| R / fitting | 289 | model training |
| Q-fit | 69 | later quality-model fitting |
| Q-cal | 34 | later assembly selection |
| H / assessment | 87 | one-time promotion evidence |
| Purged | 16 | excluded from that pilot |

The smoke uses only R and scores two declared R images. Those numbers prove
pipeline execution, artifact integrity, and non-collapse; they are not
generalization estimates.

## Required smoke gate

The immutable Kaggle stage must prove all of the following before any proxy or
full run begins:

1. the runtime is a Tesla P100 with compute capability 6.0 and compiled
   `sm_60` support;
2. pinned Detectron2 and MaskDINO sources build and the model executes real CUDA
   forward/backward operations;
3. 20 microbatches produce exactly five finite optimizer updates and change the
   model fingerprint;
4. the training model is released, the checkpoint reloads strictly, and
   inference emits non-empty raw candidates;
5. peak memory, timing, losses, metrics, and terminal state are mirrored to
   append-only JSONL and mandatory offline W&B;
6. checkpoint, candidate cache, report, W&B identity, membership, config, and
   source hashes pass the independent local audit.

Kaggle `COMPLETE` alone is not acceptance. The controller ends at
`smoke_complete` only after the downloaded artifacts pass the audit. It never
submits a competition CSV.

## Promotion boundary

Run `20260910-161954` passed as `smoke_complete` on a Tesla P100 with PyTorch
2.7.1/cu126. Twenty 1024-pixel microbatches produced five optimizer updates in
28.34 seconds with no AMP overflow. Peak allocated/reserved memory was
9.33/9.45 GB. Checkpoint reload, 128 raw candidates over two images, offline
W&B, JSONL parity, and artifact hashes passed local audit. The training input
was the private `hariohmhemantbhatt/sfsc2026-private-data` snapshot (708 files,
607,987,717 bytes, digest `e9ef61ae423e`). No competition source was attached.

Naive area scaling projects 1280-pixel reserved memory to about 14.77 GB, above
the 14.5 GB gate. The next measured target is therefore 1216/64; only after that
preflight should the staged E1 pilot, isolated E2-E6 additions, second blocked
replication, three-fold OOF calibration, fresh 601-image fit, and primary-account
competition inference run in the order defined by the full plan.
