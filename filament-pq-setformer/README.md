# PQ-SetFormer operator guide

The code implements the bounded PQ-SetFormer research seams. It does not make
an existing checkpoint promotable merely because a Kaggle kernel completed.
The 2026-09-11 E1 result is preserved as a failed diagnostic; see
[HISTORICAL_E1_DISPOSITION.md](HISTORICAL_E1_DISPOSITION.md).

Training runs under Kaggle2 (`hariohmhemantbhatt`) and mounts only the private
`hariohmhemantbhatt/sfsc2026-private-data` dataset. The kernel has no competition
source and cannot submit. Competition inference and submission remain a separate
primary-account (`hariohmbhatt`) stage.

## Local verification

From the repository root:

```bash
uv sync --extra dev
uv run pytest tests/test_pq_setformer_*.py
uv run ruff check filament-pq-setformer tests/test_pq_setformer_*.py
uv run ruff format --check filament-pq-setformer tests/test_pq_setformer_*.py
```

The local environment intentionally does not install PyTorch, Detectron2, or
MaskDINO. Model imports and execution belong to the pinned P100 kernel.

## Prepare and run the smoke

Authenticate the Kaggle CLI as the training account, then run:

```bash
uv run python filament-pq-setformer/run.py prepare
uv run python filament-pq-setformer/run.py status
uv run python filament-pq-setformer/run.py run
```

For a smoke or pilot, start the local Telegram relay in a second terminal
before `run`. It sends every controller state transition immediately, then the
complete structured training event stream plus the verified global and
per-image PQ report after Kaggle artifacts are available:

```bash
uv run python filament-pq-setformer/telegram_monitor.py --interval 45
```

It reads `SFSC_TELEGRAM_BOT_TOKEN` and `SFSC_TELEGRAM_CHAT_ID` from `.env` or
the environment. Kaggle does not expose live stdout/stderr during execution;
the relay reports live status and forwards the complete logger event stream on
the terminal artifact fetch.

After a successful 1024 smoke, prepare the measured-safe resolution preflight
explicitly, for example:

```bash
uv run python filament-pq-setformer/run.py prepare --input-size 1216
uv run python filament-pq-setformer/run.py run
```

Once that preflight is verified, launch the 6,000-microbatch E1 research pilot
at the selected resolution:

```bash
uv run python filament-pq-setformer/run.py prepare --run-kind pilot --input-size 1216
uv run python filament-pq-setformer/run.py run
```

The pilot fits only the frozen 289-image R membership and writes candidates for
the frozen Q-fit/Q-cal memberships. It cannot infer H through this stage. E3
fits the versioned logistic calibrator on Q-fit, selects a declared recipe on
Q-cal, and records both membership and candidate hashes. H is a separate,
one-shot assessment after that recipe has been frozen.

After downloaded pilot artifacts verify as `execution_complete`, run the local,
H-inaccessible calibration stage:

```bash
uv run python filament-pq-setformer/run.py calibrate
```

It writes a hashed `q-calibration.json` alongside the immutable stage directory.
Before opening H, freeze an assessment contract that identifies the challenger
and independently trained matched-R control, their exact R membership, and both
checkpoint/source/config hashes. The separate one-shot H runner then supplies
only hashed artifact locations. The controller reconstructs challenger
predictions, decodes the matched-control final masks, recomputes both sides with
the same evaluator, verifies report parity, and reruns the blocked bootstrap
locally:

```bash
uv run python filament-pq-setformer/run.py freeze-assessment \
  --evidence /path/to/assessment-contract.json
uv run python filament-pq-setformer/run.py assess \
  --evidence /path/to/h-artifact-bundle.json
uv run python filament-pq-setformer/run.py gate
```

`gate` never accepts external metrics. It consumes only the sealed assessment
stored by the verifier; altered, incomplete, wrong-membership, or hash-mismatched
artifacts cannot reach `promoted`.

`prepare` freezes the exact source, configuration, and image membership under
`outputs/pq-setformer-<timestamp>/` and writes the active record to
`reports/pq_setformer_run.json`. `run` packages that snapshot, installs the
pinned CUDA-compatible runtime, launches one private internet-enabled P100
kernel, monitors it, downloads its outputs, and audits them. A second active run
is refused.

Before pushing, the controller verifies that the active Kaggle CLI identity is
exactly `hariohmhemantbhatt`; primary-account or ambiguous credentials are
rejected. It also content-hashes all private training images and annotations and
requires the remote mount to reproduce the same snapshot.

The controller records push intent before calling Kaggle. If upload completion
is ambiguous, it remains in `pushing`; inspect the Kaggle account instead of
preparing another job. If Kaggle completes but artifact download fails, it
remains `remote_complete_fetch_pending`, and rerunning `run` retries only the
fetch/audit.

Successful artifact verification is `execution_complete`; it is not promotion.
The only post-assessment outcomes are `gate_failed` and `promoted`, and a
promotion requires matched-R control evidence, candidate diagnostics, and a
positive paired date-block bootstrap interval. For example:

```json
{"status": "execution_complete"}
```

Any `failed` state requires inspection of the saved error and remote logs before
a new run is prepared.

## W&B evidence

Each stage always writes the same logical history to `events.jsonl` and a pinned
W&B `0.30.0` offline run. Optimizer steps include all losses, both learning
rates, gradient norm, AMP scale/skips, throughput, and GPU memory. Evaluation is
logged as scalars plus a per-image Table; the checkpoint and evaluation bundle
are W&B Artifacts with hashes.

Offline runs are not synced automatically. After the controller has reached
`smoke_complete`, locate the downloaded `offline-run-*` directory, review it for
competition-data policy compliance, and explicitly sync it if desired:

```bash
find outputs/pq-setformer-*/results -type d -name 'offline-run-*'
wandb sync /absolute/path/to/offline-run-...
```

## What this does not do yet

The current runner does not launch an ablation sweep, matched-R control, OOF
campaign, 601-image deployment fit, competition inference, or Kaggle
submission. Those stages are gated on measured pilot evidence and are specified in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). This boundary prevents an
unverified custom CUDA build or unsafe memory estimate from consuming a full
nine-hour P100 session.

## Initialization and candidate migration

Random R50 runs use explicit trainable GroupNorm. A generic ImageNet R50 is
allowed only when `pretrained_backbone`, a local weight path, and that file's
SHA-256 are frozen in the configuration; the RGB stem is averaged to one input
channel and the load fails if any non-classifier backbone tensor is missing. No
weight is downloaded at runtime.

New candidates use schema v2: native RLE plus bounded, quantized low-resolution
probabilities, presence logit, conditional-IoU estimate, source provenance,
geometry, and payload hash. The cache is rejected if the probability payload
would exceed its declared bound. Version-1 candidates remain auditable only and
cannot be calibrated or assembled under E3, because their ownership logits were
discarded. The old checkpoint cannot be loaded into E2 because it has no
conditional-IoU-head weights and the pinned MaskDINO output must be patched to
provide final query and pixel-decoder features. The wrapper computes the
matched-query `loss_quality` and candidate conditional-IoU logits from that
explicit contract.
