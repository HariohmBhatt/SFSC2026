# PQ-SetFormer operator guide

The code prepares and executes the mandatory E1 Kaggle P100 smoke and the
subsequent frozen E1 pilot. The private data, CUDA/MaskDINO build, AMP, memory,
checkpoint, inference, and artifact gates passed at 1024 pixels in run
`20260910-161954` and at the selected 1216-pixel resolution in run
`20260910-163643`.

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

The pilot fits only the frozen 289-image R membership and evaluates its final
checkpoint on the frozen 87-image H membership. It does not mount a competition
source or submit predictions.

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

Terminal success is `smoke_complete` for a preflight or `pilot_complete` for the
long E1 pilot, for example:

```json
{"status": "smoke_complete"}
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
