# Solar Filament Segmentation Challenge 2026

Reproducible development ground for the [Kaggle Solar Filament Segmentation
Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026/overview).
The task is instance segmentation of solar filaments in 2048 x 2048 grayscale
H-alpha images. The primary leaderboard metric is Panoptic Quality (PQ).

## Current State

- The local dataset has been audited and is intentionally excluded from Git.
- A fixed, leakage-safe 70/15/15 development split is stored in
  `splits/split_manifest.csv`.
- The competition PQ calculation is implemented in `src/sfsc/evaluation.py`.
- No model baseline has been selected yet. The repository first fixes data and
  evaluation semantics so later experiments are comparable.

Read these before modeling:

- [`DATA.md`](DATA.md): measured dataset properties, risks, and split policy.
- [`EVAL.md`](EVAL.md): metric definition and fixed evaluation protocol.
- [`GPU.md`](GPU.md): exact compute recommendation and memory strategy.
- [`DEVELOPMENT.md`](DEVELOPMENT.md): environment and workflow.

## Quick Start

Python 3.11 is the canonical version. [`uv`](https://docs.astral.sh/uv/) is the
recommended environment manager.

```bash
uv sync --extra dev
uv run pytest
uv run sfsc-audit --data-root SFSCDataset
```

Install model-training packages only in an environment that needs them:

```bash
uv sync --extra dev --extra train
```

On Kaggle, use its preinstalled CUDA-enabled PyTorch rather than replacing it
blindly. Record the assigned GPU and package versions at the start of every run.

## Repository Layout

```text
.
|-- src/sfsc/              # Data audit, splitting, and evaluation code
|-- tests/                 # Metric and split invariants
|-- splits/                # Versioned fixed split manifests
|-- reports/               # Versioned generated dataset assessment
|-- .github/workflows/     # Continuous integration
|-- SFSCDataset/           # Local-only Kaggle data (Git-ignored)
|-- DATA.md
|-- EVAL.md
|-- GPU.md
`-- DEVELOPMENT.md
```

## Data Placement

The default expected layout is:

```text
SFSCDataset/
|-- train/
|   |-- train_images/*.jpeg
|   `-- MAGFiLO_1.0_Annotations_kaggle2026_train.json
`-- test/
    `-- test_images/*.jpeg
```

Override it with `--data-root` or `SFSC_DATA_ROOT`. Never commit competition
images, annotations, credentials, model weights, or generated predictions.

## Experiment Contract

Every reported experiment must record the Git commit, split manifest SHA-256,
seed, configuration, package versions, GPU name, peak VRAM, validation PQ, and
post-processing parameters. The internal test split is not for model selection.
