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
- The three-fleet iterative system is implemented: academic approach finding,
  resource-aware review, selected code writing, Kaggle evaluation/full training,
  and checked submission generation.
- The first [Filament ResUNet](filament-resunet/APPROACH.md) completed with
  validation PQ **0.287749** (766 TP, 1590 FP, 471 FN). Its internal test was
  already consumed and is excluded from further model selection.
- The September 6 [problem analysis](reports/competition_problem_analysis.md)
  and [public-method research](reports/competition_strategy_research.md) identify
  false detections and fragmentation as immediate priorities, with a clean
  detector/refiner and native recovery branch as the higher-upside research path.
- [Filament Precision v2](filament-precision/APPROACH.md) continues the train-only
  checkpoint on the Kaggle P100 and compares fine-tuning with recalibration alone.
  Its [run record](reports/precision_training.json) tracks launch and training
  evidence. This validation-only experiment preserves the frozen system history.
  See the [system guide](system/README.md) for the original execution workflow.

The September 6 research also updates older assumptions in the modeling docs:
organizer replies permit additional supplied metadata, and exact filename
grouping does not establish temporal independence. The original data/evaluation
contract files remain unchanged to preserve the previous experiment fingerprints.

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
uv run sfsc-system init
uv run sfsc-system status
uv run sfsc-audit --data-root SFSCDataset
```

Model training and inference run only in the private Kaggle P100 kernel. The
verified zero-install runtime uses Kaggle's preinstalled TensorFlow; see
[`kaggle/README.md`](kaggle/README.md). Do not install local CUDA packages.

## Competition Watch

`sfsc-watch` tracks the public competition activity through the authenticated
Kaggle CLI: public leaderboard, all competition notebooks (with pulled sources),
discussion topics, and related models/datasets. Each run merges into a local
state store, pulls sources only for new or changed notebooks, and re-renders a
self-contained dashboard with method tags and validity warnings.

```bash
uv run sfsc-watch                 # fetch, diff, and render
uv run sfsc-watch --skip-fetch    # re-render from stored state
uv run sfsc-watch --serve 8791    # render and serve at 127.0.0.1:8791
```

The dashboard (reports/competition_watch/index.html, Git-ignored) highlights
new notebooks, vote deltas, extracted method tags (architecture, loss, TTA,
post-processing), key hyperparameters, and code excerpts. Suspect notebooks are
flagged from source patterns: filename-keyed annotation lookups that never match
the COCO-style JSON, empty-submission fallbacks, and leaderboard-defect
mentions. Scrape state and cached sources stay local and are never committed.

## Repository Layout

```text
.
|-- src/sfsc/              # Data audit, splitting, and evaluation code
|-- system/                # Fleet policy, prompts and operator/runtime contracts
|-- tests/                 # Metric, split and end-to-end orchestration invariants
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
