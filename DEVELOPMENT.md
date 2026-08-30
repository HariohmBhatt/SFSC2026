# Development Environment

## Canonical Setup

Use Python 3.11 for local data, metric, and repository tooling.

```bash
uv sync --extra dev
uv run pytest
```

The local environment is CPU-only and sufficient for auditing data, generating
splits, encoding masks, and testing metrics. All model execution is submitted to
the Kaggle P100 through the CLI; no local training framework is installed.

## Dataset Audit And Split

The committed split is regenerated deterministically from the annotation JSON:

```bash
uv run sfsc-audit \
  --data-root SFSCDataset \
  --manifest splits/split_manifest.csv \
  --summary splits/split_summary.json \
  --report-json reports/data_assessment.json \
  --report-md reports/data_assessment.md \
  --seed 2026
```

The command fails on missing files, bad dimensions, orphan annotations, invalid
polygons, duplicate file content, or train/competition-test overlap. It does not
copy or move the source images.

## Test And Quality Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Kaggle P100 Runtime

Run and verify the zero-install remote environment with:

```bash
kaggle kernels push \
  --path kaggle/p100-smoke \
  --accelerator NvidiaTeslaP100 \
  --timeout 300

kaggle kernels status hariohmbhatt/sfsc-p100-smoke-test
```

The tested default runtime is Python 3.12.13 and TensorFlow 2.20.0. Capture the
runtime at the start of every training job:

```python
import platform
import subprocess
import tensorflow as tf

print(platform.platform())
print("tensorflow:", tf.__version__)
print("gpus:", tf.config.list_physical_devices("GPU"))
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

Kaggle's default PyTorch 2.10.0 CUDA 12.8 build is not usable on this P100:
it omits the required `sm_60` kernels. Use TensorFlow for the zero-install P100
pipeline rather than silently falling back to CPU.

Do not rely on mutable notebook state. A final notebook must execute from top to
bottom, consume only declared inputs, write outputs under `/kaggle/working`, and
round-trip-check every submitted COCO RLE.

## Git And Experiment Hygiene

- Keep the repository private during active competition work unless team policy
  says otherwise; follow Kaggle's code-sharing rules.
- Commit split manifests and small reports, not source data or model weights.
- Use one configuration per run and store metrics in machine-readable form.
- Tag serious baselines so leaderboard submissions map back to exact code.
- Do not tune against the internal test set or the public leaderboard.
