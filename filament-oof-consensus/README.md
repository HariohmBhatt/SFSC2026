# Running Consensus v3

See [APPROACH.md](APPROACH.md) for the frozen experiment and [REVIEW.md](REVIEW.md)
for the reasoning behind the revision.

- `core.py`: CPU candidate features, quality fitting, mask assembly, exact PQ and
  paired block bootstrap. Shared by every remote stage.
- `remote.py`: P100 detector/refiner training, complete-pipeline held predictions,
  quality fitting, validation, gated full fit and inference.
- `run.py`: immutable packaging, Kaggle lifecycle, stage lineage checks and final
  submission-file verification. Requires the authenticated Kaggle CLI.

The active run is recorded in `reports/consensus_training.json`. Its output
folder contains source snapshots, per-stage scripts/metadata, a controller log
and a controller PID. Do not launch a second run to check its status.

```bash
uv run python filament-oof-consensus/run.py status
```

To inspect live optimizer progress, use the active kernel ID from the report:

```bash
kaggle kernels logs -f OWNER/KERNEL
```

A Kaggle RUNNING status can still mean environment installation. Only finite-loss
`detector_training` or `refiner_training` events establish actual training.

If the controller stops, inspect its error and push/fetch logs, fix the local
orchestration issue and resume with:

```bash
uv run python filament-oof-consensus/run.py run
```

Existing kernel IDs are reused; ambiguous pushes are never blindly repeated.
Changes to core or remote training code do not affect an already frozen run.
Do not edit its frozen snapshots. A failed GPU kernel requires an explicitly
recorded replacement stage, not silent artifact reuse.

The successful terminal state is `inference_ready`. The failed-validation state
is `gate_failed`; all validation results are preserved, and final training does
not start. No leaderboard upload occurs automatically.
