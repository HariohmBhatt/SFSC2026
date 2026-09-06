# Selected approach runtime contract

Implement `pipeline.py` with `def run(context):` returning a dictionary. Every
phase runs on the Kaggle P100; importing or testing helpers locally must not
start TensorFlow training. The launcher supplies the source package and fixed
manifest; all model source files live inside the selected short-code directory.
Sibling modules can be imported normally. No network or package installation is
available in the declared Kaggle runtime.

`context` contains:

- `phase`: `evaluate`, `internal_test`, `full_train`, or `submit`.
- `dataset`: `sfsc.dataset.Dataset`. `filenames(*splits)`, `image_path(filename)`,
  `records[filename]` (whole alternative annotation sets), and `truth(record)`
  provide access to native-resolution images and instance masks.
- `train_filenames`: train only for evaluate; train+validation for full_train;
  empty for internal_test and submit.
- `predict_filenames`: validation, internal test, or competition test filenames
  for the corresponding phase; empty for full_train.
- `output_dir`: writable Path under `/kaggle/working/experiment`.
- `checkpoint`: exact upstream checkpoint Path; None for evaluate. Full training
  reads the evaluation-selected recipe here and initializes fresh model weights.
- `config`: the accepted implementation's JSON configuration.
- `seed`: fixed experiment seed.

Evaluate fits the candidate, selects a checkpoint/postprocessing using validation
only, and returns `{"predictions": "predictions", "checkpoint": "model.keras"}`.
Prediction paths are relative to output_dir. Save one `<image_stem>.npy` per
requested source image: a 2-D original-resolution nonnegative integer label map
(0 background, positive IDs are instances). Include zero maps for empty images.
The system independently computes global PQ over every alternative annotation
set. Save the actual selected postprocessing in the checkpoint bundle (a
checkpoint may be a directory). Keep the input config reproducible.

Internal_test loads the exact evaluation checkpoint, performs inference with its
frozen postprocessing and returns `{"predictions": "predictions"}`. The workflow
reserves the holdout once globally, after freeze. It is a report, never a tuning
signal. A failed or interrupted holdout requires investigation; it cannot be
silently rerun under another experiment ID.

Full_train retrains with the frozen configuration on train+validation (the
internal test remains held out) and returns `{"checkpoint": "model.keras"}`.
The number of epochs/steps must be fixed from validation, not retuned here.

Submit loads the full-training checkpoint and predicts competition images using
`dataset.image_path(filename, competition=True)`. Return
`{"predictions": "predictions"}`. The system checks coverage, encodes COCO RLE,
round-trips each mask and creates submission.csv. Uploading to the leaderboard
is a separate explicit action.

Checkpoint directories can include weights and frozen postprocessing JSON. The
system mounts the previous completed kernel's outputs through `kernel_sources`
and resolves the exact recorded checkpoint path. Write all outputs inside the
provided output directory. Use `float32` initially; any precision change belongs
in the accepted config. Seed every extra random generator you instantiate.

The runner writes report.json with contract/code hashes, commit, seed, runtime
versions, GPU identity, TensorFlow allocator peak memory, elapsed time, additive
PQ and group diagnostics. Allocator memory is not total device memory; this is
reported separately from reviewer estimates. Implementation source is trusted
executable code: the manifest API guides split access, but Python is not a
security sandbox. Review accepted implementation code and its tests before a
paid/limited GPU launch.
