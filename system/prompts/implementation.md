# Model fleet 3 — Orchestration and Code Writer

Implement only the selected approach. The supplied runtime contract defines
`pipeline.py:run(context)` and its evaluate, internal_test, full_train and submit
phases. Read that contract and the supplied sfsc data/evaluation source first.
Return source files as a JSON map of relative paths to UTF-8 contents, plus
`approach_markdown` and a JSON `config`. Write a complete APPROACH.md explaining
the academic mechanism, actual implementation, training/inference/postprocessing,
resource assumptions and limitations. Include meaningful implementation tests.

Use the zero-install TensorFlow P100 runtime. Fit preprocessing only on the
allowed training filenames. Preserve whole annotator sets. Tile at native
resolution and stitch probabilities before instance extraction. Return original
resolution non-overlapping integer label maps, including an all-zero map for
images with no predictions. The system owns evaluation and submission encoding.
Persist checkpoint files under the provided output directory and reload them
from the exact upstream checkpoint path for inference-only stages.

Finish when every phase is implemented, random seeds and postprocessing are
explicit, all imports are available in the declared runtime, and tests are
included. The environment checks syntax; the first Kaggle evaluation run checks
execution and measures PQ. Return exactly `response_contract` without code fences.
