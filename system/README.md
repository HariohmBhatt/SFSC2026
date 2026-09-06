# SFSC2026 iterative development system

This is the executable environment for **Notebook.tldr → SFSC2026**, following
the supplied PNG. The canvas is a system architecture, not a web-interface mockup.
The other Notebook page and instructions quoted in the prior session are outside
this implementation's scope. Existing DATA.md, EVAL.md and GPU.md define the
project's data, evaluation and execution contracts.

## Design → implementation

| Canvas block | Working implementation |
| --- | --- |
| Init: .md files | `init` validates and fingerprints DATA.md, EVAL.md, GPU.md and config |
| Fixed Evaluation Process | Existing strict-IoU, global PQ evaluator is fingerprinted |
| Dataset Breakup | Existing 495 train / 106 validation / 106 internal-test filenames are preserved |
| Model fleet 1 / Approach Finding | Three research jobs, at least two approaches each, with academic citations |
| Model fleet 2 / Approach Validation | Every approach reviewed on PQ potential, complexity, time, VRAM and size |
| Chosen approach handoff | Deterministic weighted ranking with hard time/VRAM/compatibility limits |
| Model fleet 3 / Code Writer | Selected implementation, config, tests and `<shortcode>/APPROACH.md` |
| Evaluation | Kaggle trains on train; the system scores validation predictions against each annotator |
| Full Training | Frozen recipe retrains on train+validation after a one-time internal-test report |
| Submission | Inference on competition images, checked label maps, compressed COCO RLE CSV |
| Iterative System | Saved feedback and validation results feed the next research cycle |

## First use

From the SFSC2026 repository:

```bash
uv sync --extra dev --locked
uv run sfsc-system init
uv run sfsc-system cycle
uv run sfsc-system status
```

Initialization is idempotent. It preserves the existing split and dataset. It
creates `.sfsc/state.json`, an event history and self-contained worker request
files. `.sfsc/` is local and ignored by Git: back it up to retain experiment
history, results and the one-time holdout record across machines. Do not reset it
to bypass the holdout policy. Concurrent handoffs use an exclusive file lock and
atomic state replacement on macOS/Linux.

Defaults in `config.json` are explicit **starting policy choices**, not facts
from the diagram: 3 research workers × 2 approaches, a 14 GB allocator envelope,
9 hours per GPU job, and ranking weights 50% PQ potential, 20% training time,
10% each complexity/VRAM/model size. Set these before `init`. Once initialized,
the evaluation contract and ranking policy stay fixed for comparable cycles.

## Connect model workers

Every worker reads one JSON request on stdin and emits one JSON result on stdout.
Logs go on stderr. Credentials belong in environment variables. Requests include
the role prompt, relevant project documents, prior-cycle evidence, input artifacts
and the response contract; attached transcripts and image files are never sent.

A complete optional OpenAI adapter is included. Supply your API key through your
normal environment/secret manager, and set `SFSC_MODEL` to an API model ID you
have access to. Research/review require a model supporting Responses web search.
Use `SFSC_RESEARCH_MODEL`, `SFSC_REVIEW_MODEL`, and `SFSC_IMPLEMENTATION_MODEL` to
choose different models per fleet. No specific model is silently selected.

```bash
# Set OPENAI_API_KEY and SFSC_MODEL in your environment first.
uv run sfsc-system fleet cycle-001 --command uv run sfsc-model-worker
```

This runs research workers concurrently (at most three), then review, then code
writing. It stops at an accepted implementation, or at an actionable failure.
It does not launch GPU jobs. Calls use Responses with academic-domain search for
research/review; model identity, usage and retrieved-source metadata are retained
in worker results. Domain filtering and schema validation do not prove a paper
supports a claim: the review fleet must read and assess the evidence. API calls
are not automatically retried after ambiguous network failures.

Other command-line agents can use the same protocol via a small wrapper. Pass
its executable after `--command`, or set argv lists for the three roles in
`config.json` before initialization. This is argv execution, not a shell command.
No worker service is required for manual handoffs:

```bash
uv run sfsc-system request JOB_ID > request.json
# Give request.json to your chosen model; save its JSON answer as result.json.
uv run sfsc-system accept JOB_ID result.json
```

Replace JOB_ID with an ID from `status`. Run one configured worker with `work
JOB_ID --command ...`. Failed handoffs retain the phase and can be retried;
completed handoffs cannot be silently replaced. After a host interruption,
confirm the old process has stopped, then use `recover-job JOB_ID --reason ...`.

## Evaluate a selected approach

The accepted source lives under its short code at the repository root. Review
`APPROACH.md`, `pipeline.py`, `config.json` and the implementation's tests. The
[runtime contract](RUNTIME.md) is included in the code writer's request. Syntax
and required files are checked at acceptance; generated tests and model behavior
are not presumed correct just because they parse.

```bash
uv run sfsc-system prepare-run cycle-001 evaluate --owner YOUR_KAGGLE_USERNAME
uv run sfsc-system launch RUN_ID
uv run sfsc-system remote-status RUN_ID
uv run sfsc-system fetch RUN_ID
```

Use the returned RUN_ID. Kaggle CLI authentication and competition access are
required for `launch`/`fetch`. Preparation is local: it creates `.sfsc/runs/RUN_ID/`
with a private kernel, fixed manifest and an exact embedded source snapshot. It
also fingerprints local annotations against the remote competition mount. Kaggle
runs contain no package-install commands and require a functioning TensorFlow
P100 GPU. Upstream checkpoints are attached through `kernel_sources`; each run
gets a unique kernel slug to avoid overwriting previous stage outputs.

`fetch` downloads the report and artifacts, verifies their run identity, hashes,
PQ sufficient statistics, split coverage and checkpoint integrity, then advances
the workflow. For already-downloaded artifacts, use `collect RUN_ID
/path/to/experiment/report.json`. Reports include global and per-site/year/
annotator-multiplicity PQ, matched IoU/Dice distributions, size diagnostics,
fragment/merge counts, wall time, runtime versions and allocator peak VRAM.
Model size is measured from checkpoint files. The size diagnostic partitions the
same global matching: matched/false-negative instances use truth area and false
positives use predicted area. It does not change the competition objective.

A malformed or failed report cannot advance the phase. A failed evaluation can
be prepared again with the same accepted code. A code/config change requires a
new cycle and short code; accepted source snapshots remain immutable. Use
`reject cycle-001 --reason ...` to end a failed approach and `cycle --feedback ...`
to search again. Evaluated cycles can feed new cycles directly. Each candidate's
review score is explicitly an estimate; measured validation PQ determines which
evaluated cycle you eventually freeze.

## Freeze → full training → submission

Once model selection is complete:

```bash
uv run sfsc-system freeze cycle-001
uv run sfsc-system prepare-run cycle-001 internal_test --owner YOUR_KAGGLE_USERNAME
# launch / remote-status / fetch this RUN_ID
uv run sfsc-system prepare-run cycle-001 full_train --owner YOUR_KAGGLE_USERNAME
# launch / remote-status / fetch this RUN_ID
uv run sfsc-system prepare-run cycle-001 submit --owner YOUR_KAGGLE_USERNAME
# launch / remote-status / fetch this RUN_ID
```

Each next phase becomes available only after the previous successful report.
Freeze closes model selection globally. Internal test is reserved once at
preparation, including if its job fails: a holdout failure requires investigation,
not silently testing another model. Full training uses train+validation; the
internal holdout is retained. The submission stage loads the exact full-training
checkpoint, covers every competition image, and generates `submission.csv`.
Leaderboard upload is deliberately a separate action.

A kernel push with an uncertain outcome stays `launching`; check `remote-status`
and collect its output instead of pushing twice. If a job is confirmed failed
without a report, record it with `fail-run RUN_ID --reason ...`. This records a
failure; it does not cancel a still-running Kaggle kernel. No background polling,
recurring task, paid model call or GPU launch occurs merely by initializing.

## Validation and practical limits

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests exercise multi-worker handoffs, failures, ranking, all execution phases,
source packaging, holdout protection, annotator aggregation and submission RLE.
Remote-boundary fixtures are synthetic and are never recorded as competition
results. The optional API adapter is tested with recorded-shape responses, not
live paid calls. New model implementations still need an actual Kaggle run.

This is a local orchestration environment, not a security sandbox. Accepted model
code is executable Python. Split access is supplied by the fixed manifest and
phase-specific filename lists; code review must ensure the implementation obeys
those lists. The runtime fails on changed annotations, wrong GPUs, malformed
predictions or measured budget excess. TensorFlow allocator peak memory is not
whole-device reserved memory, and the 9-hour platform timeout can terminate a job
before it can save a report.

## API references

- [Kaggle kernel CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Kaggle kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
- [OpenAI Responses](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
- [OpenAI web search](https://developers.openai.com/api/docs/guides/tools-web-search)
