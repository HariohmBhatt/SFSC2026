# Native reconstruction pilot

This directory contains an exploratory geometry diagnostic and the implementation
base for [APPROACH.md](APPROACH.md). The diagnostic trains only the native
512-pixel local refiner and compares it with the existing global-refiner masks on
the exact same proposals, assembly rules, and historical 106-image validation
set. It cannot promote the approach because that set has already been used during
earlier development.

Training prompts come from the three Consensus v3 OOF caches. Each training
image was held out from the upstream detector/refiner that generated its prompt.
The pilot uses the medoid annotation for positive targets, requires at least 50%
prompt purity, rejects multi-object prompts, and labels a hard negative only when
it has no overlap in any annotation record. The local audit found 4,523 positive
prompts across 490 images and retained 1,507 hard negatives.

The P100 job runs 4,000 optimizer updates with an effective batch of eight, then
reconstructs all candidates for the historical 106-image assessment. Its
diagnostic success criteria are:

- full-validation PQ improves by at least 0.015 over 0.435179;
- the non-calibration 82-image PQ improves by at least 0.010;
- the lower bound of the paired 60-day block-bootstrap interval is above zero;
- evaluator parity is exact.

The cached global-refiner output is a thresholded coarse mask, not its probability
map. This diagnostic therefore tests whether native windows can improve masks
under a cheap, reproducible conditioning substitute. The full pilot must generate
global probability maps inside the blocked R/Q/H design in `APPROACH.md`.

The diagnostic does not create or submit a competition CSV. A positive result
supports implementing the matched realistic-prompt experiment and blocked pilot;
only that evidence can advance sliced detector views and a full-data fit.

Run and inspect the controller with:

```bash
python filament-native-reconstruction/run.py prepare
python filament-native-reconstruction/run.py run
python filament-native-reconstruction/run.py status
```

The controller freezes source and membership hashes, launches a private Kaggle
P100 kernel, monitors it, downloads the report/model/cache, and verifies each
artifact hash. Its durable state is `reports/native_training.json`; a file lock
prevents duplicate launches.
