# Native reconstruction pilot

This directory implements the first controlled experiment in
[APPROACH.md](APPROACH.md). It trains only the native 512-pixel local refiner and
compares it with the existing global-refiner masks on the exact same proposals,
assembly rules, and frozen 106-image validation set.

Training prompts come from the three Consensus v3 OOF caches. Each training
image was held out from the upstream detector/refiner that generated its prompt.
The pilot uses the medoid annotation for positive targets, requires at least 50%
prompt purity, rejects multi-object prompts, and labels a hard negative only when
it has no overlap in any annotation record. The local audit found 4,523 positive
prompts across 490 images and retained 1,507 hard negatives.

The P100 job runs 4,000 optimizer updates with an effective batch of eight, then
reconstructs all candidates for the frozen 106-image assessment. Promotion
requires all of the following:

- full-validation PQ improves by at least 0.015 over 0.435179;
- the non-calibration 82-image PQ improves by at least 0.010;
- the lower bound of the paired 60-day block-bootstrap interval is above zero;
- evaluator parity is exact.

The pilot does not create or submit a competition CSV. A passing result supports
the next experiment: sliced native-scale detector views, followed by isolated
quality fitting and a full-data fit.

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
