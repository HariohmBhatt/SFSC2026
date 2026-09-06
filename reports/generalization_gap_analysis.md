# Precision-v2 generalization audit — September 6, 2026

**The available measurements do not establish test-domain shift as the cause of the roughly 0.061 validation–public gap.** The 180 competition images have a similar site mix, temporal proximity to training, and simple intensity/contrast distributions. Fine-tuning's +0.010 validation gain is modest and statistically borderline. The next model should address both unsupported small detections and missed large filaments; “tiny filaments are the weakness” is incomplete.

This audit uses saved predictions' aggregate statistics, development membership, and image-only test covariates. It never opens internal-test annotations or images, and never obtains competition-test labels. Public PQ **0.32** is the user's observed submission score, not a freshly verified full-precision score. The public scoring subset and its labels are unknown here, so comparison with all 180 test images has limits.

## Reproducibility and uncertainty

Run `.venv/bin/python tools/analysis/generalization_gap.py > reports/generalization_gap_stats.json` from the repository root. [The script](../tools/analysis/generalization_gap.py) reads the completed [precision report](../outputs/precision-v2-20260906-052615/results/experiment/report.json), [recalibrated control](../outputs/precision-v2-20260906-052615/results/experiment/baseline-metrics.json), [calibration membership](../outputs/precision-v2-20260906-052615/results/experiment/checkpoint/calibration.json), and [split manifest](../splits/split_manifest.csv). All new numerical results are in [machine-readable statistics](generalization_gap_stats.json).

The 169 annotation records belong to 106 JPEGs. Sum matched IoU, TP, FP, FN within each JPEG; resample 106 JPEGs with replacement, paired between models, and recompute global `PQ = sum(IoU)/(TP + (FP+FN)/2)`. There are 20,000 replicates, seed 20260906. Resampling annotation rows independently would exaggerate sample size because multiple annotators share the same observation.

| Frozen candidate | PQ | JPEG bootstrap 95% interval | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Recalibrated baseline weights | 0.37112 | 0.34123–0.39988 | 816 | 928 | 421 |
| Precision-v2 | 0.38117 | 0.35309–0.40836 | 800 | 801 | 437 |
| Paired difference | +0.01005 | −0.000007–+0.02015 | −16 | −127 | +16 |

Fine-tuning buys fewer false detections while losing some matches; mean matched IoU changes from 0.67789 to 0.67611. The bootstrap fraction with nonpositive improvement is 2.515%; this is not a posterior probability or an independently designed hypothesis test. These intervals condition on the selected models and thresholds, omit model-training/selection uncertainty, and do not account for dependence across observing dates. They are descriptive, potentially optimistic intervals rather than proof of a reproducible +0.010 gain.

There is no simple “calibration images are artificially easy” signal: the selected 24 JPEGs score 0.36879 versus 0.36783 for control; the other 82 score 0.38368 versus 0.37180. The other 82 are not a new untouched holdout: broader development decisions and previous checkpoint selection may already use validation. Calibration overfitting remains possible, but the measured fine-tuning gain is not confined to the 24 threshold-selection images. Source: [statistics](generalization_gap_stats.json), [calibration implementation](../filament-precision/pipeline.py).

## Temporal correlation: risk, but not the measured explanation

Distance is the absolute timestamp difference from a validation/test filename to its nearest one of 495 training timestamps, across all sites. It is a proxy, not identification of the same physical filament.

| Nearest training observation | Validation JPEGs | Validation PQ | 95% JPEG interval |
|---|---:|---:|---:|
| ≤1 day | 26 | 0.34067 | 0.291–0.405 |
| >1–3 days | 43 | 0.37365 | 0.336–0.408 |
| >3–7 days | 30 | 0.42006 | 0.370–0.463 |
| >7 days | 7 | 0.39112 | 0.274–0.537 |

There is no observed tendency for temporally closer validation images to score higher: per-JPEG Spearman rho 0.053, nominal p=0.588. The >7-day group is far too small to establish generalization to independent solar events. Median distance is 2.07 days for validation and 2.10 days for competition test; 99/106 validation and 161/180 test JPEGs are within seven days. A test observation lies only 20 seconds from a train timestamp; no visual-duplicate conclusion follows from that alone. These observations weaken a simple claim that validation is temporally near training while test is remote. They do not eliminate event leakage or site/time interactions. Source: [timestamp audit](generalization_gap_stats.json).

## Observable distribution differences

| Site | Validation images | Test images | Validation PQ |
|---|---:|---:|---:|
| Bh | 17 | 33 | 0.410 |
| Ch | 18 | 33 | 0.378 |
| Lh | 21 | 34 | 0.348 |
| Mh | 22 | 26 | 0.322 |
| Th | 14 | 27 | 0.389 |
| Uh | 14 | 27 | 0.473 |

Reweighting each validation JPEG's additive statistics by its test/validation site-count ratio yields PQ **0.38644**. Reweighting by year yields **0.37727**. These are counterfactual standardizations under unchanged within-stratum behavior, not test-score estimates. The simple observed composition changes do not explain −0.061. Years 2021–2022 are more common in test (21/180 versus 6/106), but late-year validation counts are too small for reliable conditional estimates. The particularly weak 2021 estimate, 0.159, comes from only four JPEGs. Source: [site/year counts and intervals](generalization_gap_stats.json).

Validation has 64/21/21 JPEGs with one/two/three annotation sets; their PQ is 0.38487/0.40313/0.36251. Competition-test annotation multiplicity is unavailable from filenames, so its shift cannot be measured here. **The 1,636 predicted instances in a submission are predictions, not a measurement of test label prevalence.** Ground-truth frequency, annotator policy, and difficult-object prevalence remain unobserved explanations.

Image-only summaries use grayscale images resized to 256×256, a central circle of radius 60% of half-width, and far corners. Central median, central 90th–10th percentile contrast, and vertical gradient have test-minus-validation standardized mean differences −0.134, +0.140, and −0.124. Their validation-PQ Spearman correlations are +0.075, −0.078, and −0.008. Corner median is identically zero. These coarse summaries show no large global photometric shift or strong association with PQ; they cannot detect subtle local texture, disk-position, seeing, morphology, or annotation shifts. No threshold or model decision was tuned against unlabeled-test covariates. Source: [image summary definitions/results](../tools/analysis/generalization_gap.py).

## Actual errors and competing explanations

| Truth/prediction area band | PQ | TP | FP | FN | Truth miss rate |
|---|---:|---:|---:|---:|---:|
| Tiny: <1,024 px | 0.33089 | 342 | 528 | 166 | 32.7% |
| Medium: 1,024–4,095 px | 0.44088 | 395 | 249 | 186 | 32.0% |
| Large: ≥4,096 px | 0.36482 | 63 | 24 | 85 | **57.4%** |

Matching is global; TP/FN bands use truth area, FP bands use prediction area. Thus tiny false detections may include fragments of larger objects. There are 135 fragmented-truth and 13 merged-prediction diagnostics across annotation records; these overlap counts are not mutually exclusive additional error totals. Unsupported small masks (528/801 FP) and loss of whole large structures are both major priorities. Source: [completed metrics](../outputs/precision-v2-20260906-052615/results/experiment/report.json), [metric contract](../EVAL.md).

Plausible causes of the gap remain sampling variation in the unknown public subset; selection optimism from a reused validation set; unobserved annotation/morphology differences; and differences between our conservative one-to-one evaluator and the live official evaluator. Source matching of submitted geometry/inference was checked in the parent investigation, reducing one engineering concern without proving metric parity. The local contract uses strict IoU >0.5; the user's ≥0.5 summary and public evaluator history make exact matching semantics worth checking on synthetic masks. A local confidence interval for validation does **not** measure the uncertainty of a difference against an unknown public sample. Neither “the drop is random” nor “the drop is domain shift” is established.

## Action order

1. Complete the already authorized frozen full-train candidate and make one submission. Retain current 0.32 and its exact checkpoint/postprocessing as control; report full-train's public delta without claiming validation improvements after training on validation.
2. Implement the clean detector/refiner experiment in [the companion specification](detector_refiner_experiment_spec.md). Compare full-frame PQ and paired per-JPEG counts on the original development split; explicitly measure large/long-object recovery and tiny false detections.
3. Add a second, time-blocked validation design inside the 601 development filenames. Hold complete contiguous date blocks across sites; purge training observations within seven days of held blocks, then run a fourteen-day-gap sensitivity design. Use fixed intervals rather than transitive “within seven days” chains that can group an entire observing season. Publish membership and remaining training counts before training. Every evaluated detector, refiner, dense model, and ensemble member must exclude the outer holdout.
4. Limit threshold alternatives to a preregistered compact neighborhood, report selection versus assessment separately, and retain per-image metrics/probability caches. Audit evaluator parity with empty predictions, nonoverlap, exact-IoU-0.5, duplicate masks, and multi-annotator weighting before attributing any remaining gap to model biology or camera domains. Never consume internal-test annotations to choose these changes.
