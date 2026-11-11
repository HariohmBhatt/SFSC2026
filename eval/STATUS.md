# Final V4 OOF morphology evaluation — complete

Measured on 707 native 2048×2048 training images, 1,154 annotator records and 8,199 GT polygons. 95% percentile CIs resample 656 UTC date clusters (2,000 draws, seed 20261024); V4−V3 is paired.

## Report-ready findings (all measured)

- Frozen detector evidence raises pooled OOF PQ from 0.4504 [95% CI 0.4406, 0.4606] (V3 rescored) to 0.4545 [95% CI 0.4447, 0.4642] (V4); paired gain 0.0041 [95% CI 0.0014, 0.0069]. Shared kept objects retain identical masks; only keep decisions differ.

- V4 has 5,258 TP, 2,285 FP and 2,941 FN over 1,154 annotator records. Precision is 69.7% [95% CI 68.1, 71.2%] and recall 64.1% [95% CI 62.6, 65.7%]. Matched IoU is 0.6803 [95% CI 0.6773, 0.6833] and Dice 0.8067 [95% CI 0.8045, 0.8088]; these outline scores condition on a match and exclude the stated misses.

- Counting missed objects as zero, mean GT-spine coverage is 60.4% [95% CI 59.0, 62.0%], 61.7% [95% CI 60.2, 63.3%] and 62.3% [95% CI 60.8, 63.8%] at 1/3/5 native px. The mean longest uncovered contiguous fraction at 3 px is 38.1% [95% CI 36.6, 39.6%].

- For matched objects with valid spines, predicted skeleton/GT spine length ratio is 1.2374 [95% CI 1.2247, 1.2504], width ratio 0.8406 [95% CI 0.8308, 0.8505], and elongation ratio 1.0753 [95% CI 1.0668, 1.0841]. The same width definition gives human-mask/reference-spine ratio 0.7994 [95% CI 0.7850, 0.8146]; mask skeletons count branches, unlike the single drawn spine. These are projected proxies, not physical width errors or verified barb measurements.

- GT match rate is 43.3% [95% CI 40.0, 46.6%] for area <500 px and 67.6% [95% CI 63.5, 71.5%] for area ≥5000 px. This stratification includes all GT objects, rather than only survivors.

- Mean spine coverage at 3 px is 64.0% [95% CI 61.7, 66.5%] in the nominal disk centre and 49.4% [95% CI 45.7, 53.1%] at the limb (centroid ≥0.8R). The radius is the frozen preprocessing radius, not a fitted solar limb.

- On multi-record images, V4 spine coverage at 3 px is 33.5% [95% CI 31.3, 35.9%] for GT with no other-record IoU>0.5 support and 87.8% [95% CI 85.9, 89.7%] when every other record supports it. Support is an evaluation-label property and was not supplied to the model.

- The directed human-to-human reference gives PQ 0.3399 [95% CI 0.3237, 0.3563] and 3 px spine coverage 51.3% [95% CI 48.9, 53.7%] on 296 multi-record images / 1,196 directed comparisons. It is a reference agreement level on that subset, not an upper bound. V4 PQ on the same multi-record image population is 0.4518 [95% CI 0.4388, 0.4649].

## Provenance and verification

- Adopted saved variant: **v4_evidence**, tau **0.200**; baseline **v3_rescored**, tau **0.225**. The augmented V5 variant was not adopted. Both use unchanged V3 candidate geometry.
- Every image's reconstructed Hungarian IoU sum/TP/FP/FN reproduces the saved fusion statistics to numerical precision; see `results/pq_verification.json`. The host many-to-many metric is reported separately in all final tables.
- Host matching adds one qualifying pair per model versus Hungarian (V4 official PQ 0.454473, V3 0.450369); the small convention difference is explicit.
- Local verified V4 replica PQ is 0.454413, versus historical shipped V4's rounded 0.455. These results do not silently claim the shipped value.
- V4 unique OOF predictions: 4,612; record-weighted predictions: 7,542. Missing GT spines: 0; human-reference missing spines: 0.
- Compared with V3, V4 keeps 4,230 identical shared instances, adds 382 and removes 245.
- V4 split relations: 210 / 8,199 GT objects; merge relations: 173 / 7,542 record-weighted predictions. Rates with date CIs: split 2.6% [95% CI 2.1, 3.0%], merge 2.3% [95% CI 1.9, 2.7%]. Relations require ≥16 px and ≥10% of the smaller mask; they are descriptive overlap counts.

## Deliverables and reproduction

- `SFSC2026_Morphology_Eval.ipynb`: local/Colab CPU notebook; `morphology.py`, `frozen_geometry.py`, `figures.py`, `report_summary.py`: reusable helpers.
- `PROTOCOL.md`: exact definitions, strata, denominators and limitations.
- `results/*csv.gz`: all GT/prediction/match/record measurements, including unmatched counts; summary CSV/JSON include bootstrap CIs, paired differences and distributions.
- `results/morphology_metrics.{png,pdf}`, `morphology_strata.{png,pdf}`, `oof_examples_full.{png,pdf}`, `oof_examples_crops.{png,pdf}`: report figures. Examples are deterministic V4 per-image PQ 10th/50th/90th-percentile cases, showing each annotator separately.
- `results/oof_multi_annotator_crops.{png,pdf}` additionally uses 10th/50th/90th percentiles within the 296 multi-record images, to show differing human outlines. Its population and IDs are documented separately.
- Native disjoint instance maps: `<scratch>/masks/<stem>.npz`, keys `v4_evidence` / `v3_rescored`; hashes in `results/mask_manifest.csv`. Maps stay in scratch to keep eval/ below 30 MB. Compact decision inputs are bundled in `results/inputs/`; large OOF maps and lawful training images remain required.
- Six analytic checks pass (strict threshold/host matching, spine weighting/misses, exact raster distance/contiguous gaps, skeleton edge length, polygon overlaps, date resampling). Notebook execution and artifact-size checks are recorded in `results/validation.json`.
- Local CPU only: **0 GPU hours, 0 Colab CU**; no VM started, no commit/push, no submissions or external/test-label access.

## Interpretation limits

Spine coverage tests containment in a matched mask, not bidirectional skeleton-axis recovery. Width and length are projected proxies; skeleton branches can increase length. Shape errors condition on matching, while missed objects penalize coverage and recall. GT-dependent strata report recall/coverage because unmatched predictions lack a GT bucket. Human comparisons use multi-record images only; three-record images contribute six directed pairs. Date bootstrap conditions on frozen models and thresholds and excludes training/selection uncertainty. There are no remaining required computations.
