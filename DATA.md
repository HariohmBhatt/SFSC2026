# Dataset

## What Is Supplied

The local download contains 707 training JPEGs, 180 unlabeled competition-test
JPEGs, and one 46 MB COCO-style annotation JSON. The total local footprint is
approximately 718 MB. Every inspected image is an 8-bit, single-channel (`L`)
2048 x 2048 JPEG.

The images are GONG H-alpha solar observations. A filename such as
`20140609195854Bh.jpeg` contains a UTC-like observation timestamp followed by a
two-character instrument/site code. The six codes represented are `Bh`, `Ch`,
`Lh`, `Mh`, `Th`, and `Uh`.

The complete machine-generated assessment is in
[`reports/data_assessment.md`](reports/data_assessment.md), with raw values in
[`reports/data_assessment.json`](reports/data_assessment.json).

## Annotation Unit

This is instance segmentation, not ordinary binary semantic segmentation. One
annotation row describes one filament polygon and includes:

- a single polygon in `segmentation`;
- `area`, `bbox`, and a filament `spine`;
- a category ID;
- `iscrowd = 0`.

The JSON contains 1,154 image records but only 707 unique JPEG filenames. Of
those files, 411 have one annotation set, 145 have two, and 151 have three.
These are independent labels of the same source pixels, not extra images.

There are 8,199 annotated instances. The category counts are:

| Category | Instances |
|---|---:|
| Unidentifiable | 3,074 |
| Right | 2,590 |
| Left | 2,535 |
| Ambiguous | 0 |

The challenge target is filament instance shape, so these category labels and
the supplied spines are not assumed to be legal training targets. Competition
rules should be confirmed before using metadata beyond image pixels and allowed
segmentation masks.

## Geometry

Filaments are small and thin relative to a 4.19-megapixel frame:

| Statistic | p01 | p05 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Area (pixels) | 209 | 317 | 1,228 | 6,945 | 13,703 | 37,739 |
| Box width | 15.7 | 24.0 | 74.5 | 280.7 | 499.1 | 1,067.3 |
| Box height | 16.1 | 23.7 | 69.6 | 230.7 | 358.0 | 709.8 |
| Polygon vertices | 14 | 19 | 60 | 201 | 326 | 707 |

There are 187 instances below 256 pixels and 3,473 below 1,024 pixels. No
recorded bounding box touches the image edge. This scale distribution means
aggressive whole-image downsampling will erase valid targets. Training should
use foreground-aware crops, while inference should tile at native resolution.

## Label Ambiguity

Of 296 multiply annotated JPEGs, only 40 have the same instance count in every
annotation set; 256 disagree. The largest observed spread is 13 instances for a
single source image. Therefore:

- never place annotation sets for one JPEG into different splits;
- do not merge all polygons from all annotators into one target, because that
  can duplicate or overlap interpretations;
- initially sample one complete annotation set per image per epoch, or treat
  each annotation set as a separate training example while grouping batches and
  splits by source filename;
- evaluate every supplied annotation-image record independently, matching the
  organizer's evaluation unit, then aggregate global counts.

This disagreement is also a likely upper-bound/noise issue. Later analysis
should report performance separately for singly and multiply annotated images.

## Fixed Development Split

The versioned manifest in [`splits/split_manifest.csv`](splits/split_manifest.csv)
splits unique JPEGs as follows:

| Split | Unique JPEGs | Purpose |
|---|---:|---|
| Train | 495 | Fit model and preprocessing statistics |
| Validation | 106 | Select checkpoints, thresholds, and post-processing |
| Internal test | 106 | One-time estimate after the pipeline is frozen |

Assignment is deterministic with seed `2026`, grouped by exact source filename,
and balanced against site, year, and site-year distributions. All annotation IDs
for a filename inherit its split. The 180 Kaggle competition-test images remain
unlabeled and are not part of these 707 images.

This fixed holdout supports comparable early experiments. Once the pipeline is
stable, add grouped five-fold cross-validation on the combined train+validation
portion for robustness; do not rotate the internal test into model selection.

## Leakage And Integrity Findings

- No filename overlaps between local train and competition-test directories.
- No exact SHA-256 file-content duplicates within train, within test, or across
  train and competition test.
- Annotation references resolve to known image-record IDs.
- All polygons have one component, valid coordinate ranges, positive areas and
  boxes, and `iscrowd = 0`.
- All 707 annotated filenames exist locally.

## Modeling Consequences

- Preserve grayscale input unless a pretrained encoder requires three channels.
- Use crop sampling that deliberately includes the many tiny foreground objects.
- Stitch semantic probabilities before extracting instances; extracting per
  tile will fragment filaments at tile boundaries and directly hurt PQ.
- Track results by site, year, object area, and annotation-set multiplicity.
- Inspect false merges and false splits, not just overlap scores.
