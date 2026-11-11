# Frozen OOF morphology protocol

This evaluation uses only the competition **training** JSON and native 2048 × 2048
H-alpha images. It uses no external annotation files, test labels, training, GPU,
or Kaggle submissions. The notebook is an evaluation notebook; all helpers run on
CPU in local Python or Colab.

## Frozen predictions and the PQ gate

The existing `fusion_sfsc_unet_v3_aug/selection_v5.json` adopts `v4_evidence`, not
`v5_aug` or `v4_full`. Its saved cross-fitted scores `image_stats.npz` select V3
UNet candidates at tau 0.200; the comparison `v3_rescored` uses tau 0.225. The
`aug` run also evaluates other variants but does **not** adopt the augmented one.
Both selected variants use the same candidate masks; YOLO evidence changes keep
decisions, not the outlines of shared instances. Neither scorer is fitted here.

`frozen_geometry.py` copies the geometry, hole filling, relabeling, and prefilter
functions verbatim from `/tmp/v5build/fusion_flat.py` (source hash in its docstring).
The frozen recipe supplies low threshold 0.4, contact threshold 0.5, one-pixel
erosion, and the remaining geometry settings. The reconstruction only needs
max_p and area for the prefilter; its float32 max_p conversion is preserved.
Candidate order and counts must agree with saved offsets/source codes on every
image. Saved OOF scores, **never final all-data scorer outputs**, choose instances.
Each compressed scratch file contains two integer label maps (0 background,
positive pixel-disjoint instance IDs).

Every image's reconstructed IoU sum, TP, FP, FN must reproduce the saved fusion
Hungarian statistics within 1e-9 absolute / 1e-12 relative tolerance. Full-run
morphology refuses to run before this full-OOF gate passes. Annotation SHA256 and
date-disjoint folds are checked; hashes and actual score differences are saved.
The historical shipped OOF value around 0.455 is distinct from this local CPU
replica's 0.454413. These results describe the verified local replica of final V4.

## Matching and overlap

The competition host counts **every** pair with strict IoU > 0.5: TP is the number
of such pairs, FP/FN count masks with no qualifying pair, and the numerator is
the sum of their IoUs. This many-to-many convention is implemented explicitly.
Masks are rasterized by pycocotools at native resolution, exactly as in the fusion
script. Human polygons can overlap, so human records are never flattened into a
single label map. PQ = sum IoU / (TP + 0.5 FP + 0.5 FN); SQ = sum IoU / TP;
RQ = TP / (TP + 0.5 FP + 0.5 FN). Counts pool annotator records, not image PQs.

For morphology requiring one corresponding object, maximum-sum-IoU Hungarian
matching over the same IoU > 0.5 pairs gives one-to-one correspondence. Both
conventions are reported, including the number of ambiguous threshold pairs.
`matches.csv.gz` holds every host match, its IoU/Dice, and a one-to-one flag;
GT and prediction tables include **all** unmatched objects. Matched Dice is
2 IoU / (1 + IoU), not an independent detection measure. Unmatched GT entries
have zero IoU/Dice in the object table, but matched-distribution tables explicitly
condition on a match. Empty annotator records are skipped to reproduce the
fusion/competition convention; their count is recorded (zero in this dataset).

## Spines and mask morphology

The ordered supplied GT spine polyline is interpolated in arc length with
spacing <= 0.5 native pixels, including endpoints. Consecutive duplicate vertices
are removed. Each sample represents its arc-length Voronoi interval (half weight
at endpoints), so short endpoint intervals are not over-weighted. Coverage is
the fraction of GT spine length within 1/3/5 Euclidean pixels of a foreground
**raster cell centre** (x+0.5, y+0.5) in its matched predicted instance. Distances
are exact nearest-neighbour distances, not rounded sample coordinates. This
measures spine containment by the mask; it is not bidirectional skeleton-axis
agreement and does not establish downstream axis success.

An unmatched object has coverage 0 and longest uncovered fraction 1 at every
tolerance. The longest uncovered **contiguous** run sums the same sample interval
weights, divided by total GT spine length (discretization <=0.5 pixels).
Missing/malformed/zero-length spines are missing, not silently zero; their
denominators are saved. Primary spine summaries average annotation objects.
Additional image-macro summaries first average objects within each annotator
record, then records within an image, then images.

Predicted skeletons use skimage `skeletonize` on the unmodified mask. Length sums
horizontal/vertical unit edges and diagonal sqrt(2) edges, suppressing diagonal
shortcuts when either intervening orthogonal skeleton pixel already exists.
All skeleton branches and disconnected components count; no pruning or GT
information enters predicted skeletons. A one-pixel isolated component has zero
edge length, and its length-derived width is missing. These raster graph lengths
can exceed a single drawn spine's length because branches count.

Widths are area / skeleton graph length for predictions and area / supplied
spine length for GT. Predictions also retain 2 × median EDT at skeleton pixels.
The conditional length ratio compares predicted skeleton length with matched GT
spine length; absolute relative error is |ratio−1|. Width ratios compare the
area/length estimates. Elongation is sqrt(largest/smallest pixel covariance
eigenvalue), with eigenvalues floored at 1e-3. These are projected image-plane
proxies, not calibrated physical widths or deprojected solar lengths.

An overlap relation requires >=16 native pixels and >=10% of the smaller mask's
area. A GT object with >=2 prediction neighbours counts as a split relation;
a prediction with >=2 GT neighbours counts as a merge relation. Rates divide
by all GT/prediction objects **per scored record**. Relations may coexist and
do not certify physical fragmentation/connection. Tiny incidental intersections
are excluded. The criterion is fixed for description, not tuned to outcomes.

## Strata and human reference

* GT raster area buckets: <500, 500–1499, 1500–4999, >=5000 native pixels.
* GT projected width (area/spine length): <5, 5–<10, 10–<20, >=20 px;
  missing-spine objects retain a separate missing bucket.
* Radial position: GT pixel centroid distance from image centre, divided by the
  frozen preprocessing radius 0.88 × 2048 / 2. Centre <0.5R, middle 0.5–<0.8R,
  limb >=0.8R. This is a reproducible nominal disk radius, not a fitted limb.
* Site: raw GONG filename suffix, avoiding inferred site-name mappings.
* Number of annotators: number of image records sharing the JPEG.
* Inter-annotator support: fraction of **other** records containing any object
  with IoU >0.5 to this GT polygon. Buckets none/partial/all; one-record images
  are labelled separately. Support measures label agreement using evaluation
  labels and is never an input to the predictions.

Official PQ with FP/FN is stratified by image-level site/annotator count/fold.
GT area/width/radius/support strata report GT match rate, coverage, gaps, and
conditional shape errors. Unmatched predictions have no known GT area/width or
support; assigning them an arbitrary GT bucket would produce misleading PQ.
The complete prediction table retains their measured shape and unmatched flag.

On images with multiple nonempty records, each ordered pair is evaluated:
one human record supplies GT polygons/spines, another supplies predicted masks.
The identical matching, coverage, skeleton, width and split/merge routines apply.
Both directions are retained, with source and target record IDs. Three-record
images contribute six directed comparisons; two-record images contribute two.
This is an agreement reference on a different image subset, **not an upper
bound** or proof that the model is superior to a human.

## Uncertainty, examples, and limitations

95% percentile confidence intervals use 2,000 bootstrap draws of UTC calendar
dates (seed 20261024). All sites/images, annotator records, GT/prediction objects,
and directed human comparisons from a sampled date travel together. Numerators
and denominators are re-pooled in every draw. V4−V3 intervals use the same draws
for pairing. Stratum estimates use this same date universe; draws with no
stratum denominator are excluded. Thresholds, masks and examples stay frozen;
intervals do not account for training variation or threshold-selection optimism.

Poor/median/good examples are chosen before viewing overlays by nearest rank
to the 10th/50th/90th percentile of V4 per-image pooled host PQ, ties broken by
filename. The plot manifests retain all image IDs and selection scores. Full-disk
panels and deterministic crops of the largest first-record GT polygon (32 px
margin, minimum 192 px square) are both saved; all annotator records
are shown separately. These three examples illustrate the observed range and
must not be described as a random sample or a full population summary.
An additional crop panel applies the same 10th/50th/90th-percentile rule within
the 296 multi-record images, ensuring that differences between human outlines
are visible. Its separate manifest states that restricted population explicitly.

Protocol inspiration: the already-read research summary
`~/sfsc_research/r4/C/C_external.md` section 3, describing Hao/SABR's native-pixel
spine sampling and fixed tolerances. This implementation is independent and uses
only competition training annotations. It does not reproduce their dataset,
bidirectional axis criterion, or scientific claims about solar barbs.
