# Filament ResUNet — first competition model

Selected 2026-09-05 from six academic approach families. This is a strong first
experimental candidate, not a claim of leaderboard quality. SFSC validation PQ,
training throughput and peak GPU memory remain unmeasured until the remote run.

## Architecture and rationale

A randomly initialized residual U-Net predicts foreground, instance interior and
one-pixel inner contours. Encoder widths are 24, 48, 96, 192 and 384, with four
2× reductions, residual blocks, group normalization, bilinear skip decoding and
three full-resolution output logits. Inputs are grayscale intensity and a local
contrast channel derived solely from that same image. No external weights,
spines, supplied category labels or supplied bounding boxes are used. Sampling
bounds and auxiliary labels are derived from allowed segmentation polygons.

This combines U-Net's localization/context design, residual optimization, and
DCAN's region/contour separation idea. It is an SFSC adaptation, not a reproduction
of any paper. Long, bent filament masks need not be star-convex; a dense mask
model avoids imposing that geometric constraint. Center-offset approaches and
Mask R-CNN are retained as later alternatives if the dense model fails to
separate instances adequately.

Academic sources:

- Ronneberger, Fischer, Brox. U-Net: Convolutional Networks for Biomedical Image
  Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597
- Zhang, Liu, Wang. Road Extraction by Deep Residual U-Net. IEEE GRSL 2018.
  https://arxiv.org/abs/1711.10684
- Chen et al. DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation.
  CVPR 2016. https://arxiv.org/abs/1604.02677
- Isensee et al. nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation. Nature Methods 2021.
  https://arxiv.org/abs/1904.08128

## Training

Only the 495 training source filenames enter optimization. Each sample chooses
one whole annotation set; annotators are never unioned. Sample 768×768 crops,
70% centered near a randomly chosen polygon and 30% uniformly sampled. Use mild
0.85–1.15 scaling, dihedral flips/rotations, brightness/contrast/gamma changes and
occasional mild blur. A halo prevents crop boundaries becoming artificial contour
targets. Thin objects whose eroded interiors disappear keep their full mask as
a seed target.

Batch size 2, float32, AdamW, initial learning rate 3e-4 with cosine decay to
2e-5, weight decay 1e-4, global gradient norm clipping at 1. Up to 60 epochs of
200 steps; loss combines weighted BCE and Dice with channel weights 1/.5/.25
for foreground/interior/boundary. No topology loss in the first run: clDice is a
follow-up ablation if fragmentation dominates. Seeds are fixed at 2026.

Training stops at an epoch boundary after its six-hour budget, reserving the
remaining portion of the nine-hour Kaggle envelope for calibration and complete
validation. This is an upper bound, not an estimated completion time. Saving
last and best weights protects progress; logs and progress.json record real
optimizer steps and memory. A 14 GB TensorFlow allocator check fails explicitly
if the model exceeds the configured envelope.

## Checkpoint selection and instances

After epoch 1 and every fifth epoch, score PQ on 12 validation images (two
spread across dates per site), summing each annotator record independently.
Select the best checkpoint by this PQ, not crop Dice. After training, calibrate
foreground/interior thresholds and minimum area on 24 validation images
(four per site), then evaluate the selected recipe on all 106 validation images.
All subsets are deterministic and remain inside the fixed validation partition.
The reported 106-image score is development validation, not an unbiased holdout.

Infer overlapping 768-pixel tiles at stride 512 with horizontal-flip test-time
augmentation. Blend all three probability maps at original 2048×2048 resolution
before extracting instances. Foreground connected components receive seeds from
predicted interiors suppressed by predicted contours. Seedless components remain
single candidate instances; components with multiple surviving seeds are split
by marker-controlled watershed using contour/foreground costs. Minimum-area
candidates are 8, 32 and 64 pixels, allowing the smallest audited objects to
survive calibration. No unconstrained local-maxima seeds are introduced.

The accelerated calibration PQ uses label histograms but matches the existing
strict IoU > 0.5 evaluator, including overlapping truth masks. Regression tests
check this equivalence. Final scoring uses the system's fixed evaluator.

## Saved recipe and later phases

`checkpoint/` contains best/last weights, best checkpoint evidence,
calibration.json, and recipe.json with selected epoch, config and postprocessing.
Internal-test inference reloads that exact recipe. Full training reads its frozen
recipe but initializes weights anew on train+validation, runs exactly the selected
number of epochs and preserves the original learning-rate schedule. Submission
loads the full-training checkpoint and produces integer instance maps; the
system owns final COCO RLE encoding. The internal test is not touched by this run.

## Tests and limitations

Tests cover crop-edge contours, tiny-instance interiors, ignored metadata,
complete tiling, seedless/empty masks, touching-instance separation and PQ parity.
Real crop shape/finite-value checks were performed on 13 training images.
TensorFlow graph execution and P100 memory are checked by the first remote
optimizer steps; local tests do not claim to validate those.

Boundary errors can still split or merge filaments. Repeated annotator disagreement
remains irreducible label noise. Moderate crop scaling can affect the narrowest
structures. No competitive performance is asserted until measured PQ and error
diagnostics are available. Current official overview confirms PQ; final submission
still requires source-level organizer evaluator comparison for edge cases.
