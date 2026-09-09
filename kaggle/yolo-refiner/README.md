# SFSC YOLO11l + prompted refiner

This private Kaggle P100 experiment trains on the repository's fixed 495-image
development training set. It uses the pre-existing 24-image calibration subset
for YOLO checkpointing, refiner checkpointing, and post-processing selection,
then reports the selected pipeline once on all 106 validation images and
separately on the other 82.

The detector/refiner structure is adapted from the public
`ektarr/new-metric-solar-filament` notebook. Changes include the fixed local
split, additive competition PQ, a larger detector and input, negative refiner
examples, a long-object preservation path, bounded calibration, and a P100-safe
PyTorch stack.
