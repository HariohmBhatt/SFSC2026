# Compute and efficiency evidence

Audited 2026-10-24. No GPU was started and no Colab compute units were consumed by this documentation/benchmark preparation task. Distinguish **measured log totals**, **historical timing notes** and **planning estimates** below. GPU-hours sum occupied GPU time across sessions; two parallel folds reduce wall time but not aggregate GPU-hours. Dataset copying, installation, preprocessing, calibration, retries and idle VM time are not automatically included in epoch totals.

## Final V4 pipeline

| Stage | Hardware | Cost evidence | Interpretation |
|---|---|---|---|
| V3 four-fold UNet training | NVIDIA L4, BF16, effective batch 8 | V5 loss-pilot notebook's opening markdown records a V3 fold-0 log rate of **~130 s/epoch**; 40 epochs/fold | **Inferred**, not recovered full-run total: `4 × 40 × 130 / 3600 = 5.78 L4-h` for training epochs, plus evaluation/checkpoints/preflight. Actual V3 notebook retained output only says all folds were already trained and skipped. |
| V3 OOF maps (707 images) + test dense maps (180 images) | L4 | D4 TTA, 9 tiles/image, tile 1024, stride 768; one held-out model per OOF image, four models per test image | No trustworthy retained wall-clock duration. The pilot planning note suggests ~10 min for 177-image single-model OOF; extrapolating gives ~40 min OOF + ~41 min test, **~1.36 L4-h inferred**, excluding IO and other differences. This is not a published measured throughput. |
| V3 training + evaluation/maps/calibration overall | L4 + CPU | Components above; overhead unmeasured | Plan roughly **7–9 h**, not a measured total. Recover V3 `fold*/history.csv` and environment/manifest to replace the estimate. |
| V4 YOLO11m-seg training, folds 0/1/2/3 | **NVIDIA L4**, 1280 pixels, batch 4, 40 epochs | V4 JSON code cell 20 (visible Cell 10) prints **0.955 / 0.951 / 0.953 / 0.957 h** | **Measured, rounded log sum = 3.816 L4 GPU-h**. Banners identify NVIDIA L4 with 22,563 MiB. Excludes staging, detector predictions and fusion. |
| V4 detector prediction cache | L4 | 707 held-out image passes + `4 × 180` test passes; 1,427 total image/model predictions | No complete retained duration; not inferred from unrelated YOLO model speeds. |
| V4 candidate extraction / HGB scorer / export | CPU High-RAM | Maps and detector outputs cached; no neural training needed | **0 required GPU-h** if GPU runtime is disconnected/replaced by CPU. Budget 15–60 min CPU as an unvalidated planning allowance. Running CPU work inside an allocated L4 still consumes GPU session time. |
| Complete fresh training reproduction | L4 plus CPU | 5.78 inferred UNet training + 1.36 inferred dense maps + 3.816 measured detector training | Partial subtotal **~10.95 L4-h**, mixed evidence; incomplete because validation, detector prediction and IO overhead are missing. Use **~11–14 L4-h as a planning allowance only**, not an audited training bill. |

V3 training volume is `4 folds × 40 epochs × 250 steps × effective batch 8 = 320,000 crops` of size 768². Full dense inference is `9 tiles × 8 transforms × 4 UNets = 288 tile-model forward passes` per test image, batched four tiles at a time; the 707 OOF images use 72 passes each. V4 adds four detector image passes per test image. These are algorithmic counts, not timing measurements.

The local notes use ~1.54 CU per L4-hour as a historical billing estimate; Colab rates can vary. At that rate the **measured detector training alone** corresponds to ~5.88 CU, and the mixed-evidence 11–14 h plan corresponds to ~17–22 CU. Neither is an observed invoice or current guaranteed rate. No remaining account balance was queried in this task.

## Baselines and rejected experiments

- V1/V2 are not required for the final solution. V3's comparison table describes V2 as ~30 min training with ~2.4 GiB peak; treat this as a historical summary rather than an end-to-end timed reproduction. Existing V1/V2 epoch outputs can be summed separately if needed.
- V5 high-recall seed folds took **~29 min/fold on G4**, per historical memory. This is **not L4 timing**, and there is no justified hardware conversion. No G4 rerun is requested.
- V5 loss pilot consumed **3.6 CU on L4 including debugging**, per memory, equivalent to ~2.34 occupied L4-h only if the historical 1.54 CU/h estimate applies. This is experiment cost, not V3 final training cost.
- V5 ResNet34 step consumed **~9 CU**, per memory. Locally retained history rows sum to **4.0645 h training + 0.3370 h validation** over 139 epochs (40/19/40/40 retained rows); fold 1 is incomplete despite `done.json`, so this is a **partial logged total**, not four-fold complete cost or the full billed session time. OOF/test inference and transfer are additional.
- Rotated and square refiner experiments have no complete retained L4 cost record in the inspected evidence. Do not fill them with unrelated hardware timings.

## Inference benchmark to run

No complete raw-image-to-final-RLE timing or trustworthy peak GPU memory for V4 inference was found. The new [Colab inference benchmark](../tools/SFSC2026_Inference_Benchmark.ipynb) prepares that measurement without training or modifying the final model.

Use a **fresh L4 runtime**, the pinned dependencies, a staged repository, the competition test JPEG directory, all four V3 `best.pt` checkpoints, all four V4 detector `best.pt` checkpoints, `frozen_recipe_v3.json`, V3 `fold0/config.json`, V4 `manifest_v4.json`, and the trusted `scorer_v4.pkl`. Set paths in the notebook. It loads selected existing inference definitions, never executes either notebook's training or candidate-fitting cells, checks artifact/feature/recipe compatibility, warms up one image, then measures a deterministic spread of five test images for two repeats.

Expected **planning allocation: 5–15 minutes after dependencies and assets are staged**, not measured. Cold dependency installation and Drive copies can take longer. No GPU execution was performed here. Terminate the runtime when finished; keep at most two active L4 sessions if the lead also runs another permitted training task.

The benchmark reports:

- Raw JPEG read/flattening, four-UNet D4 inference, four-detector predictions, fusion/scoring, RLE encoding and total synchronized elapsed seconds/image.
- Median/p90 and throughput, peak CUDA allocated/reserved GiB (including loaded model residency), and sampled process peak RSS. CUDA allocator metrics are not the same as total `nvidia-smi` usage; host RSS is a process high-water mark.
- Cold model-loading time, GPU/runtime/package versions, notebook/checkpoint/scorer hashes and configuration. Model load/staging and dependency installation are excluded from warm per-image latency and explicitly recorded separately.
- Optional agreement with cached V3 maps and detector predictions, plus geometry/keep-scorer/RLE sanity checks. The timed output is built from raw images, not loaded cached masks, and CSV/Drive writes are outside the timed region.

Run an initial one-image/one-repeat pilot if memory is tight. Do not silently drop folds, TTA transforms or image resolution to obtain a faster headline. The notebook intentionally fails on missing artifacts or incompatible feature/geometry metadata. GPU verification and replacing this section with actual results are open release items.
