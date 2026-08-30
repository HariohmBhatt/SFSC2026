# GPU Selection

## Final Decision

All model testing, training, validation, and inference will use a **single Kaggle
Tesla P100-PCIE-16GB**. The local machine is only for editing, Git, data audits,
metric tests, and invoking the Kaggle CLI. The RTX 3050 is not part of the
workflow.

## Verified Zero-Install Runtime

On August 30, 2026, private kernel version 5 at
[`hariohmbhatt/sfsc-p100-smoke-test`](https://www.kaggle.com/code/hariohmbhatt/sfsc-p100-smoke-test)
verified this out-of-the-box environment:

| Property | Verified value |
|---|---|
| GPU | Tesla P100-PCIE-16GB |
| Available device memory | 15,511 MB reported by TensorFlow |
| Python | 3.12.13 |
| TensorFlow | 2.20.0 |
| Driver | 580.159.04 |
| Driver CUDA capability | 13.0 |
| Competition mount | `/kaggle/input/competitions/filament-segmentation-2026` |
| 4096 x 4096 TensorFlow matrix test | Passed on `/GPU:0` in 0.239 seconds |

No package installation or internet access was used. This is the selected
framework/runtime for the simple P100-only pipeline.

## Why Default PyTorch Is Not Used

The same P100 was tested against Kaggle's preinstalled PyTorch 2.10.0+cu128 and
torchvision 0.25.0+cu128. PyTorch detected the GPU but its binary contained only
`sm_70` and newer architectures. The P100 requires `sm_60`, so Kaggle reported:

```text
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the
current PyTorch installation.
```

A pinned PyTorch 2.7.1+cu126 build was also verified successfully, but installing
it took 224 seconds and introduced package conflicts with Kaggle's default image.
TensorFlow therefore provides the requested straightforward, zero-install route.

## CLI Workflow

```bash
kaggle kernels push \
  --path kaggle/p100-smoke \
  --accelerator NvidiaTeslaP100 \
  --timeout 300

kaggle kernels status hariohmbhatt/sfsc-p100-smoke-test

kaggle kernels output \
  hariohmbhatt/sfsc-p100-smoke-test \
  --path outputs/p100-smoke \
  --force
```

Kaggle kernels are remote batch jobs, not SSH machines. Every persistent artifact
must be written under `/kaggle/working` and downloaded after completion.

## Training Envelope

Start with 768 x 768 foreground-aware crops and test 1024 x 1024 after recording
peak memory. Use mixed precision only after confirming that the P100 workload is
numerically stable; Pascal does not have Tensor Cores. Use overlapping tiled
inference at native 2048 x 2048 resolution and stitch probabilities before
extracting filament instances.

At every run, verify that TensorFlow exposes `/GPU:0`, `nvidia-smi` identifies a
P100, and a real GPU operation completes. This protects against Kaggle image or
accelerator changes.

## Sources

- [Kaggle CLI kernel documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [NVIDIA Tesla P100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-datasheet.pdf)
- [Verified runtime report](reports/p100_verification.json)
