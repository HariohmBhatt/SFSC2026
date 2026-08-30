# GPU Selection

## Exact Recommendation

Use the **local RTX 3050 4 GB for development** and prefer a **single Kaggle T4
16 GB for baseline training**. After the single-GPU pipeline is correct, use
**Kaggle T4 x2 with DistributedDataParallel** for longer sweeps or parallel
inference. Benchmark a Kaggle P100 16 GB on one representative epoch; choose it
instead only if measured epoch time is lower for the final architecture.

The decisive resource is memory per model replica, not the GPU's marketing tier.
At native 2048 x 2048 resolution, segmentation activation memory dominates model
weights. Four GB is too small for practical full-frame training and constrains
even crop training. A T4 or P100 supplies 16 GB per device.

## Hardware Trade-Off

| GPU | VRAM | Relevant strength | Role here |
|---|---:|---|---|
| RTX 3050 | 4 GB GDDR6 | Always available; Ampere Tensor Cores | Code, audits, 512 crops, batch 1, tiled inference |
| Tesla T4 | 16 GB GDDR6 | Tensor Cores and strong AMP support | Preferred baseline training and inference |
| Tesla P100 | 16 GB HBM2 | High memory bandwidth and stronger FP32 | Benchmark fallback for models that gain little from Tensor Cores |
| T4 x2 | 16 GB each | Two independent replicas | DDP training or image-parallel inference after correctness |

Two T4s do **not** form one 32 GB memory pool. Each process must fit its model,
optimizer, activations, and local batch into 16 GB. Use one process per GPU with
PyTorch `DistributedDataParallel`; do not default to `DataParallel`.

## Why T4 First

The dataset is only 707 unique training frames, so availability and disciplined
experimentation matter more than maximum cluster scale. T4 has hardware Tensor
Cores and is well suited to automatic mixed precision (AMP), while its 16 GB
memory permits materially larger crops/batches than the RTX 3050. P100 has much
higher reference memory bandwidth and good FP32 throughput but no Tensor Cores.
Actual Kaggle instances and dataloading vary, so a timed one-epoch benchmark is
the correct tie-breaker.

## Memory-Safe Starting Configuration

Start with this envelope, then measure rather than guess:

| Setting | RTX 3050 4 GB | T4/P100 16 GB |
|---|---|---|
| Crop | 512 x 512 | 768 x 768, then test 1024 |
| Physical batch | 1 | 2-4, architecture dependent |
| Precision | AMP | AMP on T4; compare AMP/FP32 on P100 |
| Effective batch | Gradient accumulation to 8 | Gradient accumulation if needed |
| Inference | 512 tiles with overlap | 768-1024 tiles with overlap |

Keep full-resolution masks on CPU, use pinned memory on CUDA hosts, and transfer
only the active crop. Stitch overlapping tile probabilities before connected
components/watershed/instance extraction. Activation checkpointing is a second
step after AMP and crop tuning, because it trades extra compute for memory.

## Required GPU Benchmark

For each candidate GPU, run the same 100 warm training steps and one validation
image with the same crop, model, physical batch, workers, and AMP setting. Record:

- median step time and images/second after warm-up;
- maximum allocated and reserved CUDA memory;
- end-to-end validation inference time;
- any out-of-memory or dataloader bottleneck;
- GPU model and PyTorch/CUDA versions from `nvidia-smi` and Python.

Select the device by reproducible throughput under the configuration that fits,
not theoretical FLOPS. Use the P100 if it wins that test; otherwise retain T4.

## Platform Strategy

- Local RTX 3050: iterate on datasets, losses, metrics, post-processing, unit
  tests, overfit-small-batch checks, and short smoke runs.
- Single Kaggle T4: establish the first serious baseline with minimal distributed
  complexity.
- Kaggle T4 x2: use DDP for confirmed pipelines, or assign independent images to
  each GPU during full-resolution inference.
- Kaggle P100: benchmark because its memory bandwidth can help convolution-heavy
  networks, but do not choose it solely from specs.

Kaggle accelerator inventory, session duration, and quota can change. Inspect the
assigned GPU with `nvidia-smi`; never assume a notebook received the selected
device. Persistent output space and session limits also require checkpointing
outside transient process memory.

## Sources

- [NVIDIA GeForce RTX 30 laptop specifications](https://www.nvidia.com/en-us/geforce/laptops/30-series/)
- [NVIDIA T4 specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [NVIDIA Tesla P100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-datasheet.pdf)
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
