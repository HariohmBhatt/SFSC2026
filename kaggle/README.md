# Kaggle P100 Runner

The local machine only edits and submits code. All CUDA execution happens in a
private Kaggle kernel on a Tesla P100.

## Smoke Test

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

The kernel performs no package installation. It passes only after confirming a
P100, finding the mounted competition data, making TensorFlow 2.20.0 expose a GPU,
and completing a 4096 x 4096 matrix multiplication on `/GPU:0`.

Kaggle's default PyTorch 2.10.0+cu128 was tested separately and cannot execute on
the P100 because it omits Pascal `sm_60` kernels. The default TensorFlow build is
therefore the verified zero-install framework.

Every file that must survive a run must be written under `/kaggle/working`.
Kaggle kernels are batch jobs; the CLI does not provide SSH or an interactive
remote shell.

The measured results are committed in `reports/p100_verification.json`.
