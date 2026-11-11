# Dependency evidence and installation limits

Audited 2026-10-24. These are pinned **direct dependencies**, not a fully resolved, hash-locked historical environment. The original notebooks use lower bounds and install only missing modules. Neither `environment_v3.txt` nor `environment_v4.txt` was available in the inspected local mirror; those files are written to Drive by the notebooks. Recover them before claiming exact environment reconstruction.

| Dependency / runtime | Pin | Evidence |
|---|---|---|
| Python | 3.13.15 observed; use a compatible Python 3.13 runtime | V1/V2 manifests and V4 YOLO training output; V1 notebook metadata incorrectly says 3.10 |
| Torch | 2.11.0, CUDA overlay `2.11.0+cu130` | V1–V4 install-cell outputs |
| Torchvision | 0.26.0, CUDA overlay `0.26.0+cu130` | V1–V3 install-cell outputs |
| segmentation-models-pytorch | 0.5.0 | V3 Cell 02 output |
| timm | 1.0.30 | V3 Cell 02 output |
| scikit-learn | 1.6.1 | V3/V4 Cell 02 outputs; important for serialized HGB scorers |
| Ultralytics | 8.4.175 | V4 Cell 02 output and detector training banner |
| OpenCV | `cv2` 4.14.0; `opencv-python` 4.14.0.94 | V1/V2 manifests record module version; the corresponding PyPI release uses build suffix .94. Original distribution/build suffix not retained. Choose `opencv-python` because Ultralytics requires that distribution by name; adding only headless would pull in a second, unpinned cv2 provider |
| NumPy | 2.4.4 | Existing `uv.lock`; reconstructed selection, not established as Colab's run version |
| SciPy | 1.17.1 | Existing `uv.lock`; reconstructed selection |
| Pillow | 12.2.0 | Existing `uv.lock`; reconstructed selection |
| pycocotools | 2.0.11 | Existing `uv.lock`, also local CPU fusion freeze |
| packaging | 26.3 | Existing `uv.lock`; reconstructed selection |
| pandas | 3.0.5 | Local CPU V5 fusion `environment_v5.txt`; not the GPU environment |
| matplotlib | 3.10.8 | Explicit compatible reconstruction choice; original Colab version not retained |
| tqdm | 4.67.1 | Explicit compatible reconstruction choice; original Colab version not retained |

The checked-in `uv.lock` is a small local evaluation/development environment (NumPy, SciPy, Pillow, pycocotools, pytest, Ruff); it omits Torch, SMP, timm and Ultralytics and declares Python `>=3.11,<3.13`. It is **not** a lock for the final notebooks and must not be used with `uv sync` to reproduce GPU training. Do not change it without resolving the intended scope. Its package versions do not establish what was installed in Colab.

## Colab procedure

Mount Drive and run `%pip install -r /content/drive/MyDrive/SFSC2026/repo/requirements-colab-cuda.txt` in a fresh L4 runtime. Restart the session before importing NumPy, OpenCV or Torch. Record `sys.version`, `torch.__version__`, `torchvision.__version__`, `torch.version.cuda`, the GPU name, `nvidia-smi`, `pip freeze` and `pip check`. Verify the desired encoder resolves without fallback and run each notebook's synthetic tests plus a separate debug run.

The CUDA overlay reflects the **observed** run, rather than assuming the latest Colab default. A runtime without a compatible CUDA driver cannot reproduce that GPU build merely by installing Python packages. Select a compatible Colab runtime image when available and document it. No GPU runtime was allocated and no CUDA installation was attempted in this task.

Colab's `google.colab` integration and NVIDIA driver belong to the hosted runtime and are intentionally not installed through requirements. Python standard-library imports need no pip packages. The V5 optional rclone workflow is a system binary plus private configuration, not a required final-pipeline dependency. Use Drive mounting for the documented workflow. The benchmark uses only standard-library notebook JSON parsing; it does not require Jupyter on the Colab kernel.

A fresh resolver/install check and original Drive freezes remain release checks. All selected direct package versions were checked against PyPI release metadata, Python 3.13 support and one another's declared requirements; this is not a full installation or transitive resolution. Colab bundles `opencv-python`, `opencv-contrib-python` and/or headless builds on some images: keep one coherent `cv2` provider and record the choice rather than silently installing multiple distributions. Transitive packages, CUDA wheels and runtime system libraries still need a captured environment for archival reproducibility.
