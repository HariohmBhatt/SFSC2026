"""Verify zero-install TensorFlow execution on Kaggle's default P100 image."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUTPUT = Path("/kaggle/working/p100-smoke-report.json")


def write_report(report: dict[str, object]) -> None:
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


report: dict[str, object] = {
    "status": "started",
    "platform": platform.platform(),
    "python": sys.version,
    "environment": "kaggle-default",
    "installed_torch": importlib.metadata.version("torch"),
    "installed_tensorflow": importlib.metadata.version("tensorflow"),
}
write_report(report)

try:
    nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, check=True, text=True).stdout
    report["nvidia_smi"] = nvidia_smi
    assert "P100" in nvidia_smi, "Kaggle did not assign a P100"

    input_root = Path("/kaggle/input")
    report["input_mounts"] = sorted(str(path) for path in input_root.glob("*/*") if path.is_dir())
    report["competition_data_mounted"] = any(
        "filament-segmentation-2026" in path for path in report["input_mounts"]
    )
    assert report["competition_data_mounted"], "Competition data is not mounted"
    write_report(report)

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    report.update(
        {
            "tensorflow": tf.__version__,
            "tensorflow_gpus": [str(gpu) for gpu in gpus],
        }
    )
    assert gpus, "TensorFlow did not find a GPU"

    operation_started = time.perf_counter()
    with tf.device("/GPU:0"):
        left = tf.random.normal((4096, 4096))
        right = tf.random.normal((4096, 4096))
        result = tf.matmul(left, right)
        result_mean = float(tf.reduce_mean(result).numpy())
    report.update(
        {
            "matrix_result_mean": result_mean,
            "matrix_operation_seconds": round(time.perf_counter() - operation_started, 3),
            "status": "passed",
        }
    )
except Exception as error:
    report.update(
        {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    )
    raise
finally:
    write_report(report)
    print(json.dumps(report, indent=2))
