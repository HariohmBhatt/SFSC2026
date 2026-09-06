"""Execute a packaged selected approach; never run training on the local host."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from sfsc.contracts import require
from sfsc.dataset import Dataset, evaluate, submission
from sfsc.system import digest, object_hash, read_json, write_json


def child(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Artifact path must be text")
    path = (root / relative).resolve()
    require(
        path != root.resolve() and path.is_relative_to(root.resolve()),
        "Artifact must be inside output_dir",
    )
    require(path.exists(), f"Missing artifact: {relative}")
    return path


def artifact_hash(path: Path) -> str:
    if path.is_file():
        return digest(path)
    files = {str(p.relative_to(path)): digest(p) for p in path.rglob("*") if p.is_file()}
    require(bool(files), "Checkpoint directory is empty")
    return object_hash(files)


def upstream(spec: dict) -> Path | None:
    previous = spec.get("upstream")
    if not previous:
        return None
    candidates = []
    for path in Path("/kaggle/input").rglob("report.json"):
        try:
            report = read_json(path)
        except (ValueError, OSError):
            continue
        if report.get("run_id") == previous["run_id"] and report.get("status") == "completed":
            checkpoint = child(path.parent, report["checkpoint"])
            require(
                artifact_hash(checkpoint) == previous["checkpoint_hash"],
                "Upstream checkpoint hash differs",
            )
            candidates.append(checkpoint)
    require(len(candidates) == 1, "Expected exactly one matching upstream checkpoint mount")
    return candidates[0]


def find_dataset_root(mount: Path) -> Path:
    """Find one annotated dataset inside either flat or nested competition mounts."""
    annotations = list(mount.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    roots = [
        path.parent.parent
        for path in annotations
        if path.parent.name == "train"
        and (path.parent / "train_images").is_dir()
        and (path.parent.parent / "test/test_images").is_dir()
    ]
    require(
        len(roots) == 1,
        f"Expected one complete competition dataset below {mount}; found {len(roots)}",
    )
    return roots[0]


def main(bundle: Path) -> None:
    spec = read_json(bundle / "run.json")
    output = Path("/kaggle/working/experiment")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "run_id": spec["run_id"],
        "spec_hash": object_hash(spec),
        "phase": spec["phase"],
        "status": "started",
        "seed": spec["seed"],
        "provenance": spec["provenance"],
    }
    started = time.perf_counter()
    try:
        require(Path("/kaggle/input").is_dir(), "Model execution requires Kaggle")
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        require("P100" in gpu and len(gpu.splitlines()) == 1, "Expected one Tesla P100")
        import tensorflow as tf

        tf.config.set_soft_device_placement(False)
        devices = tf.config.list_physical_devices("GPU")
        require(len(devices) == 1, "TensorFlow must expose one GPU")
        tf.config.experimental.set_memory_growth(devices[0], True)
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.linalg.matmul(tf.ones((16, 16)), tf.ones((16, 16))))
            require(float(probe.numpy()) == 4096.0, "GPU probe failed")
        random.seed(spec["seed"])
        np.random.seed(spec["seed"])
        tf.keras.utils.set_random_seed(spec["seed"])
        tf.config.experimental.reset_memory_stats("GPU:0")
        report.update(
            gpu=gpu,
            python=platform.python_version(),
            packages={
                name: importlib.metadata.version(name)
                for name in ("tensorflow", "numpy", "scipy", "pillow", "pycocotools")
            },
        )
        mount = Path("/kaggle/input/competitions/filament-segmentation-2026")
        if not mount.exists():
            mount = Path("/kaggle/input/filament-segmentation-2026")
        root = find_dataset_root(mount)
        report["dataset_root"] = str(root)
        print(f"Dataset discovered: {root}", flush=True)
        require(
            digest(root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
            == spec["provenance"]["annotation_hash"],
            "Remote annotation snapshot differs",
        )
        data = Dataset(root, bundle / "splits/split_manifest.csv")
        phase = spec["phase"]
        train = (
            data.filenames("train")
            if phase == "evaluate"
            else (data.filenames("train", "validation") if phase == "full_train" else [])
        )
        predict = (
            data.filenames("validation")
            if phase == "evaluate"
            else (
                data.filenames("test")
                if phase == "internal_test"
                else (
                    sorted(p.name for p in (root / "test/test_images").glob("*.jpeg"))
                    if phase == "submit"
                    else []
                )
            )
        )
        context = {
            "phase": phase,
            "dataset": data,
            "train_filenames": train,
            "predict_filenames": predict,
            "output_dir": output,
            "checkpoint": upstream(spec),
            "seed": spec["seed"],
            "config": spec["config"],
        }
        source = bundle / "approach/pipeline.py"
        sys.path.insert(0, str(source.parent))
        module_spec = importlib.util.spec_from_file_location("selected_pipeline", source)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        pipeline_started = time.perf_counter()
        result = module.run(context)
        report["pipeline_seconds"] = time.perf_counter() - pipeline_started
        require(isinstance(result, dict), "run(context) must return a dictionary")
        report["peak_vram_gb"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
        report["model_seconds"] = time.perf_counter() - started
        report["vram_measurement"] = "TensorFlow allocator peak, not total device memory"
        require(report["peak_vram_gb"] <= spec["max_vram_gb"], "Measured VRAM exceeds envelope")
        if phase in {"evaluate", "full_train"}:
            require(
                report["model_seconds"] <= spec["max_training_hours"] * 3600,
                "Measured training wall time exceeds envelope",
            )
            checkpoint = child(output, result.get("checkpoint"))
            report["checkpoint_bytes"] = (
                checkpoint.stat().st_size
                if checkpoint.is_file()
                else sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
            )
            report.update(
                checkpoint=str(checkpoint.relative_to(output)),
                checkpoint_hash=artifact_hash(checkpoint),
            )
        if phase in {"evaluate", "internal_test"}:
            report["metrics"] = evaluate(
                data,
                child(output, result.get("predictions")),
                "validation" if phase == "evaluate" else "test",
            )
        elif phase == "submit":
            report["submission"] = submission(
                root, child(output, result.get("predictions")), output / "submission.csv"
            )
            report["submission_hash"] = digest(output / "submission.csv")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "report.json", report)
        print(json.dumps(report, indent=2))
