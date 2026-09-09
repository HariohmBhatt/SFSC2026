"""Exercise the exact YOLO segmentation training stack on a Kaggle P100."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path("/kaggle/working/yolo-p100-smoke.json")
WORK = Path("/kaggle/working/yolo-smoke")
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"


def save(report: dict[str, object]) -> None:
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def install() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--force-reinstall",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "--index-url",
            TORCH_INDEX,
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "ultralytics==8.4.103",
        ]
    )


def discover_data() -> tuple[Path, Path]:
    candidates = list(
        Path("/kaggle/input").glob(
            "**/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one annotation file, found {candidates}")
    annotation_path = candidates[0]
    return annotation_path, annotation_path.parent / "train_images"


def make_tiny_dataset(annotation_path: Path, image_dir: Path) -> Path:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations: dict[str, list[dict[str, object]]] = {}
    for annotation in data["annotations"]:
        annotations.setdefault(str(annotation["image_id"]), []).append(annotation)
    records = []
    used = set()
    for record in data["images"]:
        filename = str(record["file_name"])
        if filename in used or not annotations.get(str(record["id"])):
            continue
        records.append(record)
        used.add(filename)
        if len(records) == 2:
            break
    for split in ("train", "val"):
        images = WORK / split / "images"
        labels = WORK / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for record in records:
            filename = str(record["file_name"])
            destination = images / filename
            if not destination.exists():
                os.symlink(image_dir / filename, destination)
            lines = []
            width, height = float(record["width"]), float(record["height"])
            for annotation in annotations[str(record["id"])]:
                for polygon in annotation.get("segmentation", []):
                    if len(polygon) < 6:
                        continue
                    values = []
                    for index, value in enumerate(polygon):
                        denominator = width if index % 2 == 0 else height
                        values.append(min(1.0, max(0.0, float(value) / denominator)))
                    lines.append("0 " + " ".join(f"{value:.7f}" for value in values))
            (labels / f"{Path(filename).stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
    yaml_path = WORK / "data.yaml"
    yaml_path.write_text(
        f"path: {WORK}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n",
        encoding="utf-8",
    )
    return yaml_path


report: dict[str, object] = {
    "status": "started",
    "platform": platform.platform(),
    "python": sys.version,
    "torch_index": TORCH_INDEX,
}
save(report)

try:
    started = time.perf_counter()
    install()
    report["install_seconds"] = round(time.perf_counter() - started, 3)
    save(report)

    import cv2
    import torch
    import torchvision
    from torchvision.ops import nms, roi_align
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    smi = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True
    ).stdout
    assert "P100" in smi, "Kaggle did not assign the requested P100"
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.get_device_capability() == (6, 0)

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]], device="cuda"
    )
    scores = torch.tensor([0.9, 0.8], device="cuda")
    kept = nms(boxes, scores, 0.5)
    features = torch.randn(1, 4, 32, 32, device="cuda", requires_grad=True)
    rois = torch.tensor([[0, 2, 2, 24, 24]], dtype=torch.float32, device="cuda")
    pooled = roi_align(features, rois, (7, 7))
    pooled.sum().backward()
    assert kept.tolist() == [0] and torch.isfinite(features.grad).all()

    annotation_path, image_dir = discover_data()
    yaml_path = make_tiny_dataset(annotation_path, image_dir)
    train_started = time.perf_counter()
    model = YOLO("yolo11n-seg.pt")
    result = model.train(
        data=str(yaml_path),
        epochs=1,
        imgsz=320,
        batch=1,
        workers=0,
        device=0,
        amp=True,
        plots=False,
        cache=False,
        project=str(WORK / "runs"),
        name="actual_backward",
        exist_ok=True,
        verbose=True,
    )
    prediction = model.predict(
        str(next((WORK / "val/images").glob("*.jpeg"))),
        imgsz=1024,
        conf=0.01,
        retina_masks=True,
        device=0,
        verbose=False,
    )[0]
    report.update(
        {
            "status": "passed",
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics_version,
            "opencv": cv2.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "nms_kept": kept.tolist(),
            "roi_align_backward": True,
            "training_seconds": round(time.perf_counter() - train_started, 3),
            "training_results": str(result.save_dir),
            "prediction_instances": 0
            if prediction.boxes is None
            else len(prediction.boxes),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "nvidia_smi": smi,
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
    save(report)
    print(json.dumps(report, indent=2))
