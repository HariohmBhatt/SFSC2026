from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch

from .config import load_config, paths, sha256
from .data import decode_rle
from .ensemble import Candidate, paint_instances
from .inference import encode, load_refiners, load_semantic, predict_candidates
from .metrics import annotation_sets, score
from .training import (
    emit_progress,
    generate_oof_prompts,
    prepare_data,
    read_split,
    seed_everything,
    train_detectors,
    train_refiners,
    train_semantic,
)

HERE = Path(__file__).resolve().parent


def mount_drive() -> None:
    mount = Path("/content/drive/MyDrive")
    if not mount.is_dir():
        try:
            from google.colab import drive
        except ImportError as error:
            raise RuntimeError(
                "This entrypoint must run in Google Colab with Drive mounted"
            ) from error
        drive.mount("/content/drive", force_remount=False)
    if not mount.is_dir():
        raise RuntimeError("Google Drive mount did not become available")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def snapshot_code(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in HERE.glob("*.py"):
        shutil.copy2(source, destination / source.name)
    shutil.copy2(HERE / "config.json", destination / "config.json")


def checkpoint_paths(store: Path) -> tuple[list[Path], list[Path], Path]:
    detectors = sorted(
        path
        for path in store.glob("detectors/**/weights/best.pt")
        if (path.parent.parent / "complete.json").is_file()
    )
    refiners = sorted(
        path
        for path in (store / "refiners").glob("*.pt")
        if not path.name.endswith(".last.pt") and path.with_suffix(".complete.json").is_file()
    )
    semantic = store / "semantic/full-disk.pt"
    semantic_complete = semantic.with_suffix(".complete.json")
    if not detectors or not refiners or not semantic.is_file() or not semantic_complete.is_file():
        raise FileNotFoundError("Complete detector, refiner, and semantic checkpoints are required")
    return detectors, refiners, semantic


def validate(
    config: dict, store, data_root: Path, split: dict[str, list[str]], coco: dict, output: Path
) -> dict:
    from ultralytics import YOLO

    detector_paths, refiner_paths, semantic_path = checkpoint_paths(store.checkpoints)
    expected_detectors = len(config["detectors"])
    expected_refiners = len(config["refiner_backbones"]) * int(config["refiner_folds"])
    if len(detector_paths) != expected_detectors or len(refiner_paths) != expected_refiners:
        raise RuntimeError(
            f"Incomplete ensemble: {len(detector_paths)}/{expected_detectors} detectors and "
            f"{len(refiner_paths)}/{expected_refiners} refiners"
        )
    model_hashes = {
        "detectors": {str(path): sha256(path) for path in detector_paths},
        "refiners": {str(path): sha256(path) for path in refiner_paths},
        "semantic": {str(semantic_path): sha256(semantic_path)},
    }
    signature = hashlib.sha256(
        json.dumps({"config": config, "models": model_hashes}, sort_keys=True).encode()
    ).hexdigest()[:20]
    validation_cache = store.cache / "validation" / signature
    detectors = [YOLO(str(path)) for path in detector_paths]
    refiners = load_refiners(refiner_paths)
    semantic = load_semantic(semantic_path, config["semantic_backbone"])
    cache: dict[str, list[Candidate]] = {}
    diagnostics = {}
    image_dir = data_root / "train/train_images"
    for index, filename in enumerate(split["validation"], 1):
        cache_path = validation_cache / f"{Path(filename).stem}.json"
        if cache_path.is_file():
            saved = json.loads(cache_path.read_text())
            cache[filename] = [
                Candidate(
                    decode_rle(row["mask"], tuple(row["mask"]["size"])),
                    float(row["score"]),
                    row["source"],
                    float(row["agreement"]),
                )
                for row in saved["candidates"]
            ]
            diagnostics[filename] = saved["diagnostics"]
        else:
            cache[filename], diagnostics[filename] = predict_candidates(
                image_dir / filename, detectors, refiners, semantic, config
            )
            save_json(
                cache_path,
                {
                    "filename": filename,
                    "diagnostics": diagnostics[filename],
                    "candidates": [
                        {
                            "mask": encode(row.mask),
                            "score": row.score,
                            "source": row.source,
                            "agreement": row.agreement,
                        }
                        for row in cache[filename]
                    ],
                },
            )
        emit_progress(
            "progress",
            phase="validate",
            current=index,
            total=len(split["validation"]),
            filename=filename,
            **diagnostics[filename],
        )
    truth_all = annotation_sets(coco)
    truth = {name: truth_all[name] for name in split["validation"]}
    grid = []
    best = None
    best_predictions = None
    for threshold in config["quality_thresholds"]:
        for minimum_area in config["minimum_areas"]:
            predictions = {
                name: [
                    encode(row.mask)
                    for row in paint_instances(rows, float(threshold), int(minimum_area))
                ]
                for name, rows in cache.items()
            }
            metrics = score(predictions, truth)
            row = {"score_threshold": threshold, "minimum_area": minimum_area, **metrics}
            grid.append(row)
            rank = (metrics["pq"], -metrics["fp"], threshold)
            if best is None or rank > best:
                best, best_predictions = rank, predictions
    selected = max(grid, key=lambda row: (row["pq"], -row["fp"], row["score_threshold"]))
    report = {
        "status": "complete",
        "selected": selected,
        "grid": grid,
        "diagnostics": diagnostics,
        "models": model_hashes,
        "validation_cache_signature": signature,
    }
    save_json(output / "validation-report.json", report)
    save_json(output / "validation-predictions.json", best_predictions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive-first SFSC high-recall model ensemble")
    parser.add_argument(
        "--phase",
        choices=("detectors", "prompts", "refiners", "semantic", "validate", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    mount_drive()
    config = load_config(args.config)
    store = paths(config)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA Colab runtime is required")
    seed_everything(int(config["seed"]))
    data_root = prepare_data(config, store)
    split_path = store.drive / config["split_manifest"]
    if not split_path.is_file():
        raise FileNotFoundError(f"Split manifest must be stored on Drive: {split_path}")
    split = read_split(split_path)
    annotation_path = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(annotation_path.read_text())
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output = store.persistent / run_id
    output.mkdir(parents=True, exist_ok=False)
    snapshot_code(output / "source")
    record = {
        "status": "running",
        "phase": args.phase,
        "run_id": run_id,
        "gpu": torch.cuda.get_device_name(),
        "config": config,
        "data_manifest_sha256": sha256(store.drive / config["data_manifest"]),
        "split_manifest_sha256": sha256(split_path),
    }
    save_json(output / "run.json", record)
    emit_progress("pipeline_start", phase=args.phase, run_id=run_id)
    try:
        if args.phase in {"detectors", "all"}:
            emit_progress("phase_start", phase="detectors", total=len(config["detectors"]))
            train_detectors(config, store, data_root, split, coco)
            emit_progress("phase_complete", phase="detectors")
        if args.phase in {"prompts", "all"}:
            emit_progress(
                "phase_start", phase="prompts", total=int(config["prompt_oof_folds"])
            )
            generate_oof_prompts(config, store, data_root, split, coco)
            emit_progress("phase_complete", phase="prompts")
        if args.phase in {"refiners", "all"}:
            emit_progress(
                "phase_start",
                phase="refiners",
                total=len(config["refiner_backbones"]) * int(config["refiner_folds"]),
            )
            train_refiners(config, store, data_root, split, coco)
            emit_progress("phase_complete", phase="refiners")
        if args.phase in {"semantic", "all"}:
            emit_progress(
                "phase_start", phase="semantic", total=int(config["semantic_epochs"])
            )
            train_semantic(config, store, data_root, split, coco)
            emit_progress("phase_complete", phase="semantic")
        if args.phase in {"validate", "all"}:
            emit_progress("phase_start", phase="validate", total=len(split["validation"]))
        report = (
            validate(config, store, data_root, split, coco, output)
            if args.phase in {"validate", "all"}
            else None
        )
        if args.phase in {"validate", "all"}:
            emit_progress(
                "phase_complete",
                phase="validate",
                pq=report and report["selected"]["pq"],
            )
        record.update(status="complete", validation=report and report["selected"])
        save_json(output / "run.json", record)
        emit_progress(
            "pipeline_complete",
            phase=args.phase,
            run_id=run_id,
            pq=report and report["selected"]["pq"],
        )
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        save_json(output / "run.json", record)
        emit_progress("pipeline_failed", phase=args.phase, run_id=run_id, error=repr(error))
        raise


if __name__ == "__main__":
    main()
