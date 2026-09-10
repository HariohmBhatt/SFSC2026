"""Independent integrity checks for downloaded PQ-SetFormer smoke artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_candidate(row: Any) -> bool:
    if not isinstance(row, Mapping):
        return False
    if not isinstance(row.get("query_id"), str) or not row["query_id"]:
        return False
    for field in ("presence", "predicted_iou"):
        if not _finite(row.get(field)) or not 0 <= row[field] <= 1:
            return False
    rle = row.get("rle")
    if (
        not isinstance(rle, Mapping)
        or rle.get("size") != [2048, 2048]
        or not isinstance(rle.get("counts"), str)
    ):
        return False
    try:
        decoded = coco_mask.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
    except Exception:
        return False
    return decoded.shape == (2048, 2048) and bool(np.asarray(decoded).any())


def audit_smoke(
    report: Mapping[str, Any],
    checkpoint: Path,
    candidates: Path,
    logging_directory: Path,
) -> list[str]:
    issues = []
    if not isinstance(report, Mapping):
        return ["report payload is not an object"]
    if report.get("status") != "complete":
        issues.append("smoke report is not complete")
    gpu = report.get("gpu", {})
    if (
        not isinstance(gpu, Mapping)
        or "P100" not in str(gpu.get("name", ""))
        or gpu.get("capability") != [6, 0]
        or "sm_60" not in gpu.get("architectures", [])
        or gpu.get("custom_op_forward_backward") is not True
    ):
        issues.append("P100 sm_60 evidence is missing")
    training = report.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    for name in ("last_loss", "minimum_loss", "maximum_loss"):
        if not _finite(training.get(name)):
            issues.append(f"training {name} is missing or non-finite")
    if training.get("microbatches") != 20 or training.get("optimizer_steps") != 5:
        issues.append("training did not complete the frozen 20/5 smoke schedule")
    peak_memory = training.get("peak_reserved_gb")
    memory_limit = report.get("memory_limit_gb")
    if not _finite(peak_memory) or not _finite(memory_limit) or peak_memory > memory_limit:
        issues.append("training exceeded the declared P100 memory limit")
    if report.get("candidate_images", 0) < 1 or report.get("candidate_count", 0) < 1:
        issues.append("candidate output is empty or incomplete")

    checkpoint = Path(checkpoint)
    candidates = Path(candidates)
    if not checkpoint.is_file() or report.get("checkpoint_sha256") != digest(checkpoint):
        issues.append("checkpoint hash mismatch")
    if not candidates.is_file() or report.get("candidates_sha256") != digest(candidates):
        issues.append("candidate-cache hash mismatch")
    elif candidates.is_file():
        try:
            cache = json.loads(candidates.read_text())
            if not isinstance(cache, dict) or not any(cache.values()):
                issues.append("candidate cache is collapsed")
            elif len(cache) != report.get("candidate_images") or sum(
                len(rows) for rows in cache.values() if isinstance(rows, list)
            ) != report.get("candidate_count"):
                issues.append("candidate-cache counts do not match the report")
            if isinstance(cache, dict) and any(
                not isinstance(rows, list)
                or len({row.get("query_id") for row in rows if isinstance(row, Mapping)})
                != len(rows)
                or any(not _valid_candidate(row) for row in rows)
                for rows in cache.values()
            ):
                issues.append("candidate cache contains an invalid query or RLE")
        except (json.JSONDecodeError, AttributeError):
            issues.append("candidate cache is malformed")

    logging_directory = Path(logging_directory)
    manifest_path = logging_directory / "logging-manifest.json"
    events_path = logging_directory / "events.jsonl"
    if not manifest_path.is_file() or not events_path.is_file():
        issues.append("offline W&B/JSONL evidence is missing")
        return issues
    try:
        manifest = json.loads(manifest_path.read_text())
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    except json.JSONDecodeError:
        issues.append("offline W&B/JSONL evidence is malformed")
        return issues
    if manifest.get("status") != "complete":
        issues.append("offline W&B run is not complete")
    if manifest.get("wandb_run_id") != report.get("wandb_run_id"):
        issues.append("W&B run identity mismatch")
    wandb_directory = Path(str(manifest.get("wandb_directory", "")))
    if wandb_directory.is_absolute() or ".." in wandb_directory.parts:
        issues.append("offline W&B directory is unsafe")
    else:
        bundles = list((logging_directory / wandb_directory).rglob("run-*.wandb"))
        if len(bundles) != 1:
            issues.append("offline W&B bundle is missing or ambiguous")
        elif report.get("wandb_bundle_sha256") != digest(bundles[0]):
            issues.append("offline W&B bundle hash mismatch")
    if (
        not events
        or events[-1].get("event") != "stage_terminal"
        or events[-1].get("values", {}).get("status") != "complete"
    ):
        issues.append("JSONL terminal evidence is missing")
    optimizer_steps = [
        event.get("step") for event in events if event.get("event") == "optimizer_step"
    ]
    if optimizer_steps != list(range(1, 6)):
        issues.append("optimizer-step logging is incomplete or non-monotonic")
    return issues
