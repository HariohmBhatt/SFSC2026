"""Behavioral tests for downloaded PQ-SetFormer artifact auditing."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "filament-pq-setformer/audit.py"
spec = importlib.util.spec_from_file_location("pq_setformer_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_artifacts(tmp_path: Path):
    checkpoint = tmp_path / "smoke-checkpoint.pt"
    candidates = tmp_path / "smoke-candidates.json"
    checkpoint.write_bytes(b"finite weights")
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[:2, :2] = 1
    encoded = coco_mask.encode(np.asfortranarray(mask))
    rle = {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}
    candidates.write_text(
        json.dumps(
            {
                "image.jpeg": [
                    {
                        "query_id": "q000",
                        "presence": 0.8,
                        "predicted_iou": 0.7,
                        "rle": rle,
                    }
                ]
            }
        )
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    events = [
        {"event": "optimizer_step", "step": step, "values": {"loss": 1 / step}}
        for step in range(1, 6)
    ]
    events.append({"event": "stage_terminal", "step": None, "values": {"status": "complete"}})
    (logs / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (logs / "logging-manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "wandb_run_id": "wandb-id",
                "wandb_directory": "wandb",
            }
        )
    )
    wandb_bundle = logs / "wandb/wandb/offline-run-test/run-wandb-id.wandb"
    wandb_bundle.parent.mkdir(parents=True)
    wandb_bundle.write_bytes(b"offline history")
    report = {
        "status": "complete",
        "run_id": "run",
        "source_sha256": "s" * 64,
        "membership_sha256": "m" * 64,
        "config_sha256": "c" * 64,
        "gpu": {
            "name": "Tesla P100-PCIE-16GB",
            "capability": [6, 0],
            "architectures": ["sm_60"],
            "custom_op_forward_backward": True,
        },
        "training": {
            "microbatches": 20,
            "attempted_microbatches": 20,
            "optimizer_steps": 5,
            "amp_overflows": 0,
            "last_loss": 0.2,
            "minimum_loss": 0.1,
            "maximum_loss": 1.0,
            "peak_reserved_gb": 8.0,
        },
        "memory_limit_gb": 14.5,
        "candidate_images": 1,
        "candidate_count": 1,
        "checkpoint_sha256": digest(checkpoint),
        "candidates_sha256": digest(candidates),
        "wandb_run_id": "wandb-id",
        "wandb_bundle_sha256": digest(wandb_bundle),
    }
    return report, checkpoint, candidates, logs


def test_audit_accepts_complete_finite_hashed_smoke(tmp_path) -> None:
    report, checkpoint, candidates, logs = valid_artifacts(tmp_path)

    assert audit.audit_smoke(report, checkpoint, candidates, logs) == []


def test_audit_accepts_declared_pilot_schedule(tmp_path) -> None:
    report, checkpoint, candidates, logs = valid_artifacts(tmp_path)
    report["training"].update(
        microbatches=6_000,
        attempted_microbatches=6_000,
        optimizer_steps=1_500,
    )
    events = [
        {"event": "optimizer_step", "step": step, "values": {"loss": 1 / step}}
        for step in range(1, 1_501)
    ]
    events.append({"event": "stage_terminal", "step": None, "values": {"status": "complete"}})
    (logs / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))

    assert (
        audit.audit_stage(
            report,
            checkpoint,
            candidates,
            logs,
            expected_microbatches=6_000,
            expected_optimizer_steps=1_500,
            expected_candidate_images=1,
        )
        == []
    )


def test_audit_rejects_complete_but_collapsed_or_corrupt_smoke(tmp_path) -> None:
    report, checkpoint, candidates, logs = valid_artifacts(tmp_path)
    report["candidate_count"] = 0
    report["training"]["last_loss"] = float("nan")
    checkpoint.write_bytes(b"corrupt")

    issues = audit.audit_smoke(report, checkpoint, candidates, logs)

    assert "training last_loss is missing or non-finite" in issues
    assert "candidate output is empty or incomplete" in issues
    assert "checkpoint hash mismatch" in issues


def test_audit_requires_matching_wandb_terminal_evidence(tmp_path) -> None:
    report, checkpoint, candidates, logs = valid_artifacts(tmp_path)
    manifest = json.loads((logs / "logging-manifest.json").read_text())
    manifest["wandb_run_id"] = "other"
    (logs / "logging-manifest.json").write_text(json.dumps(manifest))

    issues = audit.audit_smoke(report, checkpoint, candidates, logs)

    assert "W&B run identity mismatch" in issues


def test_audit_rejects_missing_wandb_bundle_or_excess_memory(tmp_path) -> None:
    report, checkpoint, candidates, logs = valid_artifacts(tmp_path)
    next((logs / "wandb").rglob("run-*.wandb")).unlink()
    report["training"]["peak_reserved_gb"] = 14.75

    issues = audit.audit_smoke(report, checkpoint, candidates, logs)

    assert "offline W&B bundle is missing or ambiguous" in issues
    assert "training exceeded the declared P100 memory limit" in issues


def test_audit_rejects_excessive_or_unaccounted_amp_overflows(tmp_path) -> None:
    report, checkpoint, candidates, logs = valid_artifacts(tmp_path)
    report["training"]["attempted_microbatches"] = 40
    report["training"]["amp_overflows"] = 5

    issues = audit.audit_smoke(report, checkpoint, candidates, logs)

    assert "AMP overflow accounting is invalid" in issues


def test_audit_rejects_candidate_schema_or_count_mismatch(tmp_path) -> None:
    report, checkpoint, candidates, logs = valid_artifacts(tmp_path)
    payload = json.loads(candidates.read_text())
    payload["image.jpeg"][0]["rle"]["size"] = [32, 32]
    candidates.write_text(json.dumps(payload))
    report["candidates_sha256"] = digest(candidates)
    report["candidate_count"] = 2

    issues = audit.audit_smoke(report, checkpoint, candidates, logs)

    assert "candidate-cache counts do not match the report" in issues
    assert "candidate cache contains an invalid query or RLE" in issues


def test_audit_rejects_unassemblable_v2_probability_payload(tmp_path) -> None:
    report, checkpoint, candidates, logs = valid_artifacts(tmp_path)
    payload = json.loads(candidates.read_text())
    row = payload["image.jpeg"][0]
    row.update(
        schema_version=2,
        source="identity",
        presence_logit=1.0,
        probability_codec="q8-zlib-v1",
        probability_shape=[1, 1],
        probability_native_shape=[2048, 2048],
        probability_payload="not-base64",
        probability_payload_sha256="0" * 64,
    )
    candidates.write_text(json.dumps(payload))
    report["candidates_sha256"] = digest(candidates)

    assert "candidate cache contains an invalid query or RLE" in audit.audit_smoke(
        report, checkpoint, candidates, logs
    )
