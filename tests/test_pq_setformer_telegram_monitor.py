"""Behavioral checks for PQ-SetFormer's local Telegram relay."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "filament-pq-setformer"


def _load_monitor():
    spec = importlib.util.spec_from_file_location(
        "pq_setformer_telegram_monitor", HERE / "telegram_monitor.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeNotifier:
    enabled = True

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.logs: list[tuple[list[dict], str]] = []

    def send(self, message: str, silent: bool = False) -> bool:
        self.messages.append(message)
        return True

    def send_log(self, entries: list[dict], run_id: str) -> int:
        self.logs.append((entries, run_id))
        return len(entries)


def _write_record(root: Path, status: str, report: dict | None = None) -> None:
    record = {
        "run_id": "smoke-1",
        "kernel": "hariohmhemantbhatt/sfsc-pq-setformer-smoke-1",
        "stage": "E1-smoke-1024",
        "run_kind": "smoke",
        "status": status,
        "remote_status": "KernelWorkerStatus.COMPLETE" if report else "KernelWorkerStatus.RUNNING",
        "directory": str(root / "output"),
    }
    if report is not None:
        record["report"] = report
    path = root / "reports" / "pq_setformer_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_monitor_relays_each_status_once_and_verified_pq_and_logs(tmp_path) -> None:
    monitor = _load_monitor()
    notifier = FakeNotifier()
    _write_record(tmp_path, "running")

    monitor.watch_once(tmp_path, notifier)
    monitor.watch_once(tmp_path, notifier)

    assert len(notifier.messages) == 1
    assert "running" in notifier.messages[0]

    report = {
        "status": "complete",
        "inference": {
            "pq": 0.42,
            "sq": 0.61,
            "rq": 0.69,
            "tp": 12,
            "fp": 3,
            "fn": 5,
            "files": {
                "image.jpeg": {
                    "true_positives": 12,
                    "false_positives": 3,
                    "false_negatives": 5,
                    "matched_iou": 6.72,
                }
            },
        },
        "training": {"peak_reserved_gb": 10.5, "elapsed_seconds": 120.0},
    }
    _write_record(tmp_path, "execution_complete", report)
    event_path = tmp_path / "output" / "results" / "logs" / "events.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text(
        json.dumps({"event": "smoke_evaluation", "values": {"eval/global_pq": 0.42}})
    )

    monitor.watch_once(tmp_path, notifier)

    assert any("PQ: <b>0.4200</b>" in message for message in notifier.messages)
    assert any("TP 12" in message and "FP 3" in message for message in notifier.messages)
    assert any("image.jpeg: PQ 0.4200" in message for message in notifier.messages)
    assert notifier.logs == [
        ([{"event": "smoke_evaluation", "values": {"eval/global_pq": 0.42}}], "smoke-1")
    ]
