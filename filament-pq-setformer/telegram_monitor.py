"""Relay one PQ-SetFormer controller record to Telegram.

Kaggle exposes controller state while a session runs. Once the controller has
downloaded terminal artifacts, this relay forwards the complete structured
``events.jsonl`` stream and the exact global/per-image PQ report.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_CHARS = 3_800


def _new_state() -> dict[str, dict[str, Any]]:
    return {"statuses": {}, "reports": {}, "log_entries": {}}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return _new_state()
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return _new_state()
    if not isinstance(value, dict):
        return _new_state()
    return {
        key: value.get(key, {}) if isinstance(value.get(key, {}), dict) else {}
        for key in _new_state()
    }


def _telegram_config(root: Path) -> tuple[str, str]:
    values = dict(os.environ)
    dotenv = root / ".env"
    if dotenv.is_file():
        for raw in dotenv.read_text(errors="replace").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values.setdefault(key.strip(), value.strip().strip("'\""))
    return values.get("SFSC_TELEGRAM_BOT_TOKEN", ""), values.get("SFSC_TELEGRAM_CHAT_ID", "")


class Notifier:
    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str) -> bool:
        if not self.enabled or not message:
            return False
        payload = urllib.parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": message[:MAX_MESSAGE_CHARS],
                "parse_mode": "HTML",
                "disable_web_page_preview": "true",
            }
        ).encode()
        request = urllib.request.Request(
            API_BASE.format(token=self.token), data=payload, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read()
        except OSError:
            return False
        return True

    def send_log(self, entries: list[dict], run_id: str) -> int:
        lines = [f"<b>PQ log</b> {html.escape(run_id)}"]
        delivered = 0
        for entry in entries:
            line = html.escape(
                f"{entry.get('event', 'event')} step={entry.get('step')} {entry.get('values', {})}"
            )
            if sum(len(item) + 1 for item in lines) + len(line) > MAX_MESSAGE_CHARS:
                if not self.send("\n".join(lines)):
                    return delivered
                lines = [f"<b>PQ log</b> {html.escape(run_id)}", line]
            else:
                lines.append(line)
            delivered += 1
        if len(lines) > 1 and not self.send("\n".join(lines)):
            return 0
        return delivered


def _metric(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("PQ report contains an invalid metric")
    return float(value)


def _status_message(record: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"<b>PQ-SetFormer</b> {html.escape(str(record.get('run_id', '?')))} · "
            f"{html.escape(str(record.get('stage', '?')))}",
            f"status: <b>{html.escape(str(record.get('status', '?')))}</b>",
            f"remote: {html.escape(str(record.get('remote_status', 'not submitted')))}",
        )
    )


def _report_messages(record: dict[str, Any], report: dict[str, Any]) -> list[str]:
    if report.get("status") != "complete" or not isinstance(report.get("inference"), dict):
        raise ValueError("PQ report is not a verified complete inference report")
    metrics = report["inference"]
    pq, sq, rq = (_metric(metrics[name]) for name in ("pq", "sq", "rq"))
    tp, fp, fn = (int(_metric(metrics[name])) for name in ("tp", "fp", "fn"))
    if min(tp, fp, fn) < 0:
        raise ValueError("PQ report contains negative counts")
    training = report.get("training", {})
    head = "\n".join(
        (
            f"<b>PQ-SetFormer</b> {html.escape(str(record.get('run_id', '?')))} · "
            "<b>verified complete</b>",
            f"PQ: <b>{pq:.4f}</b> (SQ {sq:.4f}, RQ {rq:.4f})",
            f"TP {tp} · FP {fp} · FN {fn}",
            f"peak VRAM: {_metric(training.get('peak_reserved_gb', 0.0)):.2f} GB · "
            f"elapsed: {_metric(training.get('elapsed_seconds', 0.0)):.0f}s",
        )
    )
    files = metrics.get("files")
    if not isinstance(files, dict):
        raise ValueError("PQ report has no per-image evidence")
    lines = ["<b>Per-image PQ</b>"]
    for filename, stats in sorted(files.items()):
        if not isinstance(stats, dict):
            raise ValueError("PQ per-image evidence is invalid")
        file_tp = _metric(stats.get("true_positives", stats.get("tp")))
        file_fp = _metric(stats.get("false_positives", stats.get("fp")))
        file_fn = _metric(stats.get("false_negatives", stats.get("fn")))
        file_iou = _metric(stats.get("matched_iou", stats.get("sum_iou")))
        denominator = file_tp + 0.5 * (file_fp + file_fn)
        file_pq = file_iou / denominator if denominator else 0.0
        lines.append(
            f"{html.escape(str(filename))}: PQ {file_pq:.4f} · TP {int(file_tp)} "
            f"FP {int(file_fp)} FN {int(file_fn)}"
        )
    messages = [head]
    current = [lines[0]]
    for line in lines[1:]:
        if sum(len(item) + 1 for item in current) + len(line) > MAX_MESSAGE_CHARS:
            messages.append("\n".join(current))
            current = [lines[0], line]
        else:
            current.append(line)
    if len(current) > 1:
        messages.append("\n".join(current))
    return messages


def _controller_events(record: dict[str, Any]) -> list[dict]:
    directory = Path(str(record.get("directory", ""))) / "results"
    paths = sorted(directory.rglob("events.jsonl")) if directory.is_dir() else []
    if len(paths) != 1:
        return []
    try:
        return [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def watch_once(root: Path, notifier: Notifier) -> None:
    """Forward new controller state, terminal structured logs, and verified PQ."""
    root = Path(root)
    record_path = root / "reports" / "pq_setformer_run.json"
    if not record_path.is_file() or not notifier.enabled:
        return
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(record, dict) or not record.get("run_id"):
        return
    state_path = root / ".sfsc" / "pq-setformer-telegram-state.json"
    state = _load_state(state_path)
    run_id = str(record["run_id"])
    status = [str(record.get("status", "")), str(record.get("remote_status", ""))]
    if state["statuses"].get(run_id) != status and notifier.send(_status_message(record)):
        state["statuses"][run_id] = status

    if record.get("status") in {"execution_complete", "failed", "cancelled"}:
        events = _controller_events(record)
        sent = int(state["log_entries"].get(run_id, 0))
        if events[sent:]:
            delivered = notifier.send_log(events[sent:], run_id)
            state["log_entries"][run_id] = sent + delivered
    report = record.get("report")
    if record.get("status") == "execution_complete" and isinstance(report, dict):
        report_fingerprint = json.dumps(report, sort_keys=True, separators=(",", ":"))
        if state["reports"].get(run_id) != report_fingerprint:
            try:
                messages = _report_messages(record, report)
            except ValueError as error:
                notifier.send(f"<b>PQ-SetFormer</b> report rejected: {html.escape(str(error))}")
            else:
                if all(notifier.send(message) for message in messages):
                    state["reports"][run_id] = report_fingerprint
    _write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=int, default=45)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    token, chat_id = _telegram_config(arguments.root)
    notifier = Notifier(token, chat_id)
    if not notifier.enabled:
        raise RuntimeError("Telegram is not configured")
    while True:
        watch_once(arguments.root, notifier)
        if arguments.once:
            return
        time.sleep(arguments.interval)


if __name__ == "__main__":
    main()
