"""Refresh the recorded remote status and optimizer evidence without downloading weights."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/precision_training.json"
    record = json.loads(path.read_text())
    status = subprocess.check_output(
        ["kaggle", "kernels", "status", record["kernel"]], text=True
    ).strip()
    logs = subprocess.check_output(["kaggle", "kernels", "logs", record["kernel"]], text=True)
    directory = root / record["directory"]
    if not logs.strip() and "RUNNING" in status.upper():
        # Kaggle may expose running logs only through the streaming endpoint.
        try:
            streamed = subprocess.run(
                ["kaggle", "kernels", "logs", record["kernel"], "--follow"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            logs = streamed.stdout
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or b""
            logs = captured.decode() if isinstance(captured, bytes) else captured
    if not logs.strip() and (directory / "live-execution.log").exists():
        logs = (directory / "live-execution.log").read_text()
    (directory / "latest-execution.log").write_text(logs)
    events = []
    for line in logs.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "event" in item:
            events.append(item)
    record.update(remote_status=status, checked_at=datetime.now(UTC).isoformat())
    if events:
        record["latest_event"] = events[-1]
    for event in events:
        if event["event"] == "training_step":
            record["last_optimizer_step"] = event
        elif event["event"] == "baseline_calibrated":
            record["baseline_calibration"] = event
    if "COMPLETE" in status.upper():
        record["status"] = "remote_completed_pending_report"
        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "output",
                record["kernel"],
                "--path",
                str(directory / "results"),
                "--force",
                "--file-pattern",
                r"experiment/(report|baseline-metrics|progress|checkpoint/.*)\.json$",
            ],
            text=True,
            capture_output=True,
        )
        (directory / "fetch.log").write_text(result.stdout + result.stderr)
        report_path = directory / "results/experiment/report.json"
        if result.returncode == 0 and report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("run_id") != record["run_id"]:
                raise RuntimeError("Downloaded report belongs to a different run")
            if report.get("spec_hash") != record["spec_hash"]:
                raise RuntimeError("Downloaded report source fingerprint differs")
            record["status"] = report["status"]
            record["validation_pq"] = report.get("metrics", {}).get("pq")
            record["peak_vram_gb"] = report.get("peak_vram_gb")
            record["report_path"] = str(report_path.relative_to(root))
    elif any(word in status.upper() for word in ["ERROR", "CANCEL", "FAIL"]):
        record["status"] = "remote_failed"
    elif "RUNNING" in status.upper():
        record["status"] = (
            "training_progress_verified" if record.get("last_optimizer_step") else "running"
        )
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
