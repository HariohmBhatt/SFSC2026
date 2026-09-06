"""Monitor the authorized run, then verify and upload exactly one submission."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "reports/full_training.json"
    initial = json.loads(path.read_text())
    run_id = initial["run_id"]
    if not initial.get("submission_authorized"):
        raise ValueError("Explicit submission authorization required")
    while True:
        record = json.loads(path.read_text())
        if record["run_id"] != run_id:
            raise ValueError("Active full-training record changed; refusing to submit another run")
        status = subprocess.check_output(
            ["kaggle", "kernels", "status", record["kernel"]], text=True
        ).strip()
        latest = None
        logfile = root / record["directory"] / "live-execution.log"
        if logfile.exists():
            for line in logfile.read_text().splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and "event" in item:
                    latest = item
        record.update(remote_status=status, checked_at=datetime.now(UTC).isoformat())
        if latest:
            record["latest_event"] = latest
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"status": status, "latest": latest}), flush=True)
        if "COMPLETE" in status.upper():
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("finish.py")), "--submit"],
                cwd=root,
                check=True,
            )
            return
        if any(word in status.upper() for word in ["ERROR", "FAIL", "CANCEL"]):
            raise RuntimeError("Remote full training failed; no submission attempted")
        time.sleep(45)


if __name__ == "__main__":
    main()
