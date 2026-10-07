"""Keep a managed Colab kernel busy while the detached training process runs."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

POINTER = Path(
    "/content/drive/MyDrive/SFSC2026/outputs/ultimate-ensemble-v1/control/"
    "active-pipeline.json"
)
EXPECTED_COMMAND = "-m filament_ensemble.run"


def process_command(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def main() -> None:
    record = json.loads(POINTER.read_text())
    pid = int(record["pid"])
    command = process_command(pid)
    if command is None or EXPECTED_COMMAND not in command:
        raise RuntimeError(f"active training PID {pid} is not running")

    print(json.dumps({"event": "keepalive_start", "pid": pid}), flush=True)
    while True:
        command = process_command(pid)
        if command is None or EXPECTED_COMMAND not in command:
            print(json.dumps({"event": "keepalive_complete", "pid": pid}), flush=True)
            return
        print(
            json.dumps(
                {
                    "event": "keepalive_heartbeat",
                    "pid": pid,
                    "time": datetime.now(UTC).isoformat(),
                }
            ),
            flush=True,
        )
        time.sleep(30)


if __name__ == "__main__":
    main()
