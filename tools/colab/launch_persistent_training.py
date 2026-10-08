#!/usr/bin/env python3
"""Launch the Drive-persistent ensemble pipeline from a Colab helper cell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY = Path("/content/SFSC2026")
CONTROL = Path(
    "/content/drive/MyDrive/SFSC2026/outputs/ultimate-ensemble-v1/control"
)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a detached Drive-persistent pipeline")
    parser.add_argument(
        "--phase",
        choices=("detectors", "prompts", "refiners", "semantic", "validate", "all"),
        default="all",
    )
    args = parser.parse_args()
    listing = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
    ).stdout
    active = [line for line in listing.splitlines() if "-m filament_ensemble.run" in line]
    if active:
        raise RuntimeError("training process already exists: " + " | ".join(active))
    if not (REPOSITORY / "filament_ensemble/run.py").is_file():
        raise FileNotFoundError(f"training source is missing from {REPOSITORY}")

    CONTROL.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    log_path = CONTROL / f"connected-{args.phase}-{stamp}.log"
    command = [sys.executable, "-u", "-m", "filament_ensemble.run", "--phase", args.phase]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    record = {
        "status": "running",
        "pid": process.pid,
        "started_utc": datetime.now(UTC).isoformat(),
        "log": str(log_path),
        "command": command,
    }
    write_json(CONTROL / "active-pipeline.json", record)
    time.sleep(4)
    return_code = process.poll()
    result = {**record, "alive_after_4s": return_code is None, "return_code": return_code}
    print(json.dumps(result))
    if return_code is not None:
        print(log_path.read_text(errors="replace")[-8000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
