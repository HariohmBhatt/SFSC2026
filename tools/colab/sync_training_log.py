#!/usr/bin/env python3
"""Mirror a Colab/Drive training log into a local file without touching training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

DEFAULT_COLAB = Path("/home/hariohm/.local/bin/colab")
DEFAULT_LOCAL = Path("outputs/ultimate-ensemble-v1/training-live.log")
DEFAULT_POINTER = (
    "/content/drive/MyDrive/SFSC2026/outputs/ultimate-ensemble-v1/control/"
    "active-pipeline.json"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="sfsc-ultimate-v1")
    parser.add_argument(
        "--remote",
        help="fixed remote log; by default the active-pipeline pointer is followed",
    )
    parser.add_argument("--pointer", default=DEFAULT_POINTER)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--colab", type=Path, default=DEFAULT_COLAB)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--max-failures", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def same_file(left: Path, right: Path) -> bool:
    if not left.is_file() or left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(1024 * 1024)
            right_chunk = right_file.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def download_remote(
    args: argparse.Namespace, remote: str, destination: Path
) -> tuple[int, bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        result = subprocess.run(
            [
                str(args.colab),
                "--auth",
                "oauth2",
                "download",
                "-s",
                args.session,
                remote,
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(message or f"colab download exited {result.returncode}")
        size = temporary.stat().st_size
        if same_file(destination, temporary):
            return size, False
        os.replace(temporary, destination)
        return size, True
    finally:
        temporary.unlink(missing_ok=True)


def active_remote(args: argparse.Namespace, directory: Path) -> str:
    if args.remote:
        return args.remote
    pointer = directory / ".active-pipeline.json.download"
    download_remote(args, args.pointer, pointer)
    try:
        remote = json.loads(pointer.read_text()).get("log")
    finally:
        pointer.unlink(missing_ok=True)
    if not isinstance(remote, str) or not remote:
        raise RuntimeError(f"active pipeline pointer has no log path: {args.pointer}")
    return remote


def mirror_path(destination: Path, remote: str) -> Path:
    digest = hashlib.sha256(remote.encode()).hexdigest()[:12]
    return destination.parent / ".training-live.logs" / f"{digest}-{Path(remote).name}"


def rebuild(destination: Path, sources: list[str]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".aggregate", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            for index, remote in enumerate(sources):
                if index:
                    output.write(f"\nSFSC_LOG_SOURCE {json.dumps({'remote': remote})}\n".encode())
                with mirror_path(destination, remote).open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def download(args: argparse.Namespace, destination: Path) -> tuple[int, str, bool]:
    remote = active_remote(args, destination.parent)
    mirror = mirror_path(destination, remote)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    size, changed = download_remote(args, remote, mirror)
    state_path = destination.parent / ".training-live.sources.json"
    sources = (
        json.loads(state_path.read_text()).get("sources", [])
        if state_path.is_file()
        else []
    )
    added = remote not in sources
    if added:
        sources.append(remote)
        state_path.write_text(json.dumps({"sources": sources}, indent=2) + "\n")
    if changed or added or not destination.is_file():
        rebuild(destination, sources)
    return size, remote, changed


def main() -> int:
    args = arguments()
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    if args.max_failures <= 0:
        raise SystemExit("--max-failures must be greater than zero")
    destination = args.local.expanduser().resolve()
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    failures = 0
    while not stopping:
        try:
            size, remote, changed = download(args, destination)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            failures += 1
            print(f"sync failed ({failures}/{args.max_failures}): {error}", flush=True)
            if failures >= args.max_failures:
                return 1
        else:
            failures = 0
            print(
                f"{'synced' if changed else 'unchanged'} {size} bytes from {remote} "
                f"at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
                flush=True,
            )
        if args.once:
            return 0
        deadline = time.monotonic() + args.interval
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(0.5, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
