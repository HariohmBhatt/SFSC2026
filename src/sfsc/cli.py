"""Command-line entry point for the notebook's iterative system."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sfsc import remote
from sfsc.contracts import ContractError
from sfsc.system import System, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="SFSC2026 three-fleet development system")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Existing SFSC repository")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("init", help="Validate and lock the fixed data/evaluation contract")
    commands.add_parser("status", help="Show durable fleet and execution state")
    cycle = commands.add_parser("cycle", help="Start the next approach-finding iteration")
    cycle.add_argument("--feedback", default="")
    request = commands.add_parser("request", help="Print a worker's self-contained JSON request")
    request.add_argument("job")
    accept = commands.add_parser("accept", help="Validate and import a worker's JSON result")
    accept.add_argument("job")
    accept.add_argument("result", type=Path)
    work = commands.add_parser("work", help="Execute a configured JSON stdin/stdout worker")
    work.add_argument("job")
    work.add_argument("--command", nargs=argparse.REMAINDER)
    fleet = commands.add_parser("fleet", help="Run research, review and code-writing fleets")
    fleet.add_argument("cycle")
    fleet.add_argument("--command", nargs=argparse.REMAINDER)
    recovery = commands.add_parser("recover-job", help="Recover a confirmed interrupted worker")
    recovery.add_argument("job")
    recovery.add_argument("--reason", required=True)
    freeze = commands.add_parser(
        "freeze", help="Close model selection before one-time internal test"
    )
    freeze.add_argument("cycle")
    reject = commands.add_parser(
        "reject", help="End an unsuccessful iteration, keeping its evidence"
    )
    reject.add_argument("cycle")
    reject.add_argument("--reason", required=True)
    prepare = commands.add_parser(
        "prepare-run", help="Build a self-contained private Kaggle kernel"
    )
    prepare.add_argument("cycle")
    prepare.add_argument("phase", choices=remote.PHASES)
    prepare.add_argument("--owner", required=True, help="Kaggle username")
    for action, help_text in (
        ("launch", "Submit prepared kernel to Kaggle P100"),
        ("remote-status", "Read Kaggle job status"),
        ("fetch", "Download artifacts and verify the result"),
    ):
        sub = commands.add_parser(action, help=help_text)
        sub.add_argument("run")
    collect = commands.add_parser(
        "collect", help="Verify an already downloaded report and artifacts"
    )
    collect.add_argument("run")
    collect.add_argument("report", type=Path)
    fail = commands.add_parser("fail-run", help="Record a confirmed remote failure with no report")
    fail.add_argument("run")
    fail.add_argument("--reason", required=True)
    args = parser.parse_args()
    system = System(args.root)
    try:
        match args.action:
            case "init":
                system.init()
                result = system.status()
            case "status":
                result = system.status()
            case "cycle":
                result = system.cycle(args.feedback)
            case "request":
                result = system.request(args.job)
            case "accept":
                result = system.accept(args.job, read_json(args.result))
            case "work":
                result = system.work(args.job, args.command)
            case "fleet":
                result = system.fleet(args.cycle, args.command)
            case "recover-job":
                result = system.recover_job(args.job, args.reason)
            case "freeze":
                result = system.freeze(args.cycle)
            case "reject":
                result = system.reject(args.cycle, args.reason)
            case "prepare-run":
                result = remote.prepare(system, args.cycle, args.phase, args.owner)
            case "launch":
                result = remote.launch(system, args.run)
            case "remote-status":
                result = remote.remote_status(system, args.run)
            case "fetch":
                result = remote.fetch(system, args.run)
            case "collect":
                result = remote.collect(system, args.run, args.report)
            case "fail-run":
                result = remote.fail(system, args.run, args.reason)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ContractError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, f"sfsc-system: {error}\n")


if __name__ == "__main__":
    main()
