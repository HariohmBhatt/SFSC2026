"""Poll active Kaggle kernels and forward status, logs, reports, and events.

Reads durable orchestration state from ``.sfsc/state.json`` and forwards four
kinds of messages (matching ``sfsc.telegram.Notifier``):

* every new ``kaggle kernels status`` transition for a running kernel,
* the kernel's full clean stdout/stderr log once it completes,
* the final ``report.json`` summary once a kernel completes or fails,
* every new system event appended to the state store.

Both Kaggle accounts are handled: each kernel is queried under the account that
owns it (``sfsc.kaggle_cli``), so private kernels on ``hariohmhemantbhatt`` are
read with the Kaggle2 credentials. ``--quota`` fetches weekly GPU/TPU limits
for every configured account and reports them on demand.

Kaggle only materializes a kernel's session log after it finishes (verified:
the output endpoint returns nothing while a kernel is running). The poller
therefore streams status live and delivers the complete clean log in batched
Telegram messages once the kernel completes, deduplicating by a persisted
per-run entry offset so nothing is sent twice.

Only runs in ``submitted`` state are polled; prepared/launching runs surface
their launch through the ``run_submitted`` event. Progress is persisted in
``.sfsc/telegram_state.json`` so restarts never resend already-reported
statuses, logs, reports, or events.

The notifier is inert unless ``SFSC_TELEGRAM_BOT_TOKEN`` and
``SFSC_TELEGRAM_CHAT_ID`` are configured.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from sfsc.kaggle_cli import account_for_owner, kernel_log, quota_all, run_cli
from sfsc.system import read_json, write_json
from sfsc.telegram import Notifier, clean_log_entries, load_config

ACTIVE_STATUSES = {"prepared", "launching", "submitted"}
POLLABLE_STATUSES = {"launching", "submitted"}
TERMINAL_KAGGE_WORDS = ("complete", "error", "failed", "cancel")
POLL_TIMEOUT = 60
DEFAULT_INTERVAL = 45


def load_watch_state(root: Path) -> dict:
    path = root / ".sfsc" / "telegram_state.json"
    if path.exists():
        try:
            return read_json(path)
        except (ValueError, OSError):
            pass
    return {"events_seen": 0, "statuses": {}, "resolved": {}}


def save_watch_state(root: Path, state: dict) -> None:
    write_json(root / ".sfsc" / "telegram_state.json", state)


def kaggle_status(kernel: str) -> str | None:
    owner = kernel.split("/", 1)[0] if "/" in kernel else None
    account = account_for_owner(owner) if owner else account_for_owner("")
    try:
        return run_cli(["kernels", "status", kernel], account, timeout=POLL_TIMEOUT).strip()
    except (RuntimeError, subprocess.SubprocessError, OSError):
        return None


def terminal_status(status: str) -> bool:
    lowered = status.lower()
    return any(word in lowered for word in TERMINAL_KAGGE_WORDS)


def find_report(folder: Path) -> dict | None:
    reports = sorted(folder.rglob("report.json"))
    for path in reports:
        try:
            report = read_json(path)
        except (ValueError, OSError):
            continue
        if isinstance(report, dict) and report.get("status"):
            return report
    return None


def fetch_report(notifier: Notifier, run: dict, root: Path) -> dict | None:
    """Download kernel output to a scratch folder and return its report.json."""
    kernel = run.get("kernel")
    if not kernel:
        return None
    owner = kernel.split("/", 1)[0]
    account = account_for_owner(owner)
    folder = root / ".sfsc" / "telegram-downloads" / run.get("run_id", "unknown")
    folder.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            run_cli(["kernels", "output", kernel, "--path", str(folder), "--force"], account)
        except (RuntimeError, subprocess.SubprocessError, OSError):
            time.sleep(10)
            continue
        report = find_report(folder)
        if report is not None:
            return report
        time.sleep(10)
    return None


def send_kernel_log(notifier: Notifier, run: dict, root: Path, watch: dict) -> None:
    """Fetch a completed kernel's log and forward only the not-yet-sent entries."""
    if not notifier.enabled:
        return
    kernel = run.get("kernel")
    if not kernel or "/" not in kernel:
        return
    run_id = run["run_id"]
    owner, slug = kernel.split("/", 1)
    account = account_for_owner(owner)
    try:
        entries = clean_log_entries(kernel_log(account, owner, slug))
    except (RuntimeError, ValueError, OSError) as error:
        print(f"  log fetch failed for {kernel}: {error}", file=sys.stderr)
        return
    sent = int(watch.get("log_sent", {}).get(run_id, 0))
    pending = entries[sent:]
    if not pending:
        return
    delivered = notifier.send_log(pending, run_id)
    watch.setdefault("log_sent", {})[run_id] = sent + delivered
    save_watch_state(root, watch)


def poll_run(notifier: Notifier, run: dict, root: Path, watch: dict) -> None:
    run_id = run["run_id"]
    status = kaggle_status(run["kernel"])
    if status is None:
        return
    previous = watch["statuses"].get(run_id)
    if previous is None or previous != status:
        notifier.send(notifier.status_change(run, previous or "unknown", status))
        watch["statuses"][run_id] = status
    if terminal_status(status) and not watch["resolved"].get(run_id):
        if "complete" in status.lower():
            report = fetch_report(notifier, run, root)
            if report is not None:
                notifier.send(notifier.report_summary(run, report))
            else:
                notifier.send(
                    f"<b>Kaggle</b> {run.get('run_id', '?')} completed but "
                    "no report.json was found in its output."
                )
            send_kernel_log(notifier, run, root, watch)
        else:
            notifier.send(
                notifier.report_summary(
                    run, {"status": "failed", "error": f"Kaggle status: {status}"}
                )
            )
        watch["resolved"][run_id] = True
        save_watch_state(root, watch)


def resolve_terminal_run(notifier: Notifier, run: dict, root: Path, watch: dict) -> None:
    run_id = run["run_id"]
    if watch["resolved"].get(run_id):
        return
    report = run.get("report") if isinstance(run.get("report"), dict) else None
    if report is None:
        report = fetch_report(notifier, run, root)
    if report is not None:
        notifier.send(notifier.report_summary(run, report))
    if report is not None and report.get("status") == "completed":
        send_kernel_log(notifier, run, root, watch)
    watch["resolved"][run_id] = True
    save_watch_state(root, watch)


def replay_events(notifier: Notifier, state: dict, watch: dict) -> None:
    events = state.get("events", [])
    start = int(watch.get("events_seen", 0))
    for event in events[start:]:
        notifier.send(notifier.event_line(event))
    watch["events_seen"] = len(events)


def watch_once(root: Path, notifier: Notifier, run_id: str | None) -> None:
    state_path = root / ".sfsc" / "state.json"
    if not state_path.exists():
        print("no .sfsc/state.json; run sfsc-system init first", file=sys.stderr)
        return
    state = read_json(state_path)
    watch = load_watch_state(root)
    replay_events(notifier, state, watch)
    runs = state.get("runs", {})
    for current, run in runs.items():
        if run_id is not None and current != run_id:
            continue
        if run.get("status") in POLLABLE_STATUSES:
            poll_run(notifier, run, root, watch)
        elif run.get("status") not in ACTIVE_STATUSES:
            resolve_terminal_run(notifier, run, root, watch)
    save_watch_state(root, watch)


def watch_loop(
    root: Path, notifier: Notifier, run_id: str | None, interval: int
) -> None:
    while True:
        try:
            watch_once(root, notifier, run_id)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            print(f"watch pass failed: {error}", file=sys.stderr)
        time.sleep(interval)


def quota_once(root: Path, notifier: Notifier) -> None:
    """Fetch weekly quota for every configured account and report it."""
    data = quota_all()
    for account_name, info in data.items():
        print(f"{account_name} ({info['username']}):")
        for row in info["rows"]:
            print(f"  {row.get('resource')}: {row.get('used')} used, "
                  f"{row.get('remaining')} left (refresh {row.get('refreshAt')})")
    notifier.send(notifier.quota_summary(data))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="SFSC repository")
    parser.add_argument("--run", help="only watch this run_id; default: all runs")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="poll seconds")
    parser.add_argument("--once", action="store_true", help="single pass, then exit")
    parser.add_argument(
        "--quota", action="store_true", help="fetch weekly quota for all accounts, then exit"
    )
    parser.add_argument(
        "--ping", action="store_true", help="send a test message to confirm configuration"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    token, chat_id = load_config(root)
    notifier = Notifier(token, chat_id)
    if not notifier.enabled:
        print(
            "Telegram not configured (set SFSC_TELEGRAM_BOT_TOKEN and "
            "SFSC_TELEGRAM_CHAT_ID); notifier is inert.",
            file=sys.stderr,
        )
    if args.ping:
        sent = notifier.send("<b>sfsc-telegram</b> poller is configured and reachable.")
        print("ping sent" if sent else "ping not sent (notifier disabled)")
        return
    if args.quota:
        quota_once(root, notifier)
        return
    if args.once:
        watch_once(root, notifier, args.run)
    else:
        watch_loop(root, notifier, args.run, args.interval)


if __name__ == "__main__":
    main()