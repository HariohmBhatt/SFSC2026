"""Forward Kaggle run status, reports, and system events to Telegram via a bot.

Kaggle kernels are batch jobs, so there is no live log stream from the remote
host. This module turns what the local machine already observes -- status
transitions from ``kaggle kernels status``, the final ``report.json`` fetched
through ``kaggle kernels output``, and durable ``.sfsc/state.json`` events --
into Telegram messages.

Credentials come from ``SFSC_TELEGRAM_BOT_TOKEN`` and ``SFSC_TELEGRAM_CHAT_ID``
in the environment, or from a ``.env`` file next to the repository root. Both
are required; without them every call to :func:`send` returns ``False`` and the
poller is inert.
"""

from __future__ import annotations

import html
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
MAX_TEXT = 4000
RETRY_SECONDS = (2, 5, 15)
ENV_TOKEN = "SFSC_TELEGRAM_BOT_TOKEN"
ENV_CHAT = "SFSC_TELEGRAM_CHAT_ID"


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` dotenv file, ignoring comments and blanks."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_config(root: Path | None = None) -> tuple[str, str]:
    """Return (bot_token, chat_id) from the environment or a repository .env file."""
    env = dict(os.environ)
    if root is not None:
        env = {**read_dotenv(root / ".env"), **env}
    token = env.get(ENV_TOKEN)
    chat_id = env.get(ENV_CHAT)
    if not token or not chat_id:
        return "", ""
    return token, chat_id


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def truncate(text: str, limit: int = MAX_TEXT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def send_message(
    token: str,
    chat_id: str,
    text: str,
    silent: bool = False,
    timeout: float = 30.0,
) -> bool:
    """POST one message to the Bot API with bounded retries. Never raises."""
    if not token or not chat_id or not text.strip():
        return False
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": truncate(text),
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
            "disable_notification": "true" if silent else "false",
        }
    ).encode()
    url = API_BASE.format(token=token)
    for attempt in RETRY_SECONDS:
        try:
            request = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read()
                return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(attempt)
    return False


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def clean_log_entries(entries: list[dict]) -> list[dict]:
    """Drop the trailing nbconvert/HTML tail Kaggle appends to every log."""
    for index, entry in enumerate(entries):
        if "[NbConvertApp]" in entry.get("data", ""):
            return list(entries[:index])
    return list(entries)


def clean_log_text(text: str) -> str:
    text = ANSI_PATTERN.sub("", text)
    return text.replace("\r", "").rstrip()


def _log_batches(entries: list[dict], run_id: str, chunk: int = 3800):
    """Return (message_text, entry_count) pairs for Telegram-sized batches."""
    header = f"<b>Log</b> {esc(run_id)}"
    batches: list[tuple[str, int]] = []
    current: list[str] = [header]
    length = len(header)
    count = 0
    for entry in entries:
        stream = esc(entry.get("stream_name") or "log")
        text = clean_log_text(str(entry.get("data", "")))
        if not text:
            continue
        seconds = entry.get("time")
        stamp = f"{float(seconds):.1f}s" if seconds is not None else "?"
        lines = text.splitlines()
        lines[0] = f"[{stamp} {stream}] {lines[0]}"
        text = "\n".join(lines)
        if length + len(text) + 1 > chunk:
            batches.append(("\n".join(current), count))
            current = [text]
            length = len(text)
            count = 1
        else:
            current.append(text)
            length += len(text) + 1
            count += 1
    if len(current) > 1:
        batches.append(("\n".join(current), count))
    return batches


def log_batches(entries: list[dict], run_id: str, chunk: int = 3800) -> list[str]:
    """Group timestamped log entries into Telegram-sized message batches."""
    return [text for text, _ in _log_batches(entries, run_id, chunk)]


class Notifier:
    """Format and deliver run status, reports, and events to one chat."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str, silent: bool = False) -> bool:
        if not self.enabled:
            return False
        return send_message(self.token, self.chat_id, text, silent=silent)

    def send_log(self, entries: list[dict], run_id: str, limit: int = 100) -> int:
        """Send log batches, returning how many entries were delivered."""
        if not self.enabled or not entries:
            return 0
        batches = _log_batches(entries, run_id)
        delivered = 0
        for text, count in batches[:limit]:
            if not self.send(text):
                break
            delivered += count
        if len(batches) > limit:
            self.send(
                f"<b>Log</b> {esc(run_id)} — truncated after {limit} messages; "
                f"{len(batches) - limit} more batches available."
            )
        return delivered

    def status_change(self, run: dict, old: str, new: str) -> str:
        phase = esc(run.get("phase") or "run")
        kernel = esc(run.get("kernel") or "")
        run_id = esc(run.get("run_id") or "?")
        return (
            f"<b>Kaggle</b> {run_id} · {phase}\n"
            f"{kernel}\n"
            f"status: <b>{esc(old)}</b> \u2192 <b>{esc(new)}</b>"
        )

    def report_summary(self, run: dict, report: dict) -> str:
        run_id = esc(run.get("run_id") or report.get("run_id") or "?")
        phase = esc(run.get("phase") or report.get("phase") or "run")
        status = report.get("status")
        head = f"<b>Kaggle</b> {run_id} · {phase} · <b>{esc(status)}</b>"
        if report.get("elapsed_seconds"):
            head += f"\nelapsed: {self._duration(report['elapsed_seconds'])}"
        if status == "completed":
            return self._completed(head, report)
        return self._failed(head, report)

    def _completed(self, head: str, report: dict) -> str:
        lines = [head, "gpu: " + esc(report.get("gpu") or "?"),
                 "python: " + esc(report.get("python") or "?")]
        if report.get("peak_vram_gb") is not None:
            lines.append(f"peak VRAM: {report['peak_vram_gb']:.2f} GB")
        metrics = report.get("metrics")
        if metrics:
            pq = metrics.get("pq")
            split = esc(metrics.get("split") or "?")
            lines.append(
                f"PQ ({split}): <b>{pq:.4f}</b> (SQ {metrics.get('sq'):.3f}, "
                f"RQ {metrics.get('rq'):.3f})"
            )
            lines.append(
                f"TP {metrics['true_positives']} · FP {metrics['false_positives']} · "
                f"FN {metrics['false_negatives']}"
            )
        if report.get("checkpoint") is not None:
            lines.append("checkpoint: " + esc(report["checkpoint"]))
            lines.append("hash: <code>" + esc(report.get("checkpoint_hash") or "") + "</code>")
        if report.get("submission_hash"):
            lines.append("submission hash: <code>" + esc(report["submission_hash"]) + "</code>")
        return "\n".join(lines)

    def _failed(self, head: str, report: dict) -> str:
        lines = [head]
        if report.get("error"):
            lines.append("error: " + esc(report["error"]))
        traceback_lines = (report.get("traceback") or "").splitlines()
        if traceback_lines:
            shown = traceback_lines[:12]
            lines.append("<pre>" + esc("\n".join(shown)) + "</pre>")
        return "\n".join(lines)

    def event_line(self, event: dict) -> str:
        at = str(event.get("at", ""))
        at = at.replace("T", " ").replace("+00:00", "Z").replace(".000000Z", "Z")
        if "." in at and "Z" in at:
            at = at[: at.index(".")] + "Z"
        at = esc(at)
        action = esc(event.get("action") or "?")
        details = " · ".join(f"{esc(k)}={esc(v)}" for k, v in event.items() if k != "at")
        return f"<b>sfsc</b> {at} · {action}{' · ' + details if details else ''}"

    def quota_summary(self, accounts: dict[str, dict]) -> str:
        lines = ["<b>Kaggle quota</b>"]
        for name, info in accounts.items():
            lines.append(f"<b>{esc(name)}</b> ({esc(info['username'])})")
            if not info["rows"]:
                lines.append("no quota data")
                continue
            for row in info["rows"]:
                refresh = esc(row.get("refreshAt", ""))
                lines.append(
                    f"{esc(row.get('resource', '?'))} "
                    f"{esc(row.get('used', '?'))} used · "
                    f"{esc(row.get('remaining', '?'))} left "
                    f"(of {esc(row.get('total', '?'))}) · refresh {refresh}"
                )
        return "\n".join(lines)

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = float(seconds)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"