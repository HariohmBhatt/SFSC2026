"""Unit tests for the Telegram notifier and Kaggle poller."""

from __future__ import annotations

import json
from pathlib import Path

from sfsc import telegram, telegram_watch

RUN = {
    "run_id": "cycle-001-full-train-e0b5f3ed",
    "phase": "full_train",
    "kernel": "hariohmbhatt/sfsc-cycle-001-full-train-e0b5f3ed",
}

COMPLETED_REPORT = {
    "run_id": "cycle-001-full-train-e0b5f3ed",
    "phase": "full_train",
    "status": "completed",
    "gpu": "Tesla P100",
    "python": "3.11.9",
    "elapsed_seconds": 7234,
    "peak_vram_gb": 11.2,
    "checkpoint": "checkpoint",
    "checkpoint_hash": "abc123",
    "metrics": {
        "split": "validation",
        "pq": 0.4123,
        "sq": 0.664,
        "rq": 0.616,
        "true_positives": 774,
        "false_positives": 503,
        "false_negatives": 463,
    },
}

FAILED_REPORT = {
    "run_id": "cycle-001-evaluate-8f1e481a",
    "phase": "evaluate",
    "status": "failed",
    "elapsed_seconds": 120,
    "error": "TensorFlow GPU probe failed",
    "traceback": (
        "Traceback (most recent call last):\n  File \"run.py\", line 1\nRuntimeError: boom"
    ),
}


def make_notifier(monkeypatch, token="bot", chat="123") -> telegram.Notifier:
    notifier = telegram.Notifier(token, chat)
    monkeypatch.setattr(telegram, "send_message", lambda *a, **k: True)
    return notifier


def test_truncate() -> None:
    assert telegram.truncate("short") == "short"
    assert len(telegram.truncate("x" * 5000)) <= telegram.MAX_TEXT
    assert telegram.truncate("x" * 5000).endswith("...")


def test_esc() -> None:
    assert telegram.esc("<b>&") == "&lt;b&gt;&amp;"


def test_status_change_format() -> None:
    notifier = telegram.Notifier("", "")
    text = notifier.status_change(RUN, "submitted", "running")
    assert RUN["run_id"] in text
    assert "submitted" in text and "running" in text
    assert "<b>running</b>" in text


def test_completed_report_summary(monkeypatch) -> None:
    notifier = telegram.Notifier("", "")
    text = notifier.report_summary(RUN, COMPLETED_REPORT)
    assert "completed" in text
    assert "Tesla P100" in text
    assert "0.4123" in text
    assert "2h 00m 34s" in text
    assert "774" in text
    assert "abc123" in text


def test_failed_report_summary(monkeypatch) -> None:
    notifier = telegram.Notifier("", "")
    text = notifier.report_summary(RUN, FAILED_REPORT)
    assert "failed" in text
    assert "GPU probe failed" in text
    assert "RuntimeError: boom" in text


def test_duration() -> None:
    assert telegram.Notifier._duration(59) == "59s"
    assert telegram.Notifier._duration(61) == "1m 01s"
    assert telegram.Notifier._duration(7234) == "2h 00m 34s"


def test_quota_summary() -> None:
    notifier = telegram.Notifier("", "")
    text = notifier.quota_summary(
        {
            "kaggle": {
                "username": "hariohmbhatt",
                "rows": [
                    {"resource": "GPU", "used": "0.00h", "remaining": "30.00h",
                     "total": "30.00h", "refreshAt": "2026-09-19T00:00:00"},
                ],
            },
            "kaggle2": {"username": "hariohmhemantbhatt", "rows": []},
        }
    )
    assert "<b>kaggle</b> (hariohmbhatt)" in text
    assert "<b>kaggle2</b> (hariohmhemantbhatt)" in text
    assert "GPU 0.00h used · 30.00h left" in text
    assert "no quota data" in text


def test_clean_log_entries_drops_nbconvert_tail() -> None:
    entries = [
        {"stream_name": "stdout", "time": 1.0, "data": "epoch 1"},
        {"stream_name": "stderr", "time": 2.0, "data": "[NbConvertApp] Converting"},
        {"stream_name": "stderr", "time": 3.0, "data": "Writing 100 bytes"},
    ]
    cleaned = telegram.clean_log_entries(entries)
    assert len(cleaned) == 1
    assert cleaned[0]["data"] == "epoch 1"


def test_clean_log_text_strips_ansi_and_carriage_returns() -> None:
    text = telegram.clean_log_text("\x1b[1mTICK\x1b[0m\r\nnext")
    assert text == "TICK\nnext"


def test_log_batches_grouping() -> None:
    entries = [
        {"stream_name": "stdout", "time": 1.0, "data": "line a"},
        {"stream_name": "stdout", "time": 2.0, "data": "line b"},
    ]
    batches = telegram.log_batches(entries, "run-1")
    assert len(batches) == 1
    assert "run-1" in batches[0]
    assert "[1.0s stdout] line a" in batches[0]
    assert "[2.0s stdout] line b" in batches[0]


def test_log_batches_splits_on_size() -> None:
    entries = [
        {"stream_name": "stdout", "time": float(i), "data": f"x{i}"} for i in range(200)
    ]
    batches = telegram.log_batches(entries, "run-1", chunk=500)
    assert len(batches) > 1
    assert all(len(b) <= 500 for b in batches)


def test_send_log_delivers_batches(monkeypatch) -> None:
    notifier = telegram.Notifier("bot", "chat")
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    entries = [
        {"stream_name": "stdout", "time": float(i), "data": "t" * 1500}
        for i in range(6)
    ]
    delivered = notifier.send_log(entries, "run-9", limit=2)
    assert delivered == 4  # first two batches cover four entries
    assert len(sent) == 3  # 2 batches + truncation note
    assert any("truncated" in text for text in sent)


def test_send_log_disabled() -> None:
    notifier = telegram.Notifier("", "")
    assert notifier.send_log([{"data": "x"}], "r") == 0


def test_event_line() -> None:
    notifier = telegram.Notifier("", "")
    event = {
        "at": "2026-09-05T21:47:32.279401+00:00",
        "action": "run_collected",
        "status": "completed",
    }
    text = notifier.event_line(event)
    assert "run_collected" in text
    assert "status=completed" in text
    assert "2026-09-05 21:47:32Z" in text


def test_read_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("A=1\nB = 'two'\n# comment\n\nC=\"three\"\n")
    values = telegram.read_dotenv(tmp_path / ".env")
    assert values == {"A": "1", "B": "two", "C": "three"}


def test_load_config_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("SFSC_TELEGRAM_BOT_TOKEN=abc\nSFSC_TELEGRAM_CHAT_ID=456\n")
    monkeypatch.delenv("SFSC_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SFSC_TELEGRAM_CHAT_ID", raising=False)
    assert telegram.load_config(tmp_path) == ("abc", "456")


def test_load_config_env_overrides_dotenv(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("SFSC_TELEGRAM_BOT_TOKEN=file\nSFSC_TELEGRAM_CHAT_ID=fileid\n")
    monkeypatch.setenv("SFSC_TELEGRAM_BOT_TOKEN", "envtoken")
    assert telegram.load_config(tmp_path) == ("envtoken", "fileid")


def test_send_message_disabled_returns_false() -> None:
    assert telegram.send_message("", "", "hi") is False


def test_send_message_success(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout: float = 30.0):
        assert request.full_url.startswith("https://api.telegram.org/botT/")
        assert b"chat_id=C" in request.data
        return Response()

    monkeypatch.setattr(telegram.urllib.request, "urlopen", fake_urlopen)
    assert telegram.send_message("T", "C", "hello") is True


def test_send_message_retries_then_fails(monkeypatch) -> None:
    def fail_urlopen(request, timeout: float = 30.0):
        raise telegram.urllib.error.URLError("down")

    monkeypatch.setattr(telegram.urllib.request, "urlopen", fail_urlopen)
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)
    assert telegram.send_message("T", "C", "hello") is False


def test_terminal_status() -> None:
    assert telegram_watch.terminal_status("running") is False
    assert telegram_watch.terminal_status("queued") is False
    assert telegram_watch.terminal_status("complete") is True
    assert telegram_watch.terminal_status("ERROR") is True
    assert telegram_watch.terminal_status("Canceled") is True


def test_find_report(tmp_path: Path) -> None:
    report = {"run_id": "r", "status": "completed"}
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "report.json").write_text(json.dumps(report))
    assert telegram_watch.find_report(tmp_path) == report
    (tmp_path / "sub" / "report.json").write_text("not json")
    assert telegram_watch.find_report(tmp_path) is None


def test_poll_run_sends_status_change(monkeypatch, tmp_path: Path) -> None:
    notifier = make_notifier(monkeypatch)
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    monkeypatch.setattr(telegram_watch, "kaggle_status", lambda kernel: "running")
    watch = {"events_seen": 0, "statuses": {}, "resolved": {}}
    telegram_watch.poll_run(notifier, RUN, tmp_path, watch)
    assert watch["statuses"][RUN["run_id"]] == "running"
    assert len(sent) == 1


def test_poll_run_skips_unchanged_status(monkeypatch, tmp_path: Path) -> None:
    notifier = make_notifier(monkeypatch)
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    monkeypatch.setattr(telegram_watch, "kaggle_status", lambda kernel: "running")
    watch = {"events_seen": 0, "statuses": {RUN["run_id"]: "running"}, "resolved": {}}
    telegram_watch.poll_run(notifier, RUN, tmp_path, watch)
    assert sent == []


def test_poll_run_sends_report_on_complete(monkeypatch, tmp_path: Path) -> None:
    notifier = make_notifier(monkeypatch)
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    monkeypatch.setattr(telegram_watch, "kaggle_status", lambda kernel: "complete")
    monkeypatch.setattr(telegram_watch, "fetch_report", lambda n, r, root: COMPLETED_REPORT)
    watch = {"events_seen": 0, "statuses": {}, "resolved": {}}
    telegram_watch.poll_run(notifier, RUN, tmp_path, watch)
    assert watch["resolved"][RUN["run_id"]] is True
    assert any("0.4123" in text for text in sent)


def test_replay_events_forwards_only_new(monkeypatch, tmp_path: Path) -> None:
    notifier = make_notifier(monkeypatch)
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    state = {"events": [{"at": "a", "action": "one"}, {"at": "b", "action": "two"}]}
    watch = {"events_seen": 1, "statuses": {}, "resolved": {}}
    telegram_watch.replay_events(notifier, state, watch)
    assert watch["events_seen"] == 2
    assert len(sent) == 1
    assert "two" in sent[0]


def test_resolve_terminal_run_uses_stored_report(monkeypatch, tmp_path: Path) -> None:
    notifier = make_notifier(monkeypatch)
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    run = {**RUN, "status": "completed", "report": COMPLETED_REPORT}
    watch = {"events_seen": 0, "statuses": {}, "resolved": {}}
    telegram_watch.resolve_terminal_run(notifier, run, tmp_path, watch)
    assert watch["resolved"][RUN["run_id"]] is True
    assert any("0.4123" in text for text in sent)


def test_watch_once_ignores_prepared(monkeypatch, tmp_path: Path) -> None:
    notifier = make_notifier(monkeypatch)
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    state = {"events": [], "runs": {"r": {**RUN, "status": "prepared"}}}
    (tmp_path / ".sfsc").mkdir(parents=True)
    (tmp_path / ".sfsc" / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(telegram_watch, "kaggle_status", lambda kernel: "running")
    telegram_watch.watch_once(tmp_path, notifier, None)
    assert sent == []


def test_send_kernel_log_forwards_only_pending(monkeypatch, tmp_path: Path) -> None:
    notifier = make_notifier(monkeypatch)
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    entries = [
        {"stream_name": "stdout", "time": 1.0, "data": "one"},
        {"stream_name": "stdout", "time": 2.0, "data": "two"},
        {"stream_name": "stdout", "time": 3.0, "data": "three"},
    ]
    monkeypatch.setattr(telegram_watch, "kernel_log", lambda a, o, s: entries)
    run = {**RUN, "kernel": "hariohmbhatt/sfsc-cycle-001-full-train-e0b5f3ed"}
    watch = {
        "events_seen": 0,
        "statuses": {},
        "resolved": {},
        "log_sent": {RUN["run_id"]: 1},
    }
    telegram_watch.send_kernel_log(notifier, run, tmp_path, watch)
    assert watch["log_sent"][RUN["run_id"]] == 3
    joined = "\n".join(sent)
    assert "one" not in joined
    assert "two" in joined and "three" in joined


def test_send_kernel_log_skips_failed_fetch(monkeypatch, tmp_path: Path) -> None:
    notifier = make_notifier(monkeypatch)
    sent: list[str] = []
    notifier.send = lambda text, silent=False: sent.append(text) or True
    monkeypatch.setattr(
        telegram_watch, "kernel_log", lambda a, o, s: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    run = {**RUN, "kernel": "hariohmbhatt/sfsc-cycle-001-full-train-e0b5f3ed"}
    watch = {"events_seen": 0, "statuses": {}, "resolved": {}, "log_sent": {}}
    telegram_watch.send_kernel_log(notifier, run, tmp_path, watch)
    assert sent == []