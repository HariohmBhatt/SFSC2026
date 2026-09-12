"""Unit tests for the multi-account Kaggle CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfsc import kaggle_cli

QUOTA_CSV = """resource,used,remaining,total,refreshAt
GPU,0.00h,30.00h,30.00h,2026-09-19T00:00:00
TPU,0.00h,20.00h,20.00h,2026-09-19T00:00:00
"""


def make_home(tmp_path: Path, with_second: bool = True) -> Path:
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "credentials.json").write_text(
        json.dumps(
            {
                "username": "hariohmbhatt",
                "refresh_token": "refresh-1",
                "access_token": "acc-1",
                "access_token_expiration": "2099-01-01T00:00:00+00:00",
            }
        )
    )
    if with_second:
        second = home / ".kaggle-2" / ".kaggle"
        second.mkdir(parents=True)
        (second / "credentials.json").write_text(
            json.dumps(
                {
                    "username": "hariohmhemantbhatt",
                    "refresh_token": "refresh-2",
                    "access_token": "acc-2",
                    "access_token_expiration": "2099-01-01T00:00:00+00:00",
                }
            )
        )
    return home


def test_accounts_discovery(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    found = kaggle_cli.accounts()
    assert [account.name for account in found] == ["kaggle", "kaggle2"]
    assert kaggle_cli.config_username(found[0]) == "hariohmbhatt"
    assert kaggle_cli.config_username(found[1]) == "hariohmhemantbhatt"


def test_second_account_omitted_when_absent(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path, with_second=False)
    monkeypatch.setattr(Path, "home", lambda: home)
    assert [account.name for account in kaggle_cli.accounts()] == ["kaggle"]


def test_account_for_owner(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    assert kaggle_cli.account_for_owner("hariohmbhatt").name == "kaggle"
    assert kaggle_cli.account_for_owner("hariohmhemantbhatt").name == "kaggle2"
    assert kaggle_cli.account_for_owner("stranger").name == "kaggle"


def test_account_for_owner_falls_back_without_config(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "bare-home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    account = kaggle_cli.account_for_owner("anyone")
    assert account.name == "kaggle"
    assert account.config_dir == home / ".kaggle"


def test_run_cli_injects_account_env(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    captured: dict[str, str] = {}

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, capture_output, text, timeout, check, env):
        captured.update(env)
        return Result()

    monkeypatch.setattr(kaggle_cli.subprocess, "run", fake_run)
    account = kaggle_cli.accounts()[1]
    output = kaggle_cli.run_cli(["quota", "--csv"], account)
    assert output == "ok"
    assert captured["KAGGLE_CONFIG_DIR"] == str(home / ".kaggle-2" / ".kaggle")
    assert captured["HOME"] == str(home / ".kaggle-2")


def test_run_cli_raises_on_failure(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)

    class Result:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(kaggle_cli.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match="boom"):
        kaggle_cli.run_cli(["kernels", "status", "o/k"], kaggle_cli.accounts()[0])


def test_quota_parses_csv(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(kaggle_cli, "run_cli", lambda *a, **k: QUOTA_CSV)
    rows = kaggle_cli.quota(kaggle_cli.accounts()[0])
    assert len(rows) == 2
    assert rows[0]["resource"] == "GPU"
    assert rows[0]["remaining"] == "30.00h"
    assert rows[1]["refreshAt"] == "2026-09-19T00:00:00"


def test_quota_all(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(kaggle_cli, "run_cli", lambda *a, **k: QUOTA_CSV)
    result = kaggle_cli.quota_all()
    assert set(result) == {"kaggle", "kaggle2"}
    assert result["kaggle"]["username"] == "hariohmbhatt"
    assert result["kaggle2"]["username"] == "hariohmhemantbhatt"


def test_kernel_log_parses_entries(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    raw_log = json.dumps(
        [
            {"stream_name": "stdout", "time": 12.5, "data": "TICK 000"},
            {"stream_name": "stderr", "time": 12.9, "data": "ERR"},
            {"stream_name": "stdout", "time": 13.1, "data": ""},
            "not a dict",
        ]
    )
    monkeypatch.setattr(
        kaggle_cli,
        "_post",
        lambda account, method, body, token=None: {"log": raw_log},
    )
    entries = kaggle_cli.kernel_log(kaggle_cli.accounts()[0], "hariohmbhatt", "k")
    assert len(entries) == 2
    assert entries[0] == {"stream_name": "stdout", "time": 12.5, "data": "TICK 000"}
    assert entries[1]["stream_name"] == "stderr"


def test_kernel_log_empty_and_bad_json(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(kaggle_cli, "_post", lambda *a, **k: {"log": ""})
    assert kaggle_cli.kernel_log(kaggle_cli.accounts()[0], "hariohmbhatt", "k") == []
    monkeypatch.setattr(kaggle_cli, "_post", lambda *a, **k: {"log": "not json"})
    assert kaggle_cli.kernel_log(kaggle_cli.accounts()[0], "hariohmbhatt", "k") == []


def test_access_token_refreshes_when_expired(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    creds = json.loads((home / ".kaggle" / "credentials.json").read_text())
    creds["access_token_expiration"] = "2020-01-01T00:00:00+00:00"
    creds["access_token"] = "old"
    (home / ".kaggle" / "credentials.json").write_text(json.dumps(creds))
    monkeypatch.setattr(
        kaggle_cli,
        "_post",
        lambda account, method, body, token=None: {
            "token": "new-token",
            "expiresIn": 1800,
            "userName": "hariohmbhatt",
        },
    )
    token = kaggle_cli.access_token(kaggle_cli.accounts()[0])
    assert token == "new-token"
    refreshed = json.loads((home / ".kaggle" / "credentials.json").read_text())
    assert refreshed["access_token"] == "new-token"


def test_access_token_uses_valid_stored(monkeypatch, tmp_path: Path) -> None:
    home = make_home(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    token = kaggle_cli.access_token(kaggle_cli.accounts()[0])
    assert token == "acc-1"