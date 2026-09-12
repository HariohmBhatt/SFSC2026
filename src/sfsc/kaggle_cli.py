"""Multi-account Kaggle CLI helpers.

The primary Kaggle account uses the default ``~/.kaggle`` config directory.
A second account is isolated by giving it its own ``$HOME``
(``~/.kaggle-2``), because the Kaggle CLI hardcodes OAuth credentials to
``$HOME/.kaggle/credentials.json``. This module mirrors that convention so
every ``kaggle`` invocation can run under the correct identity.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

CLI_TIMEOUT = 300
API_BASE = "https://api.kaggle.com/v1"
HTTP_TIMEOUT = 30
TOKEN_BUFFER_MINUTES = 30

ACCOUNT1_NAME = "kaggle"
ACCOUNT2_NAME = "kaggle2"
ACCOUNT2_HOME_NAME = ".kaggle-2"


@dataclass(frozen=True)
class Account:
    """A Kaggle identity and the environment required to invoke it."""

    name: str
    config_dir: Path
    home: Path | None

    @property
    def exists(self) -> bool:
        return any(
            (self.config_dir / name).is_file()
            for name in ("kaggle.json", "credentials.json")
        )

    def env(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment["KAGGLE_CONFIG_DIR"] = str(self.config_dir)
        if self.home is not None:
            environment["HOME"] = str(self.home)
        return environment


def accounts() -> list[Account]:
    """Return configured accounts; the second is included only if present."""
    home = Path.home()
    primary = Account(ACCOUNT1_NAME, home / ".kaggle", None)
    secondary = Account(
        ACCOUNT2_NAME, home / ACCOUNT2_HOME_NAME / ".kaggle", home / ACCOUNT2_HOME_NAME
    )
    return [account for account in (primary, secondary) if account.exists]


def config_username(account: Account) -> str | None:
    """Read the account username without exposing any credentials."""
    for name in ("kaggle.json", "credentials.json"):
        path = account.config_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(data, dict) and data.get("username"):
            return str(data["username"])
    return None


def account_for_owner(owner: str) -> Account:
    """Resolve the account that owns a kernel by its username prefix."""
    configured = accounts()
    if not configured:
        return Account(ACCOUNT1_NAME, Path.home() / ".kaggle", None)
    for account in configured:
        if config_username(account) == owner:
            return account
    return configured[0]


def run_cli(arguments: list[str], account: Account, timeout: float = CLI_TIMEOUT) -> str:
    completed = subprocess.run(
        ["kaggle", *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=account.env(),
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"kaggle {' '.join(arguments)} failed: {message[:400]}")
    return completed.stdout


def quota(account: Account) -> list[dict[str, str]]:
    """Parse ``kaggle quota --csv`` into rows for this account."""
    output = run_cli(["quota", "--csv"], account)
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return []
    return [dict(row) for row in csv.DictReader(io.StringIO("\n".join(lines)))]


def quota_all() -> dict[str, dict]:
    """Fetch quota rows for every configured account."""
    result: dict[str, dict] = {}
    for account in accounts():
        try:
            rows = quota(account)
        except (RuntimeError, ValueError, OSError):
            continue
        result[account.name] = {
            "username": config_username(account) or account.name,
            "rows": rows,
        }
    return result


def _post(account: Account, method: str, body: dict, token: str | None = None) -> dict:
    url = f"{API_BASE}/{method}"
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        api_key = _api_key_auth(account)
        if api_key is not None:
            headers["Authorization"] = f"Basic {api_key}"
    request = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Kaggle API {method} failed: HTTP {error.code}") from error


def _api_key_auth(account: Account) -> str | None:
    """Return Basic auth for a classic ``kaggle.json`` account, if present."""
    path = account.config_dir / "kaggle.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    username = data.get("username")
    key = data.get("key")
    if not username or not key:
        return None
    return base64.b64encode(f"{username}:{key}".encode()).decode()


def access_token(account: Account) -> str | None:
    """Return a valid OAuth access token, refreshing the stored one when stale."""
    path = account.config_dir / "credentials.json"
    if not path.is_file():
        return None
    try:
        credentials = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    refresh_token = credentials.get("refresh_token")
    expiration = credentials.get("access_token_expiration")
    if refresh_token:
        if expiration:
            try:
                expiry = datetime.fromisoformat(expiration)
            except ValueError:
                expiry = None
            if expiry and expiry < datetime.now(UTC) - timedelta(minutes=TOKEN_BUFFER_MINUTES):
                _refresh_access_token(account, credentials)
                return credentials["access_token"]
        return credentials.get("access_token")
    return credentials.get("access_token")


def _refresh_access_token(account: Account, credentials: dict) -> None:
    data = _post(
        account,
        "users.AccountService/GenerateAccessToken",
        {"refreshToken": credentials["refresh_token"], "apiVersion": 1},
    )
    expires_in = int(data.get("expiresIn") or 3600)
    credentials["access_token"] = data["token"]
    credentials["access_token_expiration"] = (
        datetime.now(UTC) + timedelta(seconds=expires_in)
    ).isoformat()
    path = account.config_dir / "credentials.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(credentials, indent=1))
    tmp.replace(path)


def kernel_log(account: Account, owner: str, slug: str) -> list[dict]:
    """Return the timestamped stdout/stderr entries of a completed kernel.

    Each entry mirrors the Kaggle session log format:
    ``{"stream_name": "stdout"|"stderr", "time": <seconds>, "data": "<line>"}``.
    Kaggle only materializes this log after the kernel finishes.
    """
    token = access_token(account)
    data = _post(
        account,
        "kernels.KernelsApiService/ListKernelSessionOutput",
        {"userName": owner, "kernelSlug": slug},
        token=token,
    )
    log = data.get("log") or ""
    if not log.strip():
        return []
    try:
        entries = json.loads(log)
    except (ValueError, TypeError):
        return []
    return [
        entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("data"), str) and entry["data"]
    ]