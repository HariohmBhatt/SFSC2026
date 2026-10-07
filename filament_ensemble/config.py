from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Paths:
    drive: Path
    scratch: Path
    experiment: str

    @property
    def persistent(self) -> Path:
        return self.drive / "outputs" / self.experiment

    @property
    def checkpoints(self) -> Path:
        return self.drive / "models" / self.experiment

    @property
    def cache(self) -> Path:
        return self.drive / "cache" / self.experiment


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {
        "experiment",
        "drive_root",
        "scratch_root",
        "detectors",
        "refiner_backbones",
        "refiner_folds",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if not str(config["drive_root"]).startswith("/content/drive/MyDrive/"):
        raise ValueError("Persistent storage must be under mounted Google Drive")
    return config


def paths(config: dict[str, Any]) -> Paths:
    result = Paths(
        drive=Path(config["drive_root"]),
        scratch=Path(config["scratch_root"]),
        experiment=str(config["experiment"]),
    )
    if result.drive.name == "MyDrive" and not result.drive.is_dir():
        raise RuntimeError("Google Drive is not mounted")
    for directory in (result.persistent, result.checkpoints, result.cache, result.scratch):
        directory.mkdir(parents=True, exist_ok=True)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
