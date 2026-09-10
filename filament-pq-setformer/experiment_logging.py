"""Required dual JSONL and offline W&B logging for immutable stages."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Self


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_finite(value: Any, path: str = "metric") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def digest_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(path) for path in paths), key=lambda item: item.name):
        if not path.is_file():
            raise ValueError(f"Artifact input is not a file: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class ExperimentLogger:
    """Mirror one canonical event stream to JSONL and an offline W&B run."""

    def __init__(
        self,
        *,
        context: Mapping[str, Any],
        directory: Path,
        wandb_module: ModuleType | Any,
        run: Any,
        manifest: dict[str, Any],
    ) -> None:
        self.context = dict(context)
        self.directory = directory
        self.wandb = wandb_module
        self.run = run
        self.manifest = manifest
        self.events_path = directory / "events.jsonl"
        self.manifest_path = directory / "logging-manifest.json"
        self.finished = False

    @classmethod
    def start(
        cls,
        context: Mapping[str, Any],
        directory: Path,
        *,
        wandb_module: ModuleType | Any | None = None,
    ) -> Self:
        required = {
            "experiment_run_id",
            "stage",
            "execution_source_sha256",
            "project",
            "job_type",
            "config",
        }
        missing = required - set(context)
        if missing:
            raise ValueError(f"Logging context is missing: {sorted(missing)}")
        module = wandb_module or importlib.import_module("wandb")
        expected_version = str(context.get("wandb_version", "0.30.0"))
        if str(getattr(module, "__version__", "")) != expected_version:
            raise RuntimeError(
                f"W&B version mismatch: expected {expected_version}, "
                f"found {getattr(module, '__version__', 'unknown')}"
            )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        identity = ":".join(
            (
                str(context["experiment_run_id"]),
                str(context["stage"]),
                str(context["execution_source_sha256"]),
            )
        )
        wandb_run_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        run = module.init(
            project=str(context["project"]),
            mode="offline",
            dir=str(directory / "wandb"),
            id=wandb_run_id,
            name=f"{context['experiment_run_id']}-{context['stage']}",
            group=str(context["experiment_run_id"]),
            job_type=str(context["job_type"]),
            tags=list(context.get("tags", [])),
            config=dict(context["config"]),
        )
        if run is None:
            raise RuntimeError("W&B offline initialization returned no run")
        now = datetime.now(UTC).isoformat()
        manifest = {
            "status": "running",
            "started_at": now,
            "wandb_run_id": wandb_run_id,
            "wandb_version": expected_version,
            "mode": "offline",
            "project": str(context["project"]),
            "group": str(context["experiment_run_id"]),
            "job_type": str(context["job_type"]),
            "stage": str(context["stage"]),
            "events": "events.jsonl",
            "wandb_directory": "wandb",
        }
        _write_json(directory / "logging-manifest.json", manifest)
        return cls(
            context=context,
            directory=directory,
            wandb_module=module,
            run=run,
            manifest=manifest,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, error_type, error, traceback) -> bool:
        if error is None:
            self.finish("complete")
        else:
            self.finish("failed", error=str(error))
        return False

    def _append(self, event: str, step: int | None, values: Mapping[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "step": step,
            "values": dict(sorted(values.items())),
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def log(self, event: str, *, step: int | None = None, **values: Any) -> None:
        if self.finished:
            raise RuntimeError("Cannot log after the stage has finished")
        if not event:
            raise ValueError("event must be non-empty")
        _validate_finite(values)
        self._append(event, step, values)
        self.run.log({**values, "event": event}, step=step)

    def log_table(self, name: str, rows: Sequence[Mapping[str, Any]]) -> Any:
        if self.finished or not name or not rows:
            raise ValueError("Table logging requires an active run, name, and rows")
        columns = sorted({key for row in rows for key in row})
        data = [[row.get(column) for column in columns] for row in rows]
        table = self.wandb.Table(columns=columns, data=data)
        self.run.log({f"tables/{name}": table})
        self._append("table_logged", None, {"name": name, "rows": len(rows)})
        return table

    def log_artifact(
        self,
        kind: str,
        paths: Sequence[Path],
        metadata: Mapping[str, Any],
    ) -> Any:
        if self.finished or not kind or not paths:
            raise ValueError("Artifact logging requires an active run, kind, and files")
        checksum = digest_paths(paths)
        artifact_metadata = {**dict(metadata), "sha256": checksum}
        artifact = self.wandb.Artifact(
            f"{kind}-{checksum[:12]}",
            type=kind,
            metadata=artifact_metadata,
        )
        for path in paths:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact)
        self._append(
            "artifact_logged",
            None,
            {"kind": kind, "files": len(paths), "sha256": checksum},
        )
        return artifact

    def finish(self, status: str, *, error: str | None = None) -> None:
        if self.finished:
            return
        if status not in {"complete", "failed"}:
            raise ValueError("Terminal logging status must be complete or failed")
        values = {"status": status}
        if error is not None:
            values["error"] = error
        self._append("stage_terminal", None, values)
        self.run.log({**values, "event": "stage_terminal"})
        self.run.summary["status"] = status
        if error is not None:
            self.run.summary["error"] = error
        self.run.finish(exit_code=0 if status == "complete" else 1)
        self.manifest.update(
            status=status,
            completed_at=datetime.now(UTC).isoformat(),
            error=error,
        )
        _write_json(self.manifest_path, self.manifest)
        self.finished = True
