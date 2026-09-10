"""Behavioral tests for the dual local/W&B logging seam."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "filament-pq-setformer/experiment_logging.py"
spec = importlib.util.spec_from_file_location("pq_setformer_logging", MODULE)
logging = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = logging
spec.loader.exec_module(logging)


class FakeSummary(dict):
    pass


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = FakeSummary()
        self.artifacts = []
        self.finished = False

    def log(self, values, step=None):
        self.logged.append((values, step))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code=0):
        self.finished = True
        self.exit_code = exit_code


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeWandb:
    __version__ = "0.30.0"
    Artifact = FakeArtifact

    def __init__(self):
        self.init_kwargs = None
        self.run = FakeRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    @staticmethod
    def Table(columns, data):
        return {"columns": columns, "data": data}


def context():
    return {
        "experiment_run_id": "20260910-120000",
        "stage": "smoke-train",
        "execution_source_sha256": "a" * 64,
        "project": "sfsc2026-pq-setformer",
        "job_type": "train",
        "tags": ["smoke", "1280"],
        "config": {"queries": 64, "seed": 20260910},
    }


def test_logger_mirrors_steps_tables_artifacts_and_terminal_status(tmp_path) -> None:
    fake = FakeWandb()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")

    with logging.ExperimentLogger.start(context(), tmp_path, wandb_module=fake) as logger:
        logger.log("optimizer_step", step=1, loss=0.5, grad_norm=1.2)
        logger.log_table("validation_cases", [{"image": "a.jpeg", "pq": 0.7}])
        artifact = logger.log_artifact("model", [checkpoint], {"seed": 20260910})

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    manifest = json.loads((tmp_path / "logging-manifest.json").read_text())
    assert fake.init_kwargs["mode"] == "offline"
    assert fake.init_kwargs["id"] == manifest["wandb_run_id"]
    assert fake.run.logged[0] == ({"loss": 0.5, "grad_norm": 1.2, "event": "optimizer_step"}, 1)
    assert events[0]["values"] == {"grad_norm": 1.2, "loss": 0.5}
    assert events[-1]["event"] == "stage_terminal"
    assert events[-1]["values"]["status"] == "complete"
    assert fake.run.summary["status"] == "complete"
    assert fake.run.finished
    assert artifact.metadata["sha256"] == logging.digest_paths([checkpoint])
    assert manifest["status"] == "complete"


def test_logger_records_failure_and_rejects_nonfinite_metrics(tmp_path) -> None:
    fake = FakeWandb()
    with (
        pytest.raises(RuntimeError, match="boom"),
        logging.ExperimentLogger.start(context(), tmp_path, wandb_module=fake) as logger,
    ):
        with pytest.raises(ValueError, match="finite"):
            logger.log("bad", step=1, loss=float("nan"))
        raise RuntimeError("boom")

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["values"] == {"error": "boom", "status": "failed"}
    assert fake.run.summary["status"] == "failed"
    assert fake.run.exit_code == 1


def test_logger_refuses_wrong_wandb_version_before_writing_events(tmp_path) -> None:
    fake = FakeWandb()
    fake.__version__ = "0.29.0"

    with pytest.raises(RuntimeError, match="W&B version"):
        logging.ExperimentLogger.start(
            {**context(), "wandb_version": "0.30.0"},
            tmp_path,
            wandb_module=fake,
        )

    assert not (tmp_path / "events.jsonl").exists()
