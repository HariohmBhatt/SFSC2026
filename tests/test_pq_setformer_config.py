"""Behavioral tests for frozen PQ-SetFormer configuration."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "filament-pq-setformer" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load_module("pq_setformer_config", "config.py")


def test_default_config_is_hash_stable_and_round_trips() -> None:
    first = config.ExperimentConfig()
    second = config.ExperimentConfig.from_dict(first.to_dict())

    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert first.model.queries == 64
    assert first.training.physical_batch == 1
    assert first.training.accumulation == 4
    assert first.logging.mode == "offline"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("queries", 26, "more queries"),
        ("input_size", 1000, "divisible by 32"),
        ("peak_vram_gb", 15.0, "below 15"),
    ],
)
def test_model_config_rejects_unsafe_p100_values(field, value, message) -> None:
    original = config.ModelConfig()

    with pytest.raises(ValueError, match=message):
        replace(original, **{field: value})


def test_config_hash_changes_for_a_training_change() -> None:
    original = config.ExperimentConfig()
    changed = replace(
        original,
        training=replace(original.training, decoder_lr=5e-5),
    )

    assert original.sha256 != changed.sha256
