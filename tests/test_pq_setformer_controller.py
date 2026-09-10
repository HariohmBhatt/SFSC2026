"""Behavioral tests for immutable PQ-SetFormer preparation and packaging."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "filament-pq-setformer"


def load_runner():
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location("pq_setformer_runner", HERE / "run.py")
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        return runner
    finally:
        sys.path.remove(str(HERE))
        for name in ("audit", "config", "core", "data"):
            module = sys.modules.get(name)
            if module is not None and Path(module.__file__).resolve().parent == HERE:
                sys.modules.pop(name)


runner = load_runner()


def test_prepare_freezes_complete_source_config_and_membership(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260910-120000",
    )
    directory = Path(record["directory"])

    assert record["status"] == "prepared"
    assert record["automatic_submission"] is False
    assert record["kernel"] == "hariohmhemantbhatt/sfsc-pq-setformer-20260910-120000"
    assert record["dataset"]["source"] == "hariohmhemantbhatt/sfsc2026-private-data"
    assert record["dataset"]["files"] == 708
    assert record["dataset"]["bytes"] == 607_987_717
    assert len(record["dataset"]["sha256"]) == 64
    assert record["experiment"]["model"]["input_size"] == 1024
    assert len(record["training_names"]) == 289
    assert len(record["inference_names"]) == 2
    assert len(record["source_sha256"]) == 64
    assert len(record["dataset"]["annotation_sha256"]) == 64
    assert len(record["membership_sha256"]) == 64
    assert len(record["vendor_lock_sha256"]) == 64
    assert len(record["git_sha"]) == 40
    assert set(record["source_files"]) == set(runner.source_bundle())
    assert json.loads((directory / "manifest.json").read_text())["run_id"] == record["run_id"]
    assert set(runner.source_bundle()) <= {
        str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
    }


def test_training_account_guard_rejects_primary_kaggle_credentials(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260910-120003",
    )

    runner.assert_training_account(record, "- username: hariohmhemantbhatt\n")
    with pytest.raises(RuntimeError, match="requires Kaggle account hariohmhemantbhatt"):
        runner.assert_training_account(record, "- username: hariohmbhatt\n")


def test_prepare_freezes_selected_preflight_input_size(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260910-120005",
        input_size=1216,
    )

    assert record["stage"] == "E1-smoke-1216"
    assert record["experiment"]["model"]["input_size"] == 1216


def test_kernel_bootstrap_pins_runtime_commits_sm60_and_wandb(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260910-120001",
    )

    kernel = runner.build_kernel(record)
    source = (kernel / "run.py").read_text()
    lock = json.loads((HERE / "vendor-lock.json").read_text())

    assert lock["detectron2"]["commit"] in source
    assert lock["maskdino"]["commit"] in source
    assert 'TORCH_CUDA_ARCH_LIST", "6.0"' in source
    assert "wandb==0.30.0" in source
    assert "timm==0.9.16" in source
    assert "--force-reinstall" in source
    metadata = json.loads((kernel / "kernel-metadata.json").read_text())
    assert metadata["machine_shape"] == "NvidiaTeslaP100"
    assert metadata["dataset_sources"] == ["hariohmhemantbhatt/sfsc2026-private-data"]
    assert metadata["competition_sources"] == []


def test_kernel_bootstrap_patches_maskdino_legacy_torch_dispatch(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260910-120004",
    )

    source = (runner.build_kernel(record) / "run.py").read_text()

    assert 'legacy_dispatch = "AT_DISPATCH_FLOATING_TYPES(value.type(),"' in source
    assert "dispatch_source.count(legacy_dispatch) != 2" in source
    assert 'legacy_dispatch.replace("value.type()", "value.scalar_type()")' in source
    assert source.index("dispatch_source.replace(legacy_dispatch") < source.index(
        'command("sh", "make.sh", cwd=ops)'
    )


def test_build_rejects_a_mutated_frozen_snapshot(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260910-120002",
    )
    (Path(record["directory"]) / "core.py").write_text("changed")

    with pytest.raises(ValueError, match="Frozen source changed"):
        runner.build_kernel(record)


def test_prepare_refuses_to_replace_an_active_run(tmp_path) -> None:
    record_path = tmp_path / "record.json"
    runner.write_json(record_path, {"status": "running"})

    with pytest.raises(RuntimeError, match="active"):
        runner.prepare(record_path=record_path, outputs_root=tmp_path / "outputs")


@pytest.mark.parametrize("status", ["pushing", "remote_complete_fetch_pending"])
def test_ambiguous_push_or_fetch_failure_preserves_remote_state(tmp_path, status) -> None:
    record_path = tmp_path / "record.json"
    record = {"status": status, "run_id": "run-1"}

    runner.record_controller_failure(record, record_path, RuntimeError("network timeout"))

    saved = json.loads(record_path.read_text())
    assert saved["status"] == status
    assert saved["controller_error"] == "network timeout"


def test_push_parser_rejects_cli_error_even_when_process_exits_zero() -> None:
    output = "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached."

    with pytest.raises(RuntimeError, match="Maximum weekly GPU quota"):
        runner.kernel_from_push_output(output)
