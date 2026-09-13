"""Behavioral tests for immutable PQ-SetFormer preparation and packaging."""

import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
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


def test_controller_cli_uses_the_isolated_kaggle2_credentials(monkeypatch) -> None:
    seen = {}

    def check_output(command, **kwargs):
        seen.update(command=command, environment=kwargs["env"])
        return "- username: hariohmhemantbhatt\n"

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)

    assert runner._cli("config", "view") == "- username: hariohmhemantbhatt\n"
    assert seen["command"] == ["kaggle", "config", "view"]
    assert seen["environment"]["KAGGLE_CONFIG_DIR"] == str(runner.KAGGLE2_CONFIG)
    assert seen["environment"]["HOME"] == str(runner.KAGGLE2_HOME)


def test_prepare_freezes_selected_preflight_input_size(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260910-120005",
        input_size=1216,
    )

    assert record["stage"] == "E1-smoke-1216"
    assert record["experiment"]["model"]["input_size"] == 1216


def test_prepare_freezes_full_e1_pilot_schedule_and_assessment(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260910-120006",
        input_size=1216,
        run_kind="pilot",
    )

    assert record["stage"] == "E1-pilot-1216"
    assert record["run_kind"] == "pilot"
    assert record["training_microbatches"] == 6_000
    assert len(record["training_names"]) == 289
    assert record["inference_names"] == (
        record["split"]["members"]["quality_fit"]
        + record["split"]["members"]["quality_calibration"]
    )
    assert len(record["inference_names"]) == 103
    assert record["automatic_submission"] is False


def test_prepare_freezes_extended_preflight_at_two_hundred_optimizer_updates(tmp_path) -> None:
    record = runner.prepare(
        record_path=tmp_path / "record.json",
        outputs_root=tmp_path / "outputs",
        run_id="20260913-120000",
        run_kind="preflight",
    )

    assert record["stage"] == "E1-preflight-1024"
    assert record["training_microbatches"] == 800
    assert (
        record["training_microbatches"] // record["experiment"]["training"]["accumulation"] == 200
    )
    assert len(record["inference_names"]) == 2


def test_prepare_rejects_unknown_run_kind(tmp_path) -> None:
    with pytest.raises(ValueError, match="run_kind"):
        runner.prepare(
            record_path=tmp_path / "record.json",
            outputs_root=tmp_path / "outputs",
            run_id="20260910-120007",
            run_kind="deployment",
        )


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


def test_kernel_bootstrap_compiles_with_the_pinned_e2_quality_adapter() -> None:
    lock = json.loads((HERE / "vendor-lock.json").read_text())

    launcher = runner._bootstrap("eJyrVkrLz1eyUkpKLFKqBQAQ5gQF", lock)

    compile(launcher, "launcher", "exec")
    assert "PQ-SetFormer E2 adapter" in launcher


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


def test_gate_rejects_unverified_or_tampered_manual_assessment(tmp_path) -> None:
    record_path = tmp_path / "record.json"
    manual = {
        "verified": False,
        "matched_control": {"run_id": "control"},
        "diagnostics": {
            "raw_candidate_recall": 0.8,
            "final_recall": 0.7,
            "fragmentation": 0,
            "merges": 0,
            "instance_count_absolute_error": 0,
        },
        "bootstrap": {"pq_delta": 0.02, "bootstrap_lower_95": 0.001},
        "rq_delta": 0.01,
        "sq_delta": 0.0,
    }
    record = {"status": "assessment_complete", "assessment": manual}

    with pytest.raises(ValueError, match="verified"):
        runner.record_promotion_gate(record, record_path)

    manual["verified"] = True
    manual["assessment_contract_sha256"] = "a" * 64
    manual["sha256"] = runner.canonical_digest(
        {key: value for key, value in manual.items() if key != "sha256"}
    )
    record["assessment_contract"] = {"sha256": "a" * 64}
    manual["bootstrap"]["pq_delta"] = 0.9

    with pytest.raises(ValueError, match="seal"):
        runner.record_promotion_gate(record, record_path)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _verified_assessment_fixture(tmp_path):
    filename = "20110101000000Bh.jpeg"
    membership_sha256 = runner.canonical_digest([filename])
    training_name = "20100101000000Bh.jpeg"
    training_membership_sha256 = runner.canonical_digest([training_name])
    annotation_path = tmp_path / "annotations.json"
    annotation = {
        "images": [{"id": 1, "file_name": filename}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "segmentation": [[256, 256, 768, 256, 768, 768, 256, 768]],
            }
        ],
    }
    _write_json(annotation_path, annotation)

    mask = np.zeros((2048, 2048), dtype=bool)
    mask[256:769, 256:769] = True
    probability = np.where(mask, 0.99, 0.01).astype(np.float32)
    candidate = runner.core.Candidate(
        "q000",
        mask,
        0.99,
        1.0,
        probability,
        presence_logit=4.6,
        source="identity",
    )
    candidate_path = tmp_path / "h-candidates.json"
    _write_json(candidate_path, {filename: [runner.core.serialize_candidate(candidate)]})
    cached = runner.core.deserialize_candidate(json.loads(candidate_path.read_text())[filename][0])
    index = runner.data.load_annotation_index(annotation_path, {filename})
    truths = {
        filename: [
            list(runner.data.rasterize_record(row, native_size=2048, output_size=2048))
            for row in index.records_by_file[filename]
        ]
    }

    calibrator = runner.core.Calibrator(
        feature_names=runner.core.FEATURE_NAMES,
        mean=(0.0,) * len(runner.core.FEATURE_NAMES),
        scale=(1.0,) * len(runner.core.FEATURE_NAMES),
        coefficients=(10.0,) + (0.0,) * len(runner.core.FEATURE_NAMES),
        regularization=0.02,
        examples=1,
        membership_sha256="f" * 64,
        candidate_sha256="e" * 64,
    )
    recipe = runner.AssemblyConfig(utility_threshold=0.0, minimum_area=0)
    predictions = {
        filename: runner.core.assemble_instances([cached], recipe=recipe, calibrator=calibrator)
    }
    challenger_metrics = runner.core.evaluate_run(predictions, truths)

    calibration_path = tmp_path / "q-calibration.json"
    calibration = {
        "schema_version": 1,
        "kind": "q-fit-q-cal-only",
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "membership_sha256": "3" * 64,
        "calibrator": calibrator.to_dict(),
        "calibrator_sha256": calibrator.sha256,
        "assembly": asdict(recipe),
        "selection": {"selection_split": "Q-cal"},
    }
    calibration["sha256"] = runner.canonical_digest(calibration)
    _write_json(calibration_path, calibration)

    challenger_checkpoint = tmp_path / "challenger.pt"
    challenger_checkpoint.write_bytes(b"challenger")
    control_checkpoint = tmp_path / "control.pt"
    control_checkpoint.write_bytes(b"control")
    challenger_checkpoint_sha256 = runner.digest(challenger_checkpoint)
    control_checkpoint_sha256 = runner.digest(control_checkpoint)

    control_predictions_path = tmp_path / "control-predictions.json"
    _write_json(
        control_predictions_path,
        {"schema_version": 1, "predictions": {filename: []}},
    )
    control_predictions_sha256 = runner.digest(control_predictions_path)

    challenger_report_path = tmp_path / "challenger-report.json"
    challenger_report = {
        "status": "complete",
        "run_id": "challenger-1",
        "model_role": "challenger",
        "assessment_membership_sha256": membership_sha256,
        "training_membership_sha256": training_membership_sha256,
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "checkpoint_sha256": challenger_checkpoint_sha256,
        "candidates_sha256": runner.digest(candidate_path),
        "inference": challenger_metrics,
    }
    _write_json(challenger_report_path, challenger_report)

    control_report_path = tmp_path / "control-report.json"
    control_report = {
        "status": "complete",
        "run_id": "control-1",
        "model_role": "matched_r_control",
        "assessment_membership_sha256": membership_sha256,
        "training_membership_sha256": training_membership_sha256,
        "source_sha256": "4" * 64,
        "config_sha256": "5" * 64,
        "checkpoint_sha256": control_checkpoint_sha256,
        "predictions_sha256": control_predictions_sha256,
        "inference": {
            "pq": 0.0,
            "sq": 0.0,
            "rq": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 1,
            "sum_iou": 0.0,
            "files": {
                filename: {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 1,
                    "matched_iou": 0.0,
                }
            },
        },
    }
    _write_json(control_report_path, control_report)

    record_path = tmp_path / "record.json"
    record = {
        "status": "calibration_complete",
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "calibration_path": str(calibration_path),
        "calibration_sha256": runner.digest(calibration_path),
        "dataset": {"annotation_sha256": runner.digest(annotation_path)},
        "report": {"checkpoint_sha256": challenger_checkpoint_sha256},
        "split": {
            "members": {"assessment": [filename], "fitting": [training_name]},
            "assessment_blocks": [date(2011, 1, 1).toordinal() // 60],
        },
    }
    contract_path = tmp_path / "assessment-contract.json"
    contract = {
        "assessment_membership_sha256": membership_sha256,
        "challenger": {
            "run_id": "challenger-1",
            "model_role": "challenger",
            "source_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "training_membership_sha256": training_membership_sha256,
            "checkpoint_path": str(challenger_checkpoint),
            "checkpoint_sha256": challenger_checkpoint_sha256,
        },
        "matched_control": {
            "run_id": "control-1",
            "model_role": "matched_r_control",
            "source_sha256": "4" * 64,
            "config_sha256": "5" * 64,
            "training_membership_sha256": training_membership_sha256,
            "checkpoint_path": str(control_checkpoint),
            "checkpoint_sha256": control_checkpoint_sha256,
        },
    }
    _write_json(contract_path, contract)
    runner.freeze_h_assessment_contract(record, record_path, contract_path)

    bundle_path = tmp_path / "assessment-bundle.json"
    bundle = {
        "assessment_membership_sha256": membership_sha256,
        "assessment_contract_sha256": record["assessment_contract"]["sha256"],
        "challenger": {
            "report_path": str(challenger_report_path),
            "report_sha256": runner.digest(challenger_report_path),
            "candidates_path": str(candidate_path),
            "candidates_sha256": runner.digest(candidate_path),
            "checkpoint_path": str(challenger_checkpoint),
            "checkpoint_sha256": challenger_checkpoint_sha256,
        },
        "matched_control": {
            "report_path": str(control_report_path),
            "report_sha256": runner.digest(control_report_path),
            "predictions_path": str(control_predictions_path),
            "predictions_sha256": control_predictions_sha256,
            "checkpoint_path": str(control_checkpoint),
            "checkpoint_sha256": control_checkpoint_sha256,
        },
        # These values are deliberately false; the verifier must ignore and recompute them.
        "bootstrap": {"pq_delta": -99.0},
        "diagnostics": {"raw_candidate_recall": -99.0},
    }
    _write_json(bundle_path, bundle)
    return record, record_path, bundle_path, annotation_path, candidate_path


def test_h_assessment_recomputes_evidence_and_only_verified_record_can_promote(tmp_path) -> None:
    record, record_path, bundle_path, annotation_path, _ = _verified_assessment_fixture(tmp_path)

    assessed = runner.record_frozen_h_assessment(
        record, record_path, bundle_path, annotation_path=annotation_path
    )

    assert assessed["status"] == "assessment_complete"
    assert assessed["assessment"]["verified"] is True
    assert assessed["assessment"]["bootstrap"]["pq_delta"] > 0.9
    assert assessed["assessment"]["diagnostics"]["raw_candidate_recall"] > 0.9
    assert assessed["assessment"]["evaluator_parity_error"] <= 1e-8
    assert runner.record_promotion_gate(assessed, record_path)["status"] == "promoted"


def test_h_assessment_rejects_candidate_artifact_tampered_after_bundle_freeze(tmp_path) -> None:
    record, record_path, bundle_path, annotation_path, candidate_path = (
        _verified_assessment_fixture(tmp_path)
    )
    candidate_path.write_text("{}\n")

    with pytest.raises(ValueError, match="hash-mismatched"):
        runner.record_frozen_h_assessment(
            record, record_path, bundle_path, annotation_path=annotation_path
        )


def test_h_assessment_rejects_control_metrics_not_reproduced_by_control_masks(tmp_path) -> None:
    record, record_path, bundle_path, annotation_path, _ = _verified_assessment_fixture(tmp_path)
    bundle = json.loads(bundle_path.read_text())
    report_path = Path(bundle["matched_control"]["report_path"])
    report = json.loads(report_path.read_text())
    report["inference"].update(
        pq=1.0,
        sq=1.0,
        rq=1.0,
        tp=1,
        fp=0,
        fn=0,
        sum_iou=1.0,
    )
    report["inference"]["files"][next(iter(report["inference"]["files"]))] = {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "matched_iou": 1.0,
    }
    _write_json(report_path, report)
    bundle["matched_control"]["report_sha256"] = runner.digest(report_path)
    _write_json(bundle_path, bundle)

    with pytest.raises(ValueError, match="Matched-control H evaluator metrics"):
        runner.record_frozen_h_assessment(
            record, record_path, bundle_path, annotation_path=annotation_path
        )
