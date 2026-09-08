"""Regression coverage for OOF exclusion, quality targets, and mask assembly."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "filament-oof-consensus/core.py"
spec = importlib.util.spec_from_file_location("consensus_core", MODULE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def rle(mask):
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))


def candidate(mask, pid, kind="refined", feature=0):
    return {
        "proposal_id": pid,
        "kind": kind,
        "rle": core.encode(mask),
        "area": int(mask.sum()),
        "features": [feature] + [0] * (len(core.FEATURES) - 1),
    }


def quality_model():
    return {
        "feature_names": core.FEATURES,
        "mean": [0] * len(core.FEATURES),
        "scale": [1] * len(core.FEATURES),
        "coefficients": [0, 1] + [0] * (len(core.FEATURES) - 1),
    }


def test_iou_half_is_not_a_match_and_empty_predictions_count_misses():
    truth = np.ones((8, 8), bool)
    half = truth.copy()
    half[:4] = False
    out = core.metrics(core.score([[rle(truth)]], [half]))
    assert out["tp"] == 0 and out["fp"] == 1 and out["fn"] == 1
    assert core.metrics(core.score([[rle(truth)]], []))["fn"] == 1


def test_quality_target_is_unconditional_contribution_across_annotators():
    mask = np.zeros((8, 8), bool)
    mask[:4] = True
    rows = [candidate(mask, 0)]
    core.annotate_targets(rows, [[rle(mask)], []])
    assert rows[0]["target"] == 0.5
    assert rows[0]["record_weight"] == 2


def test_duplicate_prompts_receive_same_presence_target():
    mask = np.ones((8, 8), bool)
    rows = [candidate(mask, 0), candidate(mask, 1)]
    core.annotate_targets(rows, [[rle(mask)]])
    assert [c["target"] for c in rows] == [1, 1]


def test_seed_and_refinement_are_mutually_exclusive_and_overlaps_are_removed():
    seed = np.zeros((16, 16), bool)
    seed[:10, :10] = True
    refined = np.zeros_like(seed)
    refined[:9, :9] = True
    neighbor = np.zeros_like(seed)
    neighbor[8:, 8:] = True
    rows = [
        candidate(seed, 0, "seed", 0),
        candidate(refined, 0, feature=2),
        candidate(neighbor, 1, feature=1),
        candidate(refined, 2, feature=0.5),
    ]
    masks = core.assemble(rows, quality_model(), threshold=0, minimum_area=1)
    assert len(masks) == 2
    assert np.array_equal(masks[0], refined)
    assert not (masks[0] & masks[1]).any()
    assert core.assemble([], quality_model(), 0.2) == []


def test_quality_learning_recovers_informative_feature():
    rows = {}
    for i in range(30):
        value = -2 if i % 2 else 2
        c = candidate(np.ones((8, 8), bool), 0, feature=value)
        c.update(target=0.1 if value < 0 else 0.8, record_weight=1)
        rows[str(i)] = [c]
    fitted = core.fit_quality(rows)
    assert core.quality(rows["0"][0], fitted) > core.quality(rows["1"][0], fitted) + 0.5


def test_features_have_expected_schema_and_are_finite():
    mask = np.zeros((32, 32), bool)
    mask[:8, :8] = True
    values = core.mask_features(np.zeros((32, 32), np.uint8), mask, mask, 0.9, 0.95, "seed")
    assert len(values) == len(core.FEATURES)
    assert np.isfinite(values).all()
    with pytest.raises(ValueError, match="empty"):
        core.mask_features(np.zeros((32, 32)), mask, np.zeros_like(mask), 0.9, 0.95, "seed")


def test_real_folds_cover_train_once_and_purge_across_sites():
    import csv
    from datetime import datetime

    with (ROOT / "splits/split_manifest.csv").open() as stream:
        filenames = [r["filename"] for r in csv.DictReader(stream) if r["split"] == "train"]
    folds = core.make_folds(filenames)
    held = [name for fold in folds for name in fold["held"]]
    assert len(held) == len(set(held)) == 495
    assert set(held) == set(filenames)
    for fold in folds:
        train = set(fold["train"])
        assert not train & set(fold["held"])

        def days(names):
            return np.array(
                [datetime.strptime(n[:14], "%Y%m%d%H%M%S").date().toordinal() for n in names]
            )

        distances = abs(days(fold["train"])[:, None] - days(fold["held"])[None, :])
        assert distances.min() > 7


def test_official_and_one_to_one_agree_for_ordinary_disjoint_masks():
    first = np.zeros((8, 8), bool)
    first[:3] = True
    second = np.zeros_like(first)
    second[5:] = True
    stats = core.score([[rle(first), rle(second)]], [first, second])
    out = core.metrics(stats)
    assert out["pq"] == out["one_to_one_pq"] == 1


def test_bootstrap_keeps_annotator_statistics_together():
    names = ["20110101000000Ch.jpeg", "20110102000000Mh.jpeg", "20110501000000Ch.jpeg"]
    a = {n: [1, 1, 0, 0, 1, 1, 0, 0] for n in names}
    b = {n: [0.6, 1, 0, 0, 0.6, 1, 0, 0] for n in names}
    result = core.bootstrap_delta(a, b, names, replicates=50)
    assert result["blocks"] == 2
    assert result["ci95"] == pytest.approx([0.4, 0.4])


def test_packaged_legacy_contains_definitions_without_original_main_side_effects():
    import ast

    sys.path.insert(0, str(ROOT / "filament-oof-consensus"))
    try:
        runner_spec = importlib.util.spec_from_file_location(
            "consensus_runner", ROOT / "filament-oof-consensus/run.py"
        )
        runner = importlib.util.module_from_spec(runner_spec)
        runner_spec.loader.exec_module(runner)
        tree = ast.parse(runner.legacy_source())
        assert all(
            isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
            for n in tree.body
        )
        assert not any(getattr(n, "name", "") == "main" for n in tree.body)
        assert len(runner.frozen_split()["calibration"]) == 24
    finally:
        sys.path.pop(0)


def load_runner():
    sys.path.insert(0, str(ROOT / "filament-oof-consensus"))
    try:
        loaded = importlib.util.spec_from_file_location(
            "consensus_runner_test", ROOT / "filament-oof-consensus/run.py"
        )
        module = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_controller_uses_returned_slug_and_stops_on_failed_promotion(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RECORD", tmp_path / "record.json")
    monkeypatch.setattr(runner, "build_stage", lambda *_: tmp_path)
    calls = []

    def cli(*args):
        calls.append(args)
        if args[:2] == ("kernels", "push"):
            return "Kernel version 1 successfully pushed. https://www.kaggle.com/code/owner/actual"
        assert args == ("kernels", "status", "owner/actual")
        return "KernelWorkerStatus.COMPLETE"

    monkeypatch.setattr(runner, "cli", cli)
    monkeypatch.setattr(runner, "fetch", lambda *_: ({"full_refit_gate": False}, tmp_path))
    stage = {"status": "prepared", "name": "assess", "kernel": "owner/requested"}
    record = {"stages": [stage, {"name": "final", "status": "prepared"}]}
    runner.monitor(record)
    assert record["status"] == "gate_failed"
    assert stage["kernel"] == "owner/actual"
    assert sum(c[:2] == ("kernels", "push") for c in calls) == 1
    calls.clear()
    runner.monitor(record)
    assert calls == []


def test_manual_gate_override_is_final_only_and_records_remote_provenance():
    runner = load_runner()
    source = {
        "remote.py": '''if (
                not validation["full_refit_gate"]
                or validation["run_id"]
            ):
                raise ValueError
            result = final_inference(paths, models, cfg, validation)
''',
        "core.py": "core",
        "legacy_runtime.py": "legacy",
    }
    with pytest.raises(ValueError, match="only"):
        runner.execution_source(source, "assess", manual_gate_override=True)
    patched = runner.execution_source(source, "final", manual_gate_override=True)
    assert 'CONFIG.get("manual_gate_override", False)' in patched["remote.py"]
    assert "manual_gate_override=True" in patched["remote.py"]
    assert "execution_source_sha256" in patched["remote.py"]
    assert runner.execution_source(source, "final") == source


def test_controller_can_continue_after_failed_gate_when_explicitly_overridden(
    tmp_path, monkeypatch
):
    runner = load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RECORD", tmp_path / "record.json")
    monkeypatch.setattr(runner, "build_stage", lambda *_: tmp_path)
    monkeypatch.setattr(
        runner,
        "cli",
        lambda *args: (
            "Kernel version 1 successfully pushed. "
            "https://www.kaggle.com/code/owner/final-actual"
            if args[:2] == ("kernels", "push")
            else "KernelWorkerStatus.COMPLETE"
        ),
    )
    final_report = {"full_refit_gate": False, "status": "complete"}
    monkeypatch.setattr(runner, "fetch", lambda *_: (final_report, tmp_path))
    monkeypatch.setattr(runner, "verify_final", lambda *_: "submission.csv")
    record = {
        "manual_gate_override": {"authorized_at": "already-authorized"},
        "split": {"test": []},
        "stages": [
            {
                "name": "assess",
                "status": "complete",
                "report": {"full_refit_gate": False, "quality_sha256": "quality"},
            },
            {"name": "final", "status": "prepared", "kernel": "owner/requested"},
        ],
    }
    runner.monitor(record)
    assert record["status"] == "inference_ready"
    assert record["stages"][1]["status"] == "complete"


def test_controller_restores_override_provenance_after_interrupted_push(tmp_path, monkeypatch):
    runner = load_runner()
    runner.ROOT = tmp_path
    runner.RECORD = tmp_path / "record.json"
    config = {
        "run_id": "run-1",
        "source_sha256": "frozen-source",
        "execution_source_sha256": "override-source",
        "stage": "final",
        "manual_gate_override": True,
    }
    runner.write_json(tmp_path / "run/final/task-config.json", config)
    monkeypatch.setattr(runner, "cli", lambda *_: "KernelWorkerStatus.COMPLETE")
    monkeypatch.setattr(runner, "fetch", lambda *_: ({"status": "complete"}, tmp_path))
    monkeypatch.setattr(runner, "verify_final", lambda *_: "submission.csv")
    final = {"name": "final", "status": "pushing", "kernel": "owner/final"}
    record = {
        "run_id": "run-1",
        "source_sha256": "frozen-source",
        "directory": "run",
        "manual_gate_override": {"authorized_at": "already-authorized"},
        "stages": [
            {
                "name": "assess",
                "status": "complete",
                "report": {"full_refit_gate": False, "quality_sha256": "quality"},
            },
            final,
        ],
    }
    runner.monitor(record)
    assert final["execution_source_sha256"] == "override-source"
    assert final["manual_gate_override"] is True
    assert record["status"] == "inference_ready"


def test_submission_verifier_allows_zero_predictions_only_with_complete_count_manifest(tmp_path):
    import csv

    runner = load_runner()
    runner.ROOT = tmp_path
    path = tmp_path / "submission.csv"
    mask = np.zeros((2048, 2048), bool)
    mask[:10, :10] = True
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerow(["positive_1", core.encode(mask)["counts"]])
    detector = tmp_path / "detector.pt"
    refiner = tmp_path / "refiner.pt"
    detector.write_bytes(b"detector")
    refiner.write_bytes(b"refiner")
    report = {
        "submission_sha256": core.digest_bytes(path.read_bytes()),
        "counts": {"positive.jpeg": 1, "negative.jpeg": 0},
        "detector_sha256": core.digest_bytes(detector.read_bytes()),
        "refiner_sha256": core.digest_bytes(refiner.read_bytes()),
        "quality_sha256": "quality-hash",
    }
    assert (
        runner.verify_final(report, tmp_path, list(report["counts"]), "quality-hash")
        == "submission.csv"
    )
    with pytest.raises(ValueError, match="quality-model"):
        runner.verify_final(report, tmp_path, list(report["counts"]), "wrong-quality-hash")
    with pytest.raises(ValueError, match="manifest"):
        runner.verify_final(report, tmp_path, ["positive.jpeg"])
    report["submission_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="hash"):
        runner.verify_final(report, tmp_path, list(report["counts"]))
