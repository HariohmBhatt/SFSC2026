import ast
import base64
import csv
import importlib.util
import json
import zlib
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]


def load_submission_verifier():
    path = ROOT / "kaggle/yolo-refiner/run_to_submission.py"
    spec = importlib.util.spec_from_file_location("yolo_refiner_submit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_submission


def write_submission(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)


def embedded_split(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SPLIT_B64"
            for target in node.targets
        ):
            encoded = ast.literal_eval(node.value)
            return json.loads(zlib.decompress(base64.b64decode(encoded)))
    raise AssertionError("SPLIT_B64 was not found")


def load_remote_functions(*names: str) -> dict[str, object]:
    source = (ROOT / "kaggle/yolo-refiner/train_validation.py").read_text()
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "np": np,
        "mask_utils": mask_utils,
        "linear_sum_assignment": linear_sum_assignment,
        "Sequence": list,
    }
    exec(compile(ast.Module(selected, type_ignores=[]), "remote-functions", "exec"), namespace)
    return namespace


def test_validation_job_embeds_immutable_development_split() -> None:
    payload = embedded_split(ROOT / "kaggle/yolo-refiner/train_validation.py")

    assert len(payload["train"]) == 495
    assert len(payload["validation"]) == 106
    assert len(payload["calibration"]) == 24
    assert set(payload["train"]).isdisjoint(payload["validation"])
    assert set(payload["calibration"]) <= set(payload["validation"])


def test_final_job_uses_the_same_601_images() -> None:
    validation = embedded_split(ROOT / "kaggle/yolo-refiner/train_validation.py")
    final = embedded_split(ROOT / "kaggle/yolo-refiner/train_final.py")

    assert final == validation
    assert len(set(final["train"]) | set(final["validation"])) == 601


def test_training_jobs_use_strict_official_iou_boundary_and_gate() -> None:
    validation = (ROOT / "kaggle/yolo-refiner/train_validation.py").read_text()
    final = (ROOT / "kaggle/yolo-refiner/train_final.py").read_text()

    assert "matched = iou > match_iou" in validation
    assert "full_metrics[\"one_to_one_pq\"]" in validation
    assert ">= 0.40117396647989144" in validation
    assert "not report.get(\"full_refit_gate\")" in final


def test_remote_metric_reports_official_and_one_to_one_counts() -> None:
    functions = load_remote_functions("pairwise_iou", "panoptic_quality")
    score = functions["panoptic_quality"](
        [np.ones((4, 4), bool), np.ones((4, 4), bool)],
        [np.ones((4, 4), bool)],
    )

    assert score["tp"] == 2
    assert score["one_to_one_tp"] == 1
    assert score["sum_iou"] == 2.0
    assert score["one_to_one_sum_iou"] == 1.0


def test_refiner_selection_is_calibration_only_and_final_has_no_forced_mask() -> None:
    validation = (ROOT / "kaggle/yolo-refiner/train_validation.py").read_text()
    final = (ROOT / "kaggle/yolo-refiner/train_final.py").read_text()
    final_tree = ast.parse(final)
    choose = next(
        node
        for node in final_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "choose_test_masks"
    )
    choose_source = ast.get_source_segment(final, choose)

    assert "if filename in CALIBRATION_FILES" in validation
    assert "choose_refiner_checkpoint" in validation
    assert "choose_detector_checkpoint" in validation
    assert "classical_fallback_seed" not in choose_source


def test_final_kernel_consumes_the_validation_artifact() -> None:
    metadata = json.loads(
        (ROOT / "kaggle/yolo-refiner/kernel-final-metadata.json").read_text()
    )

    assert metadata["machine_shape"] == "NvidiaTeslaP100"
    assert metadata["kernel_sources"] == [
        "hariohmbhatt/sfsc-yolo-refiner-validation"
    ]


def test_submission_verifier_allows_images_with_zero_predicted_instances(tmp_path) -> None:
    mask = np.zeros((2048, 2048), np.uint8, order="F")
    mask[10:20, 10:20] = 1
    rle = mask_utils.encode(mask)["counts"].decode()
    submission = tmp_path / "submission.csv"
    write_submission(submission, [("predicted_1", rle)])

    assert load_submission_verifier()(
        submission, {"predicted.jpeg", "empty.jpeg"}
    ) == 1


def test_submission_verifier_rejects_all_empty_artifact(tmp_path) -> None:
    submission = tmp_path / "submission.csv"
    write_submission(submission, [])

    try:
        load_submission_verifier()(submission, {"empty.jpeg"})
    except ValueError as error:
        assert "all-empty" in str(error)
    else:
        raise AssertionError("all-empty submission should be rejected")
