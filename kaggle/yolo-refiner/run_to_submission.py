"""Monitor validation, launch the gated full refit, verify, and submit once."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

COMPETITION = "filament-segmentation-2026"
VALIDATION_KERNEL = "hariohmbhatt/sfsc-yolo-refiner-validation"
FINAL_KERNEL = "hariohmbhatt/sfsc-yolo-refiner-full"


def cli(*args: str) -> str:
    return subprocess.check_output(["kaggle", *args], text=True)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def kernel_status(kernel: str) -> str:
    return cli("kernels", "status", kernel).strip()


def wait_for_kernel(kernel: str, record: dict[str, object], record_path: Path) -> None:
    while True:
        status = kernel_status(kernel)
        record.update(remote_status=status, checked_at=datetime.now(UTC).isoformat())
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"kernel": kernel, "status": status}), flush=True)
        upper = status.upper()
        if "COMPLETE" in upper:
            return
        if any(word in upper for word in ["ERROR", "FAIL", "CANCEL"]):
            raise RuntimeError(f"Remote kernel failed: {status}")
        time.sleep(45)


def fetch_validation(root: Path) -> dict[str, object]:
    output = root / "outputs/yolo-refiner-validation"
    output.mkdir(parents=True, exist_ok=True)
    cli(
        "kernels", "output", VALIDATION_KERNEL, "--path", str(output),
        "--force", "--file-pattern",
        r".*(validation-report\.json|detector_runs/fold_0/results\.csv)$",
    )
    reports = list(output.rglob("validation-report.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one validation report, found {reports}")
    report = json.loads(reports[0].read_text())
    if report.get("status") != "complete":
        raise RuntimeError(f"Validation did not complete: {report}")
    return report


def stage_and_push_final(root: Path) -> None:
    source = root / "kaggle/yolo-refiner"
    stage = root / "outputs/yolo-refiner-full/staging"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "train_final.py", stage / "train_final.py")
    shutil.copy2(source / "kernel-final-metadata.json", stage / "kernel-metadata.json")
    result = cli("kernels", "push", "-p", str(stage))
    (stage / "push.log").write_text(result)
    print(result, flush=True)


def verify_submission(path: Path, expected_names: set[str]) -> int:
    grouped: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()
    identities = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Invalid submission header")
        for row in reader:
            identity = row["filament_id"]
            stem, separator, suffix = identity.rpartition("_")
            filename = stem + ".jpeg"
            if not separator or not suffix.isdecimal() or filename not in expected_names:
                raise ValueError(f"Invalid filament identity: {identity}")
            if identity in identities:
                raise ValueError(f"Duplicate filament identity: {identity}")
            identities.add(identity)
            counts[filename] += 1
            grouped.setdefault(filename, []).append(row["segmentation_rle"])
    # The competition represents an image with zero predicted instances by
    # omitting it from the instance-row CSV.  A previously accepted submission
    # in this project contains 179 of the 180 test IDs for exactly this reason.
    # Still reject an empty artifact and any unknown IDs (checked above).
    if not identities:
        raise ValueError("Refuse all-empty submission")
    if not set(counts) <= expected_names:
        raise ValueError("Submission contains an unknown test image")
    for filename, encodings in grouped.items():
        occupied = np.zeros((2048, 2048), dtype=bool)
        for encoding in encodings:
            mask = coco_mask.decode(
                {"size": [2048, 2048], "counts": encoding.encode("ascii")}
            ).astype(bool)
            if not mask.any() or np.any(mask & occupied):
                raise ValueError(f"Empty or overlapping instance in {filename}")
            occupied |= mask
    return len(identities)


def fetch_and_verify_final(
    root: Path, validation: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    output = root / "outputs/yolo-refiner-full/results"
    output.mkdir(parents=True, exist_ok=True)
    cli(
        "kernels", "output", FINAL_KERNEL, "--path", str(output), "--force",
        "--file-pattern",
        r".*(final-report\.json|submission\.csv|models/fold_0/(detector|refiner)\.pt)$",
    )
    report_paths = list(output.rglob("final-report.json"))
    submission_paths = list(output.rglob("submission.csv"))
    if len(report_paths) != 1 or len(submission_paths) != 1:
        raise RuntimeError("Final output is missing its report or submission")
    report = json.loads(report_paths[0].read_text())
    if report.get("status") != "complete" or report.get("train_images") != 601:
        raise RuntimeError(f"Final refit did not complete: {report}")
    if report.get("selected") != validation.get("selected"):
        raise RuntimeError("Final inference changed the selected validation recipe")
    expected = {
        path.name for path in (root / "SFSCDataset/test/test_images").glob("*.jpeg")
    }
    if len(expected) != 180:
        raise RuntimeError("Local competition test set does not contain 180 images")
    instances = verify_submission(submission_paths[0], expected)
    if instances != report.get("submission_rows"):
        raise RuntimeError("Verified submission row count differs from remote report")
    return submission_paths[0], report


def submit_once(
    path: Path,
    description: str,
    record: dict[str, object],
    record_path: Path,
) -> dict[str, str]:
    history = list(
        csv.DictReader(
            io.StringIO(cli("competitions", "submissions", "-c", COMPETITION, "-v"))
        )
    )
    existing = [row for row in history if row["description"] == description]
    if existing:
        return existing[0]
    if record.get("submission_attempted_at"):
        raise RuntimeError("A prior upload outcome is uncertain; refusing a duplicate retry")
    today = datetime.now(UTC).date().isoformat()
    if sum(row["date"].startswith(today) for row in history) >= 5:
        raise RuntimeError("Daily submission limit reached before the gated upload")
    record.update(status="submitting", submission_attempted_at=datetime.now(UTC).isoformat())
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(
        [
            "kaggle", "competitions", "submit", COMPETITION,
            "-f", str(path), "-m", description,
        ],
        check=True,
    )
    for _ in range(20):
        time.sleep(15)
        history = list(
            csv.DictReader(
                io.StringIO(
                    cli("competitions", "submissions", "-c", COMPETITION, "-v")
                )
            )
        )
        matching = [row for row in history if row["description"] == description]
        if matching and "COMPLETE" in matching[0].get("status", "").upper():
            return matching[0]
    return matching[0] if matching else {"status": "uploaded_score_pending"}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    record_path = root / "reports/yolo_refiner_run.json"
    record: dict[str, object] = {
        "status": "validation_running",
        "validation_kernel": VALIDATION_KERNEL,
        "final_kernel": FINAL_KERNEL,
        "submission_authorized": True,
        "started_at": datetime.now(UTC).isoformat(),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        wait_for_kernel(VALIDATION_KERNEL, record, record_path)
        validation = fetch_validation(root)
        record.update(validation=validation, status="validation_complete")
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if not validation.get("full_refit_gate"):
            record["status"] = "validation_gate_failed"
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            raise RuntimeError(
                f"Validation PQ {validation['full_validation']['pq']:.5f} did not clear "
                "the frozen +0.02 gate; no leaderboard attempt was spent"
            )
        try:
            existing_final_status = kernel_status(FINAL_KERNEL)
        except subprocess.CalledProcessError:
            existing_final_status = ""
        if not any(
            state in existing_final_status.upper() for state in ["RUNNING", "COMPLETE"]
        ):
            stage_and_push_final(root)
        else:
            print(f"Reusing existing final kernel: {existing_final_status}", flush=True)
        record["status"] = "full_refit_running"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        wait_for_kernel(FINAL_KERNEL, record, record_path)
        submission, final = fetch_and_verify_final(root, validation)
        artifact_hash = digest(submission)
        description = (
            f"yolo11l-refiner-601 pq{validation['full_validation']['pq']:.4f} "
            f"{validation['selected']['kind']} {artifact_hash[:8]}"
        )
        record.update(
            status="submission_verified",
            final=final,
            submission_file=str(submission.relative_to(root)),
            submission_sha256=artifact_hash,
            submission_description=description,
        )
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        result = submit_once(submission, description, record, record_path)
        record.update(
            status="leaderboard_submitted",
            submission=result,
            completed_at=datetime.now(UTC).isoformat(),
        )
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    except Exception as error:
        stopped_status = (
            "failed"
            if record.get("status") != "validation_gate_failed"
            else record["status"]
        )
        record.update(
            status=stopped_status,
            error_type=type(error).__name__,
            error=str(error),
            stopped_at=datetime.now(UTC).isoformat(),
        )
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
