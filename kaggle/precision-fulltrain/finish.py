"""Fetch, verify and optionally upload one explicitly authorized frozen submission."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from sfsc.system import digest

COMPETITION = "filament-segmentation-2026"


def cli(*args):
    return subprocess.check_output(["kaggle", *args], text=True)


def verify_csv(path, expected_counts):
    """Independently decode every submitted RLE and check instance identity/overlap."""
    counts, seen_ids = Counter(), set()
    grouped = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Invalid submission header")
        for row in reader:
            identity = row["filament_id"]
            stem, separator, suffix = identity.rpartition("_")
            filename = stem + ".jpeg"
            if not separator or not suffix.isdecimal() or filename not in expected_counts:
                raise ValueError(f"Invalid image identity: {identity}")
            if identity in seen_ids:
                raise ValueError("Duplicate instance ID")
            seen_ids.add(identity)
            grouped.setdefault(filename, []).append(row["segmentation_rle"])
            counts[filename] += 1
    for filename, expected in expected_counts.items():
        if counts[filename] != expected:
            raise ValueError(f"Instance count differs from verified report: {filename}")
        occupied = np.zeros((2048, 2048), dtype=bool)
        for rle in grouped.get(filename, []):
            mask = coco_mask.decode({"size": [2048, 2048], "counts": rle.encode("ascii")})
            if mask.shape != occupied.shape or not mask.any() or np.any(occupied & mask):
                raise ValueError(f"Empty, invalid or overlapping mask: {filename}")
            occupied |= mask.astype(bool)
    if not seen_ids:
        raise ValueError("Refuse all-empty submission")
    return len(seen_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record_path = root / "reports/full_training.json"
    record = json.loads(record_path.read_text())
    directory = root / record["directory"]

    def save():
        record_path.write_text(json.dumps(record, indent=2) + "\n")

    remote_status = cli("kernels", "status", record["kernel"]).strip()
    record.update(remote_status=remote_status, checked_at=datetime.now(UTC).isoformat())
    save()
    if "COMPLETE" not in remote_status.upper():
        print(remote_status)
        return
    if not record.get("artifacts_verified"):
        result = cli(
            "kernels",
            "output",
            record["kernel"],
            "--path",
            str(directory / "results"),
            "--force",
            "--file-pattern",
            r"experiment/(report\.json|submission-report\.json|submission\.csv|"
            r"checkpoint/(best\.weights\.h5|recipe\.json|best\.json))$",
        )
        (directory / "fetch.log").write_text(result)
        output = directory / "results/experiment"
        report = json.loads((output / "report.json").read_text())
        details = json.loads((output / "submission-report.json").read_text())
        recipe = json.loads((output / "checkpoint/recipe.json").read_text())
        frozen = json.loads(
            (
                root
                / "outputs/precision-v2-20260906-052615/results/experiment/checkpoint/recipe.json"
            ).read_text()
        )
        if report["status"] != "completed" or details["status"] != "completed":
            raise ValueError("Remote training/submission generation did not complete")
        if report["run_id"] != record["run_id"] or report["spec_hash"] != record["spec_hash"]:
            raise ValueError("Run or immutable source fingerprint mismatch")
        if details["submission_sha256"] != digest(output / "submission.csv"):
            raise ValueError("Submission hash mismatch")
        if details["weights_sha256"] != digest(output / "checkpoint/best.weights.h5"):
            raise ValueError("Full-training weights hash mismatch")
        if recipe["config"] != frozen["config"] or recipe["postprocess"] != frozen["postprocess"]:
            raise ValueError("Frozen precision recipe changed")
        if recipe["best_epoch"] != frozen["best_epoch"] or details["train_images"] != 601:
            raise ValueError("Frozen training duration or training scope changed")
        expected_names = {p.name for p in (root / "SFSCDataset/test/test_images").glob("*.jpeg")}
        if set(details["counts"]) != expected_names or len(expected_names) != 180:
            raise ValueError("Competition image coverage differs")
        instances = verify_csv(output / "submission.csv", details["counts"])
        if instances != details["instances"]:
            raise ValueError("Total instance count differs")
        record.update(
            status="submission_ready",
            artifacts_verified=True,
            instances=instances,
            submission_sha256=details["submission_sha256"],
            submission_file=str((output / "submission.csv").relative_to(root)),
            peak_vram_gb=report["peak_vram_gb"],
            elapsed_seconds=report["elapsed_seconds"],
        )
        save()
    if args.submit:
        if not record.get("submission_authorized"):
            raise ValueError("No recorded user authorization for leaderboard submission")
        description = (
            f"{record['run_id']} 601img baseline60+precision15 frozen fg.55 conf.75 area256"
        )
        history = list(
            csv.DictReader(io.StringIO(cli("competitions", "submissions", "-c", COMPETITION, "-v")))
        )
        existing = [row for row in history if row["description"] == description]
        if existing:
            record.update(submission=existing[0], status="leaderboard_submitted")
            save()
        elif record.get("submission_attempted_at"):
            raise ValueError("Previous upload outcome uncertain; inspect history before any retry")
        else:
            today = datetime.now(UTC).date().isoformat()
            if sum(row["date"].startswith(today) for row in history) >= 5:
                raise ValueError("Five submissions already used today")
            path = root / record["submission_file"]
            if digest(path) != record["submission_sha256"]:
                raise ValueError("Submission file changed after verification")
            record.update(
                status="submitting", submission_attempted_at=datetime.now(UTC).isoformat()
            )
            save()
            result = subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "submit",
                    COMPETITION,
                    "-f",
                    str(path),
                    "-m",
                    description,
                ],
                capture_output=True,
                text=True,
            )
            (directory / "submission-upload.log").write_text(result.stdout + result.stderr)
            record.update(
                status="leaderboard_submitted"
                if result.returncode == 0
                else "submission_uncertain",
                submission_returncode=result.returncode,
                submission_description=description,
            )
            save()
            print(result.stdout + result.stderr)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
