"""Freeze, launch, monitor, and verify the native-reconstruction pilot."""

from __future__ import annotations

import argparse
import ast
import base64
import fcntl
import hashlib
import json
import re
import subprocess
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RECORD = ROOT / "reports/native_training.json"
OWNER = "hariohmbhatt"
UPSTREAM_RUN = ROOT / "outputs/consensus-20260907-125349"
VALIDATION = (
    ROOT / "outputs/yolo-refiner-validation/sfsc-yolo-refiner-validation/validation-candidates.json"
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cli(*args: str) -> str:
    return subprocess.check_output(
        ["kaggle", *args], text=True, stderr=subprocess.STDOUT, timeout=300
    )


def legacy_source() -> str:
    """Extract definitions without executing the legacy install or main blocks."""
    source = (ROOT / "kaggle/yolo-refiner/train_validation.py").read_text()
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
        and getattr(node, "name", "") != "main"
    ]
    return ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n"


def frozen_split() -> dict:
    tree = ast.parse((ROOT / "kaggle/yolo-refiner/train_validation.py").read_text())
    encoded = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SPLIT_B64" for target in node.targets
        )
    )
    return json.loads(zlib.decompress(base64.b64decode(encoded)))


def upstream_candidates() -> tuple[list[str], list[str]]:
    hashes, kernels = [], []
    for fold in range(3):
        report_path = next(
            (UPSTREAM_RUN / f"fold-{fold}/results").rglob(f"fold-{fold}-report.json")
        )
        report = json.loads(report_path.read_text())
        candidate_path = next(
            (UPSTREAM_RUN / f"fold-{fold}/results").rglob(f"fold-{fold}-candidates.json")
        )
        if digest(candidate_path) != report["candidates_sha256"]:
            raise ValueError(f"Fold {fold} candidate hash mismatch")
        hashes.append(report["candidates_sha256"])
        kernels.append(f"{OWNER}/sfsc-c3-20260907-125349-fold-{fold}")
    return hashes, kernels


def prepare() -> dict:
    if RECORD.exists():
        previous = json.loads(RECORD.read_text())
        if previous["status"] not in ("complete", "failed", "cancelled"):
            raise RuntimeError("An existing native pilot must be resumed")
    split = frozen_split()
    if len(split["train"]) != 495 or len(split["validation"]) != 106:
        raise ValueError("Unexpected frozen development split")
    hashes, kernels = upstream_candidates()
    source = {
        "core.py": (HERE / "core.py").read_text(),
        "remote.py": (HERE / "remote.py").read_text(),
        "metric_core.py": (ROOT / "filament-oof-consensus/core.py").read_text(),
        "legacy_runtime.py": legacy_source(),
    }
    source_sha256 = digest_bytes(json.dumps(source, sort_keys=True).encode())
    membership = {
        "train": split["train"],
        "validation": split["validation"],
        "calibration": split["calibration"],
    }
    membership_sha256 = digest_bytes(json.dumps(membership, sort_keys=True).encode())
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    directory = ROOT / "outputs" / f"native-pilot-{run_id}"
    directory.mkdir(parents=True)
    for name, value in source.items():
        (directory / name).write_text(value)
    config = {
        "run_id": run_id,
        "source_sha256": source_sha256,
        "membership_sha256": membership_sha256,
        "train_names": split["train"],
        "validation_names": split["validation"],
        "calibration_names": split["calibration"],
        "oof_candidate_sha256": hashes,
        "validation_candidates_sha256": digest(VALIDATION),
        "expected_control_pq": 0.4351790286733054,
        "seed": 20260908,
        "batch_size": 2,
        "accumulation": 4,
        "optimizer_updates": 4000,
        "warmup_updates": 200,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "gradient_clip": 2.0,
        "amp": False,
        "inference_batch": 2,
        "mask_threshold_grid": [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70],
        "mask_threshold_default": 0.50,
        "score_threshold": 0.30,
        "minimum_area": 64,
    }
    record = {
        "status": "prepared",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "directory": str(directory.relative_to(ROOT)),
        "kernel": f"{OWNER}/sfsc-native-{run_id}",
        "source_sha256": source_sha256,
        "membership_sha256": membership_sha256,
        "upstream_kernels": kernels,
        "config": config,
        "automatic_submission": False,
    }
    write_json(directory / "manifest.json", record)
    write_json(RECORD, record)
    return record


def build_kernel(record: dict) -> Path:
    directory = ROOT / record["directory"]
    source = {
        name: (directory / name).read_text()
        for name in ("core.py", "remote.py", "metric_core.py", "legacy_runtime.py")
    }
    if digest_bytes(json.dumps(source, sort_keys=True).encode()) != record["source_sha256"]:
        raise ValueError("Frozen source changed")
    payload = {**source, "task-config.json": json.dumps(record["config"])}
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    bootstrap = """import base64, json, os, runpy, subprocess, sys, zlib
from pathlib import Path
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
    '--force-reinstall', 'torch==2.7.1', 'torchvision==0.22.1',
    '--index-url', 'https://download.pytorch.org/whl/cu126'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
    'pycocotools==2.0.11'])
payload = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
root = Path('/kaggle/working')
for name, source in payload.items():
    (root / name).write_text(source)
sys.path.insert(0, str(root))
os.chdir(root)
runpy.run_path(str(root / 'remote.py'), run_name='__main__')
""".replace("PAYLOAD", repr(encoded))
    (directory / "train.py").write_text(bootstrap)
    metadata = {
        "id": record["kernel"],
        "title": f"SFSC native pilot {record['run_id']}",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaP100",
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": [
            *record["upstream_kernels"],
            f"{OWNER}/sfsc-yolo-refiner-validation",
        ],
        "model_sources": [],
    }
    write_json(directory / "kernel-metadata.json", metadata)
    write_json(directory / "task-config.json", record["config"])
    record["code_sha256"] = digest_bytes(bootstrap.encode())
    return directory


def fetch_and_verify(record: dict) -> dict:
    destination = ROOT / record["directory"] / "results"
    destination.mkdir(parents=True, exist_ok=True)
    output = cli(
        "kernels",
        "output",
        record["kernel"],
        "-p",
        str(destination),
        "--force",
        "--file-pattern",
        r".*(pilot-report\.json|native-refiner\.pt|native-validation-candidates\.json|training-samples\.json)$",
    )
    (destination.parent / "fetch.log").write_text(output)
    reports = list(destination.rglob("pilot-report.json"))
    if len(reports) != 1:
        raise ValueError("Missing unique pilot report")
    report = json.loads(reports[0].read_text())
    if (
        report.get("status") != "complete"
        or report.get("run_id") != record["run_id"]
        or report.get("source_sha256") != record["source_sha256"]
        or report.get("membership_sha256") != record["membership_sha256"]
    ):
        raise ValueError("Pilot report lineage mismatch")
    expected = {
        "native-refiner.pt": report["model_sha256"],
        "native-validation-candidates.json": report["assessment"]["candidate_cache_sha256"],
        "training-samples.json": report["sample_manifest_sha256"],
    }
    for filename, expected_hash in expected.items():
        files = list(destination.rglob(filename))
        if len(files) != 1 or digest(files[0]) != expected_hash:
            raise ValueError(f"Artifact verification failed for {filename}")
    training = report["training"]
    assessment = report["assessment"]
    if training.get("nonfinite_batches") != 0:
        raise ValueError("Remote training reported non-finite batches")
    if assessment.get("nonempty_candidates", 0) == 0:
        raise ValueError("Remote reconstruction produced no non-empty masks")
    return report


def monitor(record: dict) -> None:
    if record["status"] == "prepared":
        directory = build_kernel(record)
        record["status"] = "pushing"
        write_json(RECORD, record)
        output = cli(
            "kernels",
            "push",
            "-p",
            str(directory),
            "--accelerator",
            "NvidiaTeslaP100",
            "--timeout",
            "32400",
        )
        (directory / "push.log").write_text(output)
        urls = re.findall(r"https://www\.kaggle\.com/code/([\w-]+/[\w-]+)", output)
        if not urls:
            raise RuntimeError("Push result did not contain a kernel URL")
        record["kernel"] = urls[-1]
        record["status"] = "running"
        record["started_at"] = datetime.now(UTC).isoformat()
        write_json(RECORD, record)
    while True:
        status = cli("kernels", "status", record["kernel"]).strip()
        record.update(
            status="running",
            remote_status=status,
            checked_at=datetime.now(UTC).isoformat(),
        )
        write_json(RECORD, record)
        print(json.dumps({"kernel": record["kernel"], "remote_status": status}), flush=True)
        upper = status.upper()
        if "COMPLETE" in upper:
            break
        if any(word in upper for word in ("ERROR", "FAIL", "CANCEL")):
            raise RuntimeError(status)
        time.sleep(45)
    report = fetch_and_verify(record)
    record.update(
        status="complete",
        completed_at=datetime.now(UTC).isoformat(),
        report=report,
    )
    write_json(RECORD, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "status"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    if args.command == "status":
        print(RECORD.read_text())
        return
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    lock = RECORD.with_suffix(".lock").open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A native-pilot controller is already running") from error
    record = json.loads(RECORD.read_text()) if RECORD.exists() else prepare()
    try:
        monitor(record)
    except Exception as error:
        record.update(status="failed", error=str(error), stopped_at=datetime.now(UTC).isoformat())
        write_json(RECORD, record)
        raise


if __name__ == "__main__":
    main()
