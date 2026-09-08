"""Freeze, launch, and monitor the staged Kaggle consensus training pipeline."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import fcntl
import json
import re
import subprocess
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import core

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RECORD = ROOT / "reports/consensus_training.json"
OWNER = "hariohmbhatt"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def cli(*args):
    return subprocess.check_output(
        ["kaggle", *args], text=True, stderr=subprocess.STDOUT, timeout=300
    )


def legacy_source():
    source = (ROOT / "kaggle/yolo-refiner/train_validation.py").read_text()
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
        and getattr(node, "name", "") != "main"
    ]
    return ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n"


def frozen_split():
    tree = ast.parse((ROOT / "kaggle/yolo-refiner/train_validation.py").read_text())
    encoded = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SPLIT_B64" for t in node.targets)
    )
    return json.loads(zlib.decompress(base64.b64decode(encoded)))


def prepare():
    if RECORD.exists():
        previous = json.loads(RECORD.read_text())
        if previous["status"] not in ("inference_ready", "gate_failed", "cancelled"):
            raise RuntimeError("An existing run must be resumed; refusing duplicate training")
    split = frozen_split()
    if len(split["train"]) != 495 or len(split["validation"]) != 106:
        raise ValueError("Unexpected development split")
    folds = core.make_folds(split["train"])
    source = {
        "core.py": (HERE / "core.py").read_text(),
        "remote.py": (HERE / "remote.py").read_text(),
        "legacy_runtime.py": legacy_source(),
    }
    source_hash = core.digest_bytes(json.dumps(source, sort_keys=True).encode())
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    home = ROOT / "outputs" / f"consensus-{run_id}"
    home.mkdir(parents=True)
    for name, value in source.items():
        (home / name).write_text(value)
    baseline = (
        ROOT / "outputs/yolo-refiner-validation/sfsc-yolo-refiner-validation/"
        "validation-candidates.json"
    )
    record = {
        "status": "prepared",
        "run_id": run_id,
        "directory": str(home.relative_to(ROOT)),
        "source_sha256": source_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_candidates_sha256": core.digest_bytes(baseline.read_bytes()),
        "split": split,
        "folds": folds,
        "stages": [],
        "automatic_submission": False,
    }
    names = [f"fold-{i}" for i in range(3)] + ["assess", "final"]
    for stage in names:
        short = run_id
        kernel = f"{OWNER}/sfsc-c3-{short}-{stage}"
        record["stages"].append({"name": stage, "kernel": kernel, "status": "prepared"})
    write_json(home / "manifest.json", record)
    write_json(RECORD, record)
    for fold in folds:
        print(
            f"fold {fold['fold']}: train={len(fold['train'])}, "
            f"held={len(fold['held'])}, purged={len(fold['purged'])}"
        )
    return record


def execution_source(source, stage_name, manual_gate_override=False):
    """Return the exact source executed by a stage, preserving override provenance."""
    payload = dict(source)
    if not manual_gate_override:
        return payload
    if stage_name != "final":
        raise ValueError("The promotion gate can only be overridden for the final stage")
    original = 'not validation["full_refit_gate"]\n                or validation["run_id"]'
    replacement = (
        '(not validation["full_refit_gate"] and not '
        'CONFIG.get("manual_gate_override", False))\n'
        '                or validation["run_id"]'
    )
    if 'CONFIG.get("manual_gate_override", False)' not in payload["remote.py"]:
        if payload["remote.py"].count(original) != 1:
            raise ValueError("Could not apply the frozen final-gate override")
        payload["remote.py"] = payload["remote.py"].replace(original, replacement)
    result_line = "            result = final_inference(paths, models, cfg, validation)\n"
    result_provenance = result_line + (
        "            result.update(\n"
        "                manual_gate_override=True,\n"
        "                assessment_full_refit_gate=validation[\"full_refit_gate\"],\n"
        "                execution_source_sha256=CONFIG[\"execution_source_sha256\"],\n"
        "            )\n"
    )
    if "assessment_full_refit_gate" not in payload["remote.py"]:
        if payload["remote.py"].count(result_line) != 1:
            raise ValueError("Could not add final-gate override provenance")
        payload["remote.py"] = payload["remote.py"].replace(result_line, result_provenance)
    return payload


def build_stage(record, stage, manual_gate_override=False):
    home = ROOT / record["directory"]
    source = {
        name: (home / name).read_text() for name in ("core.py", "remote.py", "legacy_runtime.py")
    }
    if core.digest_bytes(json.dumps(source, sort_keys=True).encode()) != record["source_sha256"]:
        raise ValueError("Frozen source changed")
    source = execution_source(source, stage["name"], manual_gate_override)
    config = {
        "run_id": record["run_id"],
        "stage": stage["name"],
        "source_sha256": record["source_sha256"],
        "execution_source_sha256": core.digest_bytes(
            json.dumps(source, sort_keys=True).encode()
        ),
        "manual_gate_override": manual_gate_override,
    }
    sources = []
    if stage["name"].startswith("fold-"):
        fold = record["folds"][int(stage["name"].split("-")[-1])]
        config.update(train_names=fold["train"], held_names=fold["held"])
    elif stage["name"] == "assess":
        config.update(
            train_names=record["split"]["train"],
            validation_names=record["split"]["validation"],
            calibration_names=record["split"]["calibration"],
            folds=record["folds"],
            baseline_candidates_sha256=record["baseline_candidates_sha256"],
        )
        sources = [s["kernel"] for s in record["stages"][:3]]
        sources.append(f"{OWNER}/sfsc-yolo-refiner-validation")
    else:
        config.update(train_names=sorted(record["split"]["train"] + record["split"]["validation"]))
        sources = [record["stages"][3]["kernel"]]
    stage_dir = home / stage["name"]
    stage_dir.mkdir(exist_ok=True)
    payload = dict(source)
    payload["task-config.json"] = json.dumps(config)
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    bootstrap = """import base64, json, os, runpy, subprocess, sys, zlib
from pathlib import Path
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
    '--force-reinstall', 'torch==2.7.1', 'torchvision==0.22.1',
    '--index-url', 'https://download.pytorch.org/whl/cu126'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
    'ultralytics==8.4.103', 'pycocotools==2.0.11'])
payload = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
root = Path('/kaggle/working')
for name, source in payload.items():
    (root / name).write_text(source)
sys.path.insert(0, str(root))
os.chdir(root)
runpy.run_path(str(root / 'remote.py'), run_name='__main__')
""".replace("PAYLOAD", repr(encoded))
    (stage_dir / "train.py").write_text(bootstrap)
    metadata = {
        "id": stage["kernel"],
        "title": f"SFSC C3 {record['run_id']} {stage['name']}",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaP100",
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": sources,
        "model_sources": [],
    }
    write_json(stage_dir / "kernel-metadata.json", metadata)
    write_json(stage_dir / "task-config.json", config)
    stage["code_sha256"] = core.digest_bytes(bootstrap.encode())
    stage["execution_source_sha256"] = config["execution_source_sha256"]
    stage["manual_gate_override"] = manual_gate_override
    return stage_dir


def fetch(record, stage):
    home = ROOT / record["directory"] / stage["name"] / "results"
    home.mkdir(parents=True, exist_ok=True)
    file_pattern = r".*(report\.json|quality\.json|selection\.json|submission\.csv)$"
    if stage["name"] == "final":
        file_pattern = (
            r".*(report\.json|quality\.json|selection\.json|submission\.csv|"
            r"detector\.pt|refiner\.pt)$"
        )
    output = cli(
        "kernels",
        "output",
        stage["kernel"],
        "-p",
        str(home),
        "--force",
        "--file-pattern",
        file_pattern,
    )
    (home.parent / "fetch.log").write_text(output)
    report_name = (
        stage["name"] + "-report.json"
        if stage["name"].startswith("fold-")
        else "assessment-report.json"
        if stage["name"] == "assess"
        else "final-report.json"
    )
    files = list(home.rglob(report_name))
    if len(files) != 1:
        raise ValueError(f"Missing unique stage report {report_name}")
    report = json.loads(files[0].read_text())
    if (
        report.get("status") != "complete"
        or report["run_id"] != record["run_id"]
        or report["source_sha256"] != record["source_sha256"]
    ):
        raise ValueError("Stage completion/source lineage mismatch")
    if (
        stage["name"] == "final"
        and stage.get("manual_gate_override")
        and (
            report.get("manual_gate_override") is not True
            or report.get("assessment_full_refit_gate") is not False
            or report.get("execution_source_sha256")
            != stage.get("execution_source_sha256")
        )
    ):
        raise ValueError("Final override provenance mismatch")
    if stage["name"] == "assess":
        quality_files = list(home.rglob("quality.json"))
        if (
            len(quality_files) != 1
            or core.digest_bytes(quality_files[0].read_bytes()) != report["quality_sha256"]
        ):
            raise ValueError("Assessment quality-model hash mismatch")
    if stage["name"].startswith("fold-"):
        fold = record["folds"][int(stage["name"].split("-")[-1])]
        if report["train_names"] != fold["train"] or report["held_names"] != fold["held"]:
            raise ValueError("Fold report membership mismatch")
    return report, home


def verify_final(report, home, expected, expected_quality_sha256=None):
    files = list(home.rglob("submission.csv"))
    if len(files) != 1 or core.digest_bytes(files[0].read_bytes()) != report["submission_sha256"]:
        raise ValueError("Submission hash mismatch")
    if set(report["counts"]) != set(expected):
        raise ValueError("Missing inference image manifest")
    if (
        expected_quality_sha256 is not None
        and report.get("quality_sha256") != expected_quality_sha256
    ):
        raise ValueError("Final quality-model lineage mismatch")
    counts = dict.fromkeys(expected, 0)
    occupied, seen = {}, set()
    with files[0].open() as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Invalid CSV columns")
        for row in reader:
            identity = row["filament_id"]
            stem, separator, number = identity.rpartition("_")
            name = stem + ".jpeg"
            if not separator or not number.isdecimal() or name not in counts or identity in seen:
                raise ValueError("Invalid or duplicate instance ID")
            seen.add(identity)
            mask = core.decode({"size": [2048, 2048], "counts": row["segmentation_rle"]})
            if not mask.any():
                raise ValueError("Empty instance")
            if name in occupied and (occupied[name] & mask).any():
                raise ValueError("Overlapping instances")
            occupied[name] = mask if name not in occupied else occupied[name] | mask
            counts[name] += 1
    if counts != report["counts"] or not seen:
        raise ValueError("Submission count mismatch")
    for filename, field in (
        ("detector.pt", "detector_sha256"),
        ("refiner.pt", "refiner_sha256"),
    ):
        weights = list(home.rglob(filename))
        if len(weights) != 1 or core.digest_bytes(weights[0].read_bytes()) != report[field]:
            raise ValueError(f"{filename} hash mismatch")
    return str(files[0].relative_to(ROOT))


def monitor(record, manual_gate_override=False):
    manual_gate_override = manual_gate_override or bool(record.get("manual_gate_override"))
    if manual_gate_override:
        final = next(s for s in record["stages"] if s["name"] == "final")
        if final["status"] != "prepared" and "execution_source_sha256" not in final:
            config_path = ROOT / record["directory"] / "final/task-config.json"
            config = json.loads(config_path.read_text())
            if (
                config.get("run_id") != record["run_id"]
                or config.get("source_sha256") != record["source_sha256"]
                or config.get("stage") != "final"
                or config.get("manual_gate_override") is not True
            ):
                raise ValueError("Cannot restore final override provenance")
            final["execution_source_sha256"] = config["execution_source_sha256"]
            final["manual_gate_override"] = True
            write_json(RECORD, record)
    for stage in record["stages"]:
        if stage["status"] == "complete":
            if (
                stage["name"] == "assess"
                and not stage["report"]["full_refit_gate"]
                and not manual_gate_override
            ):
                record["status"] = "gate_failed"
                write_json(RECORD, record)
                return
            continue
        if stage["status"] == "prepared":
            directory = build_stage(record, stage, manual_gate_override)
            stage["status"] = "pushing"
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
                raise RuntimeError("Push outcome lacks a kernel URL; inspect push.log")
            stage["kernel"] = urls[-1]
            stage["status"] = "running"
            write_json(RECORD, record)
        # A previous ambiguous push is checked by ID, never blindly uploaded again.
        while True:
            status = cli("kernels", "status", stage["kernel"]).strip()
            record.update(
                status="running",
                active_stage=stage["name"],
                checked_at=datetime.now(UTC).isoformat(),
                remote_status=status,
            )
            write_json(RECORD, record)
            print(json.dumps({"stage": stage["name"], "remote_status": status}), flush=True)
            if "COMPLETE" in status.upper():
                break
            if any(word in status.upper() for word in ("ERROR", "FAIL", "CANCEL")):
                raise RuntimeError(status)
            time.sleep(45)
        report, home = fetch(record, stage)
        stage.update(status="complete", report=report)
        write_json(RECORD, record)
        if (
            stage["name"] == "assess"
            and not report["full_refit_gate"]
            and not manual_gate_override
        ):
            record["status"] = "gate_failed"
            write_json(RECORD, record)
            return
        if stage["name"] == "final":
            expected = sorted(
                p.name for p in (ROOT / "SFSCDataset/test/test_images").glob("*.jpeg")
            )
            assessment = next(s for s in record["stages"] if s["name"] == "assess")
            record["submission_file"] = verify_final(
                report,
                home,
                expected,
                assessment["report"]["quality_sha256"],
            )
    record["status"] = "inference_ready"
    record["completed_at"] = datetime.now(UTC).isoformat()
    write_json(RECORD, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "status"])
    parser.add_argument(
        "--override-gate",
        action="store_true",
        help="run the final fit after a failed assessment, preserving override provenance",
    )
    args = parser.parse_args()
    if args.override_gate and args.command != "run":
        parser.error("--override-gate is only valid with the run command")
    if args.command == "prepare":
        prepare()
    elif args.command == "status":
        record = json.loads(RECORD.read_text())
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in (
                        "status",
                        "run_id",
                        "active_stage",
                        "checked_at",
                        "remote_status",
                        "error",
                    )
                },
                indent=2,
            )
        )
    else:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        lock = RECORD.with_suffix(".lock").open("w")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A consensus controller is already running") from error
        record = json.loads(RECORD.read_text()) if RECORD.exists() else prepare()
        if args.override_gate:
            assessment = next(s for s in record["stages"] if s["name"] == "assess")
            if assessment["status"] != "complete":
                raise RuntimeError("The assessment must complete before overriding its gate")
            if assessment["report"]["full_refit_gate"]:
                raise RuntimeError("The assessment passed; a gate override is unnecessary")
            if "manual_gate_override" not in record:
                record["manual_gate_override"] = {
                    "authorized_at": datetime.now(UTC).isoformat(),
                    "assessment_full_refit_gate": assessment["report"]["full_refit_gate"],
                    "assessment82_pq": assessment["report"]["assessment82"]["pq"],
                }
            record.pop("error", None)
            record.pop("stopped_at", None)
            write_json(RECORD, record)
        try:
            monitor(record, manual_gate_override=args.override_gate)
        except Exception as error:
            record.update(
                status="failed", error=str(error), stopped_at=datetime.now(UTC).isoformat()
            )
            write_json(RECORD, record)
            raise


if __name__ == "__main__":
    main()
