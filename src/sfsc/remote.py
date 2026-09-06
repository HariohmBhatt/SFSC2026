"""Build self-contained Kaggle kernels and ingest verified stage artifacts."""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import uuid
import zipfile
from pathlib import Path

from sfsc.contracts import number, require
from sfsc.kaggle_runtime import artifact_hash, child
from sfsc.system import System, digest, now, object_hash, read_json, write_json

PHASES = {
    "evaluate": "implemented",
    "internal_test": "frozen",
    "full_train": "tested",
    "submit": "trained",
}
NEXT = {
    "evaluate": "evaluated",
    "internal_test": "tested",
    "full_train": "trained",
    "submit": "submission_ready",
}


def prepare(system: System, cycle_id: str, phase: str, owner: str) -> dict:
    require(phase in PHASES, "Unknown execution phase")
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", owner) is not None, "Invalid Kaggle owner")
    with system.transaction() as state:
        system.check(state)
        require(cycle_id in state["cycles"], "Unknown cycle")
        cycle = state["cycles"][cycle_id]
        require(cycle["status"] == PHASES[phase], f"{phase} requires {PHASES[phase]} state")
        require(
            not any(
                r["cycle_id"] == cycle_id and r["status"] in {"prepared", "launching", "submitted"}
                for r in state["runs"].values()
            ),
            "Resolve the existing run before preparing another",
        )
        if phase == "internal_test":
            require(not cycle.get("holdout_run"), "Internal test has already been reserved once")
        system.check_implementation(cycle)
        run_id = f"{cycle_id}-{phase.replace('_', '-')}-{uuid.uuid4().hex[:8]}"
        kernel = f"{owner}/sfsc-{run_id}"
        directory = system.home / "runs" / run_id
        directory.mkdir(parents=True)
        upstream = None
        if phase in {"internal_test", "full_train", "submit"}:
            previous = state["runs"][
                cycle["training_run" if phase == "submit" else "evaluation_run"]
            ]
            upstream = {
                "run_id": previous["run_id"],
                "kernel": previous["kernel"],
                "checkpoint_hash": previous["report"]["checkpoint_hash"],
            }
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=system.root, capture_output=True, text=True
        )
        annotation = system.root / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        require(annotation.is_file(), "Local annotations are required to bind the remote dataset")
        provenance = {
            "commit": git.stdout.strip() if git.returncode == 0 else None,
            "contract": state["contract"],
            "implementation": cycle["implementation_hashes"],
            "annotation_hash": digest(annotation),
        }
        config = state["config"]
        spec = {
            "run_id": run_id,
            "cycle_id": cycle_id,
            "phase": phase,
            "shortcode": cycle["selected"],
            "seed": config["seed"],
            "config": read_json(system.root / cycle["selected"] / "config.json"),
            "max_vram_gb": config["max_vram_gb"],
            "max_training_hours": config["max_training_hours"],
            "provenance": provenance,
            "upstream": upstream,
        }
        contents = {
            "run.json": json.dumps(spec, sort_keys=True).encode(),
            "splits/split_manifest.csv": (system.root / "splits/split_manifest.csv").read_bytes(),
        }
        for path in (system.root / "src/sfsc").glob("*.py"):
            contents[f"src/sfsc/{path.name}"] = path.read_bytes()
        for name in cycle["implementation_hashes"]:
            contents[f"approach/{name}"] = (system.root / cycle["selected"] / name).read_bytes()
        hashes = {
            name: __import__("hashlib").sha256(data).hexdigest() for name, data in contents.items()
        }
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as packed:
            for name, data in contents.items():
                packed.writestr(name, data)
        payload = base64.b64encode(archive.getvalue()).decode()
        launcher = (
            '"""Generated SFSC kernel: exact source snapshot, no installation."""\n'
            "import base64, hashlib, io, sys, zipfile\nfrom pathlib import Path\n"
            f"payload = {payload!r}\nhashes = {hashes!r}\n"
            'root = Path("/kaggle/working/sfsc-source")\nroot.mkdir(exist_ok=False)\n'
            "with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:\n"
            "    for name, expected in hashes.items():\n"
            "        data = archive.read(name)\n"
            "        if hashlib.sha256(data).hexdigest() != expected:\n"
            '            raise RuntimeError("Bundle integrity failure: " + name)\n'
            "        target = root / name\n"
            "        target.parent.mkdir(parents=True, exist_ok=True)\n"
            "        target.write_bytes(data)\n"
            'sys.path.insert(0, str(root / "src"))\n'
            "from sfsc.kaggle_runtime import main\nmain(root)\n"
        )
        (directory / "run.py").write_text(launcher)
        metadata = {
            "id": kernel,
            "title": f"SFSC {run_id}",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": False,
            "machine_shape": "NvidiaTeslaP100",
            "dataset_sources": [],
            "competition_sources": ["filament-segmentation-2026"],
            "kernel_sources": [upstream["kernel"]] if upstream else [],
            "model_sources": [],
        }
        write_json(directory / "kernel-metadata.json", metadata)
        write_json(directory / "run.json", spec)
        run = {
            "run_id": run_id,
            "cycle_id": cycle_id,
            "phase": phase,
            "kernel": kernel,
            "status": "prepared",
            "created_at": now(),
            "spec_hash": object_hash(spec),
            "bundle_hashes": hashes,
            "launcher_hash": digest(directory / "run.py"),
            "metadata_hash": digest(directory / "kernel-metadata.json"),
        }
        state["runs"][run_id] = run
        if phase == "internal_test":
            cycle["holdout_run"] = run_id
        system.event(state, "run_prepared", run_id=run_id)
        return run


def launch(system: System, run_id: str) -> dict:
    with system.transaction() as state:
        system.check(state)
        require(run_id in state["runs"], "Unknown run")
        run = state["runs"][run_id]
        require(run["status"] == "prepared", "Run already launched; inspect remote status")
        directory = system.home / "runs" / run_id
        require(
            digest(directory / "run.py") == run["launcher_hash"]
            and digest(directory / "kernel-metadata.json") == run["metadata_hash"],
            "Prepared kernel was modified",
        )
        run["status"] = "launching"
        timeout = int(state["config"]["max_training_hours"] * 3600)
    # An ambiguous push stays 'launching': inspect it, never automatically double-submit.
    result = subprocess.run(
        [
            "kaggle",
            "kernels",
            "push",
            "--path",
            str(directory),
            "--accelerator",
            "NvidiaTeslaP100",
            "--timeout",
            str(timeout),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    (directory / "launch.log").write_text(result.stdout + result.stderr)
    with system.transaction() as state:
        state["runs"][run_id]["status"] = "submitted"
        system.event(state, "run_submitted", run_id=run_id)
        return state["runs"][run_id]


def collect(system: System, run_id: str, report_path: Path) -> dict:
    report = read_json(report_path)
    with system.transaction() as state:
        system.check(state)
        require(run_id in state["runs"], "Unknown run")
        run = state["runs"][run_id]
        require(run["status"] in {"prepared", "launching", "submitted"}, "Run already resolved")
        require(
            report.get("run_id") == run_id and report.get("spec_hash") == run["spec_hash"],
            "Report belongs to another run or source snapshot",
        )
        require(report.get("phase") == run["phase"], "Report phase mismatch")
        spec = read_json(system.home / "runs" / run_id / "run.json")
        require(object_hash(spec) == run["spec_hash"], "Saved run specification changed")
        require(report.get("provenance") == spec["provenance"], "Report provenance mismatch")
        require(report.get("status") in {"completed", "failed"}, "Report is not terminal")
        cycle = state["cycles"][run["cycle_id"]]
        if report["status"] == "completed":
            require("P100" in report.get("gpu", ""), "Missing P100 runtime evidence")
            number(report.get("peak_vram_gb"), "measured VRAM", 0, spec["max_vram_gb"])
            number(report.get("elapsed_seconds"), "elapsed_seconds", 0)
            require(
                bool(report.get("packages")) and bool(report.get("python")),
                "Runtime package versions required",
            )
            if run["phase"] in {"evaluate", "full_train"}:
                number(
                    report.get("model_seconds"),
                    "model_seconds",
                    0,
                    spec["max_training_hours"] * 3600,
                )
                require(
                    artifact_hash(child(report_path.parent, report.get("checkpoint")))
                    == report.get("checkpoint_hash"),
                    "Checkpoint integrity failure",
                )
            if run["phase"] in {"evaluate", "internal_test"}:
                metric = report.get("metrics", {})
                number(metric.get("pq"), "measured PQ", 0, 1)
                split = "validation" if run["phase"] == "evaluate" else "test"
                require(metric.get("split") == split, "Wrong evaluation split")
                require(
                    metric.get("images") == state["split_counts"][split],
                    "Evaluation did not cover the fixed split",
                )
                for name in ("true_positives", "false_positives", "false_negatives"):
                    number(metric.get(name), name, 0)
                    require(type(metric[name]) is int, f"{name} must be an integer")
                number(metric.get("matched_iou"), "matched_iou", 0, metric["true_positives"])
                denominator = metric["true_positives"] + 0.5 * (
                    metric["false_positives"] + metric["false_negatives"]
                )
                calculated = metric["matched_iou"] / denominator if denominator else 1
                require(abs(calculated - metric["pq"]) < 1e-10, "PQ totals do not agree")
                cycle["validation" if split == "validation" else "internal_test"] = metric
            if run["phase"] == "submit":
                require(
                    digest(report_path.parent / "submission.csv") == report.get("submission_hash"),
                    "Submission integrity failure",
                )
                cycle["submission_path"] = str((report_path.parent / "submission.csv").resolve())
            cycle["status"] = NEXT[run["phase"]]
            if run["phase"] == "evaluate":
                cycle["evaluation_run"] = run_id
            if run["phase"] == "full_train":
                cycle["training_run"] = run_id
        run.update(
            status=report["status"],
            report=report,
            collected_at=now(),
            report_path=str(report_path.resolve()),
        )
        write_json(system.home / "runs" / run_id / "accepted-report.json", report)
        system.event(state, "run_collected", run_id=run_id, status=run["status"])
        return run


def remote_status(system: System, run_id: str) -> str:
    state = system.status()
    require(run_id in state["runs"], "Unknown run")
    result = subprocess.run(
        ["kaggle", "kernels", "status", state["runs"][run_id]["kernel"]],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.strip()


def fetch(system: System, run_id: str) -> dict:
    state = system.status()
    require(run_id in state["runs"], "Unknown run")
    folder = system.home / "runs" / run_id / "download"
    subprocess.run(
        [
            "kaggle",
            "kernels",
            "output",
            state["runs"][run_id]["kernel"],
            "--path",
            str(folder),
            "--force",
        ],
        check=True,
        timeout=300,
    )
    reports = list(folder.rglob("report.json"))
    matching = [p for p in reports if read_json(p).get("run_id") == run_id]
    require(len(matching) == 1, "Expected exactly one run report in downloaded outputs")
    return collect(system, run_id, matching[0])


def fail(system: System, run_id: str, reason: str) -> dict:
    with system.transaction() as state:
        system.check(state)
        require(run_id in state["runs"] and bool(reason.strip()), "Run and reason required")
        run = state["runs"][run_id]
        require(run["status"] in {"prepared", "launching", "submitted"}, "Run already resolved")
        run.update(status="failed", error=reason)
        system.event(state, "run_failed_manually", run_id=run_id, reason=reason)
        return run
