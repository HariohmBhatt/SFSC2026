"""Package exact baseline lineage and frozen precision configuration for P100."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from sfsc.system import digest, object_hash


def main():
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).parent
    baseline_dir = root / ".sfsc/runs/cycle-001-full-train-e0b5f3ed"
    baseline_report = json.loads((baseline_dir / "accepted-report.json").read_text())
    baseline_recipe = json.loads(
        (baseline_dir / "download/experiment/checkpoint/recipe.json").read_text()
    )
    result_dir = root / "outputs/precision-v2-20260906-052615/results/experiment"
    precision_report = json.loads((result_dir / "report.json").read_text())
    precision_recipe = json.loads((result_dir / "checkpoint/recipe.json").read_text())
    assert baseline_report["status"] == precision_report["status"] == "completed"
    assert baseline_report["phase"] == "full_train" and precision_report["phase"] == "evaluate"
    manifest_hash = digest(root / "splits/split_manifest.csv")
    for report in [baseline_report, precision_report]:
        assert report["provenance"]["contract"]["splits/split_manifest.csv"] == manifest_hash
    run_id = "precision-full-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    directory = root / "outputs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    upstream = {
        "run_id": baseline_report["run_id"],
        "kernel": "hariohmbhatt/sfsc-cycle-001-full-train-e0b5f3ed",
        "checkpoint_hash": baseline_report["checkpoint_hash"],
    }
    contents = {
        "approach/pipeline.py": (source / "train.py").read_bytes(),
        "approach/plan.py": (source / "plan.py").read_bytes(),
        "approach/precision_model.py": (root / "filament-precision/pipeline.py").read_bytes(),
        "approach/geometry.py": (root / "filament-precision/geometry.py").read_bytes(),
        "approach/precision-recipe.json": json.dumps(precision_recipe, sort_keys=True).encode(),
        "approach/baseline-recipe.json": json.dumps(baseline_recipe, sort_keys=True).encode(),
        "splits/split_manifest.csv": (root / "splits/split_manifest.csv").read_bytes(),
    }
    contents.update(
        {f"src/sfsc/{path.name}": path.read_bytes() for path in (root / "src/sfsc").glob("*.py")}
    )
    hashes = {
        name: __import__("hashlib").sha256(data).hexdigest() for name, data in contents.items()
    }
    spec = {
        "run_id": run_id,
        "cycle_id": "precision-full",
        "phase": "full_train",
        "shortcode": "precision-fulltrain",
        "seed": 2026,
        "config": precision_recipe["config"],
        "upstream": upstream,
        "max_vram_gb": 14.0,
        "max_training_hours": 9.0,
        "provenance": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "source_hashes": hashes,
            "contract": precision_report["provenance"]["contract"],
            "annotation_hash": precision_report["provenance"]["annotation_hash"],
            "frozen_precision_run": precision_report["run_id"],
            "frozen_precision_checkpoint_hash": precision_report["checkpoint_hash"],
            "baseline_reused": "Verified completed 60-epoch baseline on train+validation",
        },
    }
    contents["run.json"] = json.dumps(spec, sort_keys=True).encode()
    # Do not mutate hashes inside spec after calculating the source provenance.
    bundle_hashes = {n: __import__("hashlib").sha256(b).hexdigest() for n, b in contents.items()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    payload = base64.b64encode(stream.getvalue()).decode()
    launcher = (
        "import base64, hashlib, io, sys, zipfile\nfrom pathlib import Path\n"
        f"payload = {payload!r}\nhashes = {bundle_hashes!r}\n"
        'root = Path("/kaggle/working/sfsc-source")\nroot.mkdir(exist_ok=False)\n'
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:\n"
        "    for name, expected in hashes.items():\n"
        "        data = archive.read(name)\n"
        "        assert hashlib.sha256(data).hexdigest() == expected\n"
        "        target = root / name\n"
        "        target.parent.mkdir(parents=True, exist_ok=True)\n"
        "        target.write_bytes(data)\n"
        'sys.path.insert(0, str(root / "src"))\n'
        "from sfsc.kaggle_runtime import main\nmain(root)\n"
    )
    compile(launcher, "run.py", "exec")
    (directory / "run.py").write_text(launcher)
    metadata = {
        "id": f"hariohmbhatt/sfsc-{run_id}",
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
        "kernel_sources": [upstream["kernel"]],
        "model_sources": [],
    }
    for name, value in [("run.json", spec), ("kernel-metadata.json", metadata)]:
        (directory / name).write_text(json.dumps(value, indent=2) + "\n")
    record = {
        "run_id": run_id,
        "kernel": metadata["id"],
        "directory": str(directory.relative_to(root)),
        "spec_hash": object_hash(spec),
        "status": "prepared",
        "submission_authorized": True,
        "frozen_precision_epoch": precision_recipe["best_epoch"],
        "train_images": 601,
        "baseline_checkpoint_hash": upstream["checkpoint_hash"],
    }
    (root / "reports/full_training.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
