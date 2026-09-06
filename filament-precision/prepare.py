"""Package a validation-only P100 experiment without altering frozen cycle history."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from sfsc.system import digest, object_hash


def main():
    root = Path(__file__).resolve().parents[1]
    approach = root / "filament-precision"
    previous = root / ".sfsc/runs/cycle-001-evaluate-eaae4548"
    old_spec = json.loads((previous / "run.json").read_text())
    old_report = json.loads((previous / "accepted-report.json").read_text())
    assert old_report["status"] == "completed" and old_report["phase"] == "evaluate"
    # Reject any accidental attachment of the train+validation/full-training model.
    assert old_spec["phase"] == "evaluate" and old_spec["upstream"] is None
    assert (
        digest(root / "splits/split_manifest.csv")
        == old_spec["provenance"]["contract"]["splits/split_manifest.csv"]
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"precision-v2-{timestamp}"
    destination = root / "outputs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    sources = {f"approach/{p.name}": p.read_bytes() for p in approach.glob("*.py")}
    sources["approach/config.json"] = (approach / "config.json").read_bytes()
    sources["approach/APPROACH.md"] = (approach / "APPROACH.md").read_bytes()
    sources.update({f"src/sfsc/{p.name}": p.read_bytes() for p in (root / "src/sfsc").glob("*.py")})
    sources["splits/split_manifest.csv"] = (root / "splits/split_manifest.csv").read_bytes()
    upstream = {
        "run_id": old_spec["run_id"],
        "kernel": "hariohmbhatt/sfsc-cycle-001-evaluate-eaae4548",
        "checkpoint_hash": old_report["checkpoint_hash"],
    }
    spec = {
        "run_id": run_id,
        "cycle_id": "research-precision-v2",
        "phase": "evaluate",
        "shortcode": "filament-precision",
        "seed": 2026,
        "config": json.loads((approach / "config.json").read_text()),
        "max_vram_gb": 14.0,
        "max_training_hours": 9.0,
        "upstream": upstream,
        "provenance": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "working_tree": "Source hashes are authoritative; existing uncommitted work preserved",
            "contract": old_spec["provenance"]["contract"],
            "annotation_hash": digest(
                root / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
            ),
            "source_hashes": {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()},
            "holdout_policy": "Previously consumed internal test; validation only; no reset",
        },
    }
    sources["run.json"] = json.dumps(spec, sort_keys=True).encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sources.items():
            z.writestr(name, data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    payload = base64.b64encode(archive.getvalue()).decode()
    launcher = (
        '"""Private SFSC precision experiment; embedded source snapshot."""\n'
        "import base64, hashlib, io, sys, zipfile\nfrom pathlib import Path\n"
        f"payload = {payload!r}\nhashes = {hashes!r}\n"
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
    # Validate the exact executable sent to Kaggle before creating a remote job.
    compile(launcher, "run.py", "exec")
    (destination / "run.py").write_text(launcher)
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
    for filename, data in [("run.json", spec), ("kernel-metadata.json", metadata)]:
        (destination / filename).write_text(json.dumps(data, indent=2) + "\n")
    record = {
        "run_id": run_id,
        "kernel": metadata["id"],
        "url": f"https://www.kaggle.com/code/{metadata['id']}",
        "directory": str(destination.relative_to(root)),
        "spec_hash": object_hash(spec),
        "status": "prepared",
    }
    (root / "reports/precision_training.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
