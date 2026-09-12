"""Freeze, launch, monitor, and verify the PQ-SetFormer P100 smoke run."""

from __future__ import annotations

import argparse
import ast
import base64
import fcntl
import functools
import hashlib
import json
import re
import subprocess
import time
import zlib
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import audit
import core
import data
from config import ExperimentConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RECORD = ROOT / "reports/pq_setformer_run.json"
OWNER = "hariohmhemantbhatt"
PRIVATE_DATASET_SOURCE = "hariohmhemantbhatt/sfsc2026-private-data"
TERMINAL_STATUSES = {
    # The two legacy statuses are accepted only to preserve historical evidence.
    "smoke_complete",
    "pilot_complete",
    "execution_complete",
    "calibration_complete",
    "assessment_complete",
    "gate_failed",
    "promoted",
    "failed",
    "cancelled",
}
SOURCE_PATHS = (
    "config.py",
    "core.py",
    "data.py",
    "experiment_logging.py",
    "model.py",
    "remote.py",
    "audit.py",
    "vendor-lock.json",
)
BUNDLE_PATHS = (*SOURCE_PATHS, "sfsc/__init__.py", "sfsc/evaluation.py")


class PushRejected(RuntimeError):
    """Kaggle explicitly rejected a push, even if its CLI exited successfully."""


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return digest_bytes(encoded)


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cli(*args: str, timeout: int = 300) -> str:
    return subprocess.check_output(
        ["kaggle", *args], text=True, stderr=subprocess.STDOUT, timeout=timeout
    )


def frozen_split() -> dict[str, list[str]]:
    """Read the repository's immutable development split without importing legacy code."""
    path = ROOT / "kaggle/yolo-refiner/train_validation.py"
    tree = ast.parse(path.read_text())
    encoded = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SPLIT_B64" for target in node.targets
        )
    )
    return json.loads(zlib.decompress(base64.b64decode(encoded)))


def source_bundle() -> dict[str, str]:
    """Return every source file needed by the isolated Kaggle runtime."""
    bundle = {name: (HERE / name).read_text() for name in SOURCE_PATHS}
    bundle.update(
        {
            "sfsc/__init__.py": (ROOT / "src/sfsc/__init__.py").read_text(),
            "sfsc/evaluation.py": (ROOT / "src/sfsc/evaluation.py").read_text(),
        }
    )
    return bundle


def source_digest(source: dict[str, str]) -> str:
    return canonical_digest(source)


def _snapshot_source(directory: Path) -> dict[str, str]:
    return {name: (directory / name).read_text() for name in BUNDLE_PATHS}


def _assert_live_source_fresh(record: dict[str, Any]) -> None:
    if source_digest(source_bundle()) != record["source_sha256"]:
        raise ValueError("Prepared source is stale; prepare a new immutable run")


@functools.cache
def local_dataset_snapshot() -> dict[str, int | str]:
    return data.training_dataset_snapshot(ROOT / "SFSCDataset/train")


def assert_training_account(record: dict[str, Any], kaggle_config: str) -> None:
    """Require the isolated Kaggle2 identity before any remote mutation."""
    match = re.search(r"(?m)^- username:\s*([^\s]+)\s*$", kaggle_config)
    kernel_owner = record["kernel"].split("/", 1)[0]
    dataset_owner = record["dataset"]["source"].split("/", 1)[0]
    if match is None or match.group(1) != OWNER or kernel_owner != OWNER or dataset_owner != OWNER:
        actual = match.group(1) if match is not None else "unknown"
        raise RuntimeError(
            f"PQ training requires Kaggle account {OWNER}; configured account is {actual}"
        )


def prepare(
    *,
    record_path: Path = RECORD,
    outputs_root: Path | None = None,
    run_id: str | None = None,
    input_size: int = 1024,
    run_kind: str = "smoke",
) -> dict[str, Any]:
    """Freeze the E1 smoke source, configuration, and exact image membership."""
    record_path = Path(record_path)
    outputs_root = Path(outputs_root) if outputs_root is not None else ROOT / "outputs"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous.get("status") not in TERMINAL_STATUSES:
            raise RuntimeError("An existing PQ-SetFormer run is active")
    if run_kind not in {"smoke", "pilot"}:
        raise ValueError("run_kind must be 'smoke' or 'pilot'")

    legacy = frozen_split()
    if {name: len(values) for name, values in legacy.items()} != {
        "train": 495,
        "validation": 106,
        "calibration": 24,
    }:
        raise ValueError("Unexpected frozen development membership")
    split = core.make_experiment_split(legacy["train"], pilot=1)
    if tuple(map(len, split.members.values())) != (289, 69, 34, 87, 16):
        raise ValueError("Blocked pilot-1 membership drifted")

    experiment = ExperimentConfig(model=replace(ModelConfig(), input_size=input_size))
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    directory = outputs_root / f"pq-setformer-{run_id}"
    directory.mkdir(parents=True, exist_ok=False)
    source = source_bundle()
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=30
    ).strip()
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise ValueError("Could not bind the run to a Git commit")
    for name, value in source.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)

    training_names = split.fitting
    # Pilot H is one-shot promotion evidence.  It cannot be placed in the
    # train/infer task that creates Q-fit/Q-cal calibration artifacts.
    inference_names = (
        training_names[:2] if run_kind == "smoke" else split.quality_fit + split.quality_calibration
    )
    training_microbatches = (
        experiment.training.smoke_microbatches
        if run_kind == "smoke"
        else experiment.training.target_microbatches
    )
    membership = {
        "training_names": training_names,
        "inference_names": inference_names,
        "split": split.to_dict(),
    }
    record = {
        "status": "prepared",
        "stage": f"E1-{run_kind}-{input_size}",
        "run_kind": run_kind,
        "training_microbatches": training_microbatches,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "directory": str(directory.resolve()),
        "kernel": f"{OWNER}/sfsc-pq-setformer-{run_id}",
        "source_sha256": source_digest(source),
        "source_files": list(source),
        "vendor_lock_sha256": digest_bytes(source["vendor-lock.json"].encode()),
        "git_sha": git_sha,
        "membership_sha256": canonical_digest(membership),
        "config_sha256": experiment.sha256,
        "experiment": experiment.to_dict(),
        "training_names": training_names,
        "inference_names": inference_names,
        "split": split.to_dict(),
        "automatic_submission": False,
        "dataset": {"source": PRIVATE_DATASET_SOURCE, **local_dataset_snapshot()},
    }
    write_json(directory / "manifest.json", record)
    write_json(record_path, record)
    return record


def _bootstrap(payload: str, lock: dict[str, Any]) -> str:
    detectron = lock["detectron2"]["commit"]
    maskdino = lock["maskdino"]["commit"]
    return f'''"""Generated immutable PQ-SetFormer Kaggle launcher."""
import base64, json, os, runpy, subprocess, sys, zlib
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("MAX_JOBS", "2")

def command(*args, cwd=None):
    subprocess.check_call(list(args), cwd=cwd)

command(sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps",
        "torch==2.7.1", "torchvision==0.22.1",
        "--index-url", "https://download.pytorch.org/whl/cu126")
command(sys.executable, "-m", "pip", "install", "--quiet",
        "wandb==0.30.0", "pycocotools==2.0.11", "scipy==1.15.3",
        "timm==0.9.16", "ninja==1.11.1.4")
command(sys.executable, "-m", "pip", "install", "--quiet", "--no-build-isolation",
        "git+https://github.com/facebookresearch/detectron2.git@{detectron}")

vendor = Path("/kaggle/working/vendor")
command("git", "clone", "--quiet", "https://github.com/IDEA-Research/MaskDINO.git", str(vendor))
command("git", "checkout", "--quiet", "{maskdino}", cwd=vendor)
decoder = vendor / "maskdino/modeling/transformer_decoder/maskdino_decoder.py"
decoder_source = decoder.read_text()
quality_anchor = "        out = {{\\n            'pred_logits': predictions_class[-1],"
quality_replacement = (
    "        # PQ-SetFormer E2 adapter: final matching-query features and\\n"
    "        # stride-4 pixel features stay behind the upstream decoder interface.\\n"
    "        pq_query_features = hs[-1].transpose(0, 1)\\n"
    "        if mask_dict is not None:\\n"
    "            pq_query_features = pq_query_features[:, mask_dict['pad_size']:, :]\\n"
    "        out = {{\\n"
    "            'pred_logits': predictions_class[-1],"
)
if decoder_source.count(quality_anchor) != 1:
    raise RuntimeError("Pinned MaskDINO decoder quality adapter anchor changed unexpectedly")
decoder_source = decoder_source.replace(quality_anchor, quality_replacement)
quality_fields = "            'pred_boxes':out_boxes[-1],"
quality_fields_replacement = (
    "            'pred_boxes':out_boxes[-1],\\n"
    "            'pred_query_features': pq_query_features,\\n"
    "            'pixel_features': mask_features,"
)
if decoder_source.count(quality_fields) != 1:
    raise RuntimeError("Pinned MaskDINO decoder quality output anchor changed unexpectedly")
decoder.write_text(decoder_source.replace(quality_fields, quality_fields_replacement))
ops = vendor / "maskdino/modeling/pixel_decoder/ops"
dispatch = ops / "src/cuda/ms_deform_attn_cuda.cu"
dispatch_source = dispatch.read_text()
legacy_dispatch = "AT_DISPATCH_FLOATING_TYPES(value.type(),"
if dispatch_source.count(legacy_dispatch) != 2:
    raise RuntimeError("Pinned MaskDINO dispatch source changed unexpectedly")
modern_dispatch = legacy_dispatch.replace("value.type()", "value.scalar_type()")
dispatch.write_text(dispatch_source.replace(legacy_dispatch, modern_dispatch))
command("sh", "make.sh", cwd=ops)

payload = json.loads(zlib.decompress(base64.b64decode({payload!r})))
root = Path("/kaggle/working")
for name, source in payload.items():
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
sys.path[:0] = [str(root), str(vendor)]
os.chdir(root)
runpy.run_path(str(root / "remote.py"), run_name="__main__")
'''


def build_kernel(record: dict[str, Any]) -> Path:
    """Build a self-contained launcher only from the prepared snapshot."""
    directory = Path(record["directory"])
    frozen = _snapshot_source(directory)
    if source_digest(frozen) != record["source_sha256"]:
        raise ValueError("Frozen source changed")
    task = {
        key: record[key]
        for key in (
            "run_id",
            "stage",
            "run_kind",
            "training_microbatches",
            "source_sha256",
            "source_files",
            "vendor_lock_sha256",
            "git_sha",
            "membership_sha256",
            "config_sha256",
            "experiment",
            "training_names",
            "inference_names",
            "split",
            "dataset",
        )
    }
    payload = {**frozen, "task-config.json": json.dumps(task, sort_keys=True)}
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode(), level=9)).decode()
    lock = json.loads(frozen["vendor-lock.json"])
    launcher = _bootstrap(encoded, lock)
    (directory / "run.py").write_text(launcher)
    metadata = {
        "id": record["kernel"],
        "title": f"SFSC PQ-SetFormer {record['run_kind']} {record['run_id']}",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaP100",
        "dataset_sources": [record["dataset"]["source"]],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(directory / "kernel-metadata.json", metadata)
    write_json(directory / "task-config.json", task)
    record["launcher_sha256"] = digest_bytes(launcher.encode())
    return directory


def _find_unique(directory: Path, filename: str) -> Path:
    matches = list(directory.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename}, found {len(matches)}")
    return matches[0]


def fetch_and_verify(record: dict[str, Any]) -> dict[str, Any]:
    """Download all smoke evidence and independently validate its lineage."""
    directory = Path(record["directory"])
    destination = directory / "results"
    destination.mkdir(parents=True, exist_ok=True)
    output = _cli(
        "kernels",
        "output",
        record["kernel"],
        "-p",
        str(destination),
        "--force",
        timeout=900,
    )
    (directory / "fetch.log").write_text(output)
    prefix = record["run_kind"]
    report_path = _find_unique(destination, f"{prefix}-report.json")
    checkpoint = _find_unique(destination, f"{prefix}-checkpoint.pt")
    candidates = _find_unique(destination, f"{prefix}-candidates.json")
    logging_manifest = _find_unique(destination, "logging-manifest.json")
    report = json.loads(report_path.read_text())
    if any(
        report.get(name) != record[name]
        for name in (
            "run_id",
            "stage",
            "run_kind",
            "training_microbatches",
            "source_sha256",
            "vendor_lock_sha256",
            "git_sha",
            "membership_sha256",
            "config_sha256",
            "dataset",
        )
    ):
        raise ValueError("Smoke report lineage mismatch")
    expected_steps = (
        record["training_microbatches"] // record["experiment"]["training"]["accumulation"]
    )
    issues = audit.audit_stage(
        report,
        checkpoint,
        candidates,
        logging_manifest.parent,
        expected_microbatches=record["training_microbatches"],
        expected_optimizer_steps=expected_steps,
        expected_candidate_images=len(record["inference_names"]),
        accumulation=record["experiment"]["training"]["accumulation"],
    )
    if issues:
        raise ValueError("Smoke artifact audit failed: " + "; ".join(issues))
    return report


def _poll_status(record: dict[str, Any], record_path: Path) -> None:
    failures = 0
    while True:
        try:
            status = _cli("kernels", "status", record["kernel"]).strip()
            failures = 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            failures += 1
            if failures >= 5:
                raise
            time.sleep(30)
            continue
        record.update(
            status="running",
            remote_status=status,
            checked_at=datetime.now(UTC).isoformat(),
        )
        write_json(record_path, record)
        print(json.dumps({"kernel": record["kernel"], "remote_status": status}), flush=True)
        upper = status.upper()
        if "COMPLETE" in upper:
            record.update(
                status="remote_complete_fetch_pending",
                remote_completed_at=datetime.now(UTC).isoformat(),
            )
            write_json(record_path, record)
            return
        if any(word in upper for word in ("ERROR", "FAIL", "CANCEL")):
            raise RuntimeError(status)
        time.sleep(45)


def kernel_from_push_output(output: str) -> str:
    errors = re.findall(r"Kernel push error:\s*([^\r\n]+)", output, flags=re.IGNORECASE)
    if errors:
        raise PushRejected(errors[-1].strip())
    urls = re.findall(r"https://www\.kaggle\.com/code/([\w-]+/[\w-]+)", output)
    if not urls:
        raise RuntimeError("Push result did not contain a Kaggle kernel URL")
    return urls[-1]


def monitor(record: dict[str, Any], *, record_path: Path = RECORD) -> None:
    """Submit at most once, then monitor and verify the completed smoke run."""
    record_path = Path(record_path)
    if record["status"] == "prepared":
        assert_training_account(record, _cli("config", "view", timeout=30))
        _assert_live_source_fresh(record)
        directory = build_kernel(record)
        record.update(status="pushing", push_started_at=datetime.now(UTC).isoformat())
        write_json(record_path, record)
        output = _cli(
            "kernels",
            "push",
            "-p",
            str(directory),
            "--accelerator",
            "NvidiaTeslaP100",
            "--timeout",
            "32400",
            timeout=600,
        )
        (directory / "push.log").write_text(output)
        record.update(
            kernel=kernel_from_push_output(output),
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )
        write_json(record_path, record)
    if record["status"] in {"pushing"}:
        raise RuntimeError("Push outcome is ambiguous; inspect Kaggle before retrying")
    if record["status"] == "running":
        _poll_status(record, record_path)
    if record["status"] != "remote_complete_fetch_pending":
        raise RuntimeError(f"Cannot fetch run in state {record['status']!r}")
    report = fetch_and_verify(record)
    record.update(
        status="execution_complete",
        verified_at=datetime.now(UTC).isoformat(),
        report=report,
    )
    write_json(record_path, record)


def record_promotion_gate(
    record: dict[str, Any], record_path: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    """Record the only terminal state permitted after a successful execution.

    Promotion needs an independently trained matched-R control and the complete
    CPU diagnostic/blocked-bootstrap evidence.  This function deliberately
    cannot launch a stage and an incomplete evidence packet is durable failure,
    not an invitation to tune on H.
    """
    if record.get("status") != "assessment_complete":
        raise ValueError("Only a frozen H assessment can enter the promotion gate")
    if evidence != record.get("assessment"):
        raise ValueError("Promotion gate evidence must be the immutable recorded H assessment")
    diagnostics = evidence.get("diagnostics")
    bootstrap = evidence.get("bootstrap")
    reason = None
    if not isinstance(evidence.get("matched_control"), dict):
        reason = "matched control evidence is absent"
    elif not isinstance(diagnostics, dict) or not {
        "raw_candidate_recall",
        "final_recall",
    } <= set(diagnostics):
        reason = "required candidate recall diagnostics are absent"
    elif not isinstance(bootstrap, dict) or not {
        "pq_delta",
        "bootstrap_lower_95",
    } <= set(bootstrap):
        reason = "paired date-block bootstrap evidence is absent"
    elif bootstrap["pq_delta"] < 0.015 or bootstrap["bootstrap_lower_95"] <= 0:
        reason = "PQ gate did not pass"
    elif evidence.get("rq_delta", 0.0) <= 0 or evidence.get("sq_delta", 0.0) < -0.01:
        reason = "recognition/segmentation quality gate did not pass"
    record["promotion_evidence"] = evidence
    record["gate_decided_at"] = datetime.now(UTC).isoformat()
    if reason:
        record["status"] = "gate_failed"
        record["gate_failure_reason"] = reason
    else:
        record["status"] = "promoted"
    write_json(record_path, record)
    return record


def record_frozen_h_assessment(
    record: dict[str, Any], record_path: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    """Attach downloaded one-shot H evidence only after Q calibration freezes."""
    if record.get("status") != "calibration_complete":
        raise ValueError("H assessment requires a completed frozen Q calibration")
    required = {"challenger", "matched_control", "diagnostics", "bootstrap"}
    if not required <= set(evidence):
        raise ValueError("Frozen H assessment evidence is incomplete")
    h_membership = canonical_digest(sorted(record["split"]["members"]["assessment"]))
    if evidence.get("assessment_membership_sha256") != h_membership:
        raise ValueError("Frozen H assessment membership hash is invalid")
    record.update(
        status="assessment_complete",
        assessment=evidence,
        assessment_completed_at=datetime.now(UTC).isoformat(),
    )
    write_json(record_path, record)
    return record


def calibrate_q_artifacts(record: dict[str, Any], *, record_path: Path) -> dict[str, Any]:
    """Fit Q-fit and choose Q-cal only; H is structurally unavailable here."""
    if record.get("status") != "execution_complete":
        raise ValueError("Only execution_complete Q inference can be calibrated")
    if record.get("run_kind") != "pilot":
        raise ValueError("Only a frozen pilot has Q-fit/Q-cal memberships")
    split = record["split"]["members"]
    q_fit_names = set(split["quality_fit"])
    q_cal_names = set(split["quality_calibration"])
    if q_fit_names & q_cal_names or not q_fit_names or not q_cal_names:
        raise ValueError("Frozen Q-fit/Q-cal membership is invalid")
    report = record.get("report")
    if not isinstance(report, dict):
        raise ValueError("Downloaded execution report is required before calibration")
    cache_path = _find_unique(Path(record["directory"]) / "results", "pilot-candidates.json")
    if digest(cache_path) != report.get("candidates_sha256"):
        raise ValueError("Q candidate cache hash does not match verified execution report")
    raw_cache = json.loads(cache_path.read_text())
    if set(raw_cache) != q_fit_names | q_cal_names:
        raise ValueError("Q candidate cache coverage differs from frozen Q membership")
    candidates = {
        filename: [core.deserialize_candidate(row) for row in rows]
        for filename, rows in raw_cache.items()
    }
    annotation = ROOT / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    index = data.load_annotation_index(annotation, q_fit_names | q_cal_names)
    truths = {
        filename: [
            list(data.rasterize_record(row, native_size=2048, output_size=2048))
            for row in index.records_by_file[filename]
        ]
        for filename in candidates
    }
    experiment = ExperimentConfig.from_dict(record["experiment"])
    recipes = [
        replace(
            experiment.assembly,
            mask_threshold=threshold,
            utility_threshold=utility,
            minimum_area=minimum_area,
        )
        for threshold in (0.4, 0.5, 0.6)
        for utility in (0.1, 0.2, 0.3)
        for minimum_area in (0, 32, 64, 96)
    ]
    calibrator, recipe, selection = core.fit_and_select_q_recipe(
        {name: candidates[name] for name in q_fit_names},
        {name: truths[name] for name in q_fit_names},
        {name: candidates[name] for name in q_cal_names},
        {name: truths[name] for name in q_cal_names},
        recipes,
    )
    artifact = {
        "schema_version": 1,
        "kind": "q-fit-q-cal-only",
        "source_sha256": record["source_sha256"],
        "config_sha256": record["config_sha256"],
        "membership_sha256": record["membership_sha256"],
        "calibrator": calibrator.to_dict(),
        "calibrator_sha256": calibrator.sha256,
        "assembly": asdict(recipe),
        "selection": selection,
    }
    artifact["sha256"] = canonical_digest(artifact)
    path = Path(record["directory"]) / "q-calibration.json"
    write_json(path, artifact)
    record.update(
        status="calibration_complete",
        calibration_path=str(path),
        calibration_sha256=digest(path),
    )
    write_json(record_path, record)
    return record


def record_controller_failure(record: dict[str, Any], record_path: Path, error: Exception) -> None:
    """Persist failures without turning an ambiguous remote success into a rerunnable job."""
    if isinstance(error, PushRejected):
        record.update(
            status="failed",
            error=str(error),
            stopped_at=datetime.now(UTC).isoformat(),
        )
    elif record.get("status") in {"pushing", "remote_complete_fetch_pending"}:
        record.update(
            controller_error=str(error),
            controller_stopped_at=datetime.now(UTC).isoformat(),
        )
    else:
        record.update(
            status="failed",
            error=str(error),
            stopped_at=datetime.now(UTC).isoformat(),
        )
    write_json(record_path, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "status", "calibrate", "assess", "gate")
    )
    parser.add_argument("--evidence", type=Path, help="Frozen H assessment or promotion-gate JSON")
    parser.add_argument(
        "--input-size",
        type=int,
        choices=(1024, 1152, 1216, 1280),
        default=1024,
        help="Frozen square input size for a newly prepared preflight",
    )
    parser.add_argument(
        "--run-kind",
        choices=("smoke", "pilot"),
        default="smoke",
        help="Run a short hardware smoke or the frozen 6,000-microbatch E1 pilot",
    )
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        print(
            json.dumps(
                prepare(input_size=arguments.input_size, run_kind=arguments.run_kind), indent=2
            )
        )
        return
    if arguments.command == "status":
        if not RECORD.is_file():
            raise RuntimeError("No PQ-SetFormer run has been prepared")
        print(RECORD.read_text(), end="")
        return
    if arguments.command == "calibrate":
        if not RECORD.is_file():
            raise RuntimeError("No PQ-SetFormer execution has been prepared")
        print(
            json.dumps(
                calibrate_q_artifacts(json.loads(RECORD.read_text()), record_path=RECORD), indent=2
            )
        )
        return
    if arguments.command in {"assess", "gate"}:
        if not RECORD.is_file() or arguments.evidence is None:
            raise RuntimeError("Assessment/gate needs an active record and --evidence JSON")
        evidence = json.loads(arguments.evidence.read_text())
        record = json.loads(RECORD.read_text())
        result = (
            record_frozen_h_assessment(record, RECORD, evidence)
            if arguments.command == "assess"
            else record_promotion_gate(record, RECORD, evidence)
        )
        print(json.dumps(result, indent=2))
        return

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with RECORD.with_suffix(".lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A PQ-SetFormer controller is already running") from error
        record = json.loads(RECORD.read_text()) if RECORD.exists() else prepare()
        try:
            monitor(record)
        except Exception as error:
            record_controller_failure(record, RECORD, error)
            raise


if __name__ == "__main__":
    main()
