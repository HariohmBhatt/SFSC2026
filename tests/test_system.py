"""Exercise the complete orchestration lifecycle without a model service or GPU."""

from __future__ import annotations

import ast
import base64
import copy
import csv
import io
import json
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sfsc import remote
from sfsc.contracts import ContractError, implementation, rank, research
from sfsc.system import System, digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def system(tmp_path):
    for name in ("DATA.md", "EVAL.md", "GPU.md"):
        shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "system", tmp_path / "system")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    (tmp_path / "splits").mkdir()
    with (tmp_path / "splits/split_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "split"])
        writer.writeheader()
        for index in range(20):
            writer.writerow(
                {
                    "filename": f"image-{index}.jpeg",
                    "split": "train" if index < 14 else "validation" if index < 17 else "test",
                }
            )
    annotation = tmp_path / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text("{}")
    instance = System(tmp_path)
    instance.init()
    return instance


def proposal(code):
    return {
        "shortcode": code,
        "title": "Synthetic protocol fixture",
        "mechanism": "fixture",
        "data_fit": "fixture",
        "runtime_plan": "fixture",
        "risks": "fixture",
        "experiment": "fixture",
        "sources": [
            {
                "kind": "preprint",
                "title": "Fixture",
                "authors": "Fixture",
                "year": 2020,
                "venue": "Fixture",
                "url": "https://example.org/test-paper",
                "claim": "Contract test only",
            }
        ],
    }


def review(candidates):
    return {
        "reviews": [
            {
                "shortcode": c["shortcode"],
                "pq_potential": 0.7,
                "complexity": 2,
                "training_hours": 2,
                "vram_gb": 8,
                "model_mb": 100,
                "compatible": True,
                "rationale": "Fixture",
                "assumptions": "Fixture",
                "uncertainty": "Fixture",
                "training_roi": "Fixture",
            }
            for c in candidates
        ]
    }


def code_result():
    return {
        "approach_markdown": "# Synthetic test implementation",
        "config": {"epochs": 1},
        "files": {
            "pipeline.py": "def run(context):\n    return {}\n",
            "tests/test_pipeline.py": "def test_fixture():\n    assert True\n",
        },
    }


def advance(system):
    cycle = system.cycle()
    for index, job in enumerate(cycle["research_jobs"]):
        cycle = system.accept(
            job, {"approaches": [proposal(f"fixture-{index}-{j}") for j in range(2)]}
        )
    state = read_json(system.state_path)
    candidates = state["cycles"][cycle["id"]]["candidates"]
    cycle = system.accept(cycle["review_job"], review(candidates))
    return system.accept(cycle["implementation_job"], code_result())


def fake_report(system, run, folder):
    """Remote boundary fixture, explicitly not a real measured experiment."""
    folder.mkdir(parents=True)
    spec = read_json(system.home / "runs" / run["run_id"] / "run.json")
    report = {
        "run_id": run["run_id"],
        "spec_hash": run["spec_hash"],
        "phase": run["phase"],
        "provenance": spec["provenance"],
        "status": "completed",
        "gpu": "Test P100",
        "python": "fixture",
        "packages": {"tensorflow": "fixture"},
        "peak_vram_gb": 1,
        "elapsed_seconds": 1,
        "model_seconds": 1,
    }
    if run["phase"] in {"evaluate", "full_train"}:
        (folder / "model.keras").write_bytes(b"synthetic checkpoint")
        report.update(checkpoint="model.keras", checkpoint_hash=digest(folder / "model.keras"))
    if run["phase"] in {"evaluate", "internal_test"}:
        report["metrics"] = {
            "pq": 0.5,
            "matched_iou": 1.5,
            "true_positives": 2,
            "false_positives": 1,
            "false_negatives": 1,
            "images": 3,
            "split": "validation" if run["phase"] == "evaluate" else "test",
        }
    if run["phase"] == "submit":
        (folder / "submission.csv").write_text("filament_id,segmentation_rle\n")
        report["submission_hash"] = digest(folder / "submission.csv")
    write_json(folder / "report.json", report)
    return folder / "report.json"


def test_entire_three_fleet_and_four_stage_lifecycle(system, tmp_path):
    cycle = advance(system)
    assert cycle["status"] == "implemented"
    assert len(cycle["ranking"]) == 6
    assert (system.root / cycle["selected"] / "APPROACH.md").exists()
    for phase in remote.PHASES:
        if phase == "internal_test":
            system.freeze(cycle["id"])
        run = remote.prepare(system, cycle["id"], phase, "fixture")
        path = fake_report(system, run, tmp_path / f"result-{phase}")
        remote.collect(system, run["run_id"], path)
    final = system.status()
    assert final["cycles"][cycle["id"]]["status"] == "submission_ready"
    with pytest.raises(ContractError, match="frozen"):
        system.cycle()


def test_parallel_research_does_not_lose_results(system):
    cycle = system.cycle()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                system.accept,
                job,
                {"approaches": [proposal(f"parallel-{index}-{j}") for j in range(2)]},
            )
            for index, job in enumerate(cycle["research_jobs"])
        ]
        for future in futures:
            future.result()
    state = read_json(system.state_path)
    assert len(state["cycles"][cycle["id"]]["candidates"]) == 6
    assert len([j for j in state["jobs"].values() if j["role"] == "review"]) == 1


def test_invalid_result_is_atomic_and_retryable(system):
    cycle = system.cycle()
    before = system.state_path.read_bytes()
    with pytest.raises(ContractError, match="at least 2"):
        system.accept(cycle["research_jobs"][0], {"approaches": [proposal("one")]})
    assert system.state_path.read_bytes() == before
    system.accept(cycle["research_jobs"][0], {"approaches": [proposal("one"), proposal("two")]})
    with pytest.raises(ContractError, match="already accepted"):
        system.accept(cycle["research_jobs"][0], {})


def test_fixed_contract_and_source_changes_are_rejected(system):
    cycle = advance(system)
    source = system.root / cycle["selected"] / "pipeline.py"
    source.write_text("# changed")
    with pytest.raises(ContractError, match="implementation changed"):
        remote.prepare(system, cycle["id"], "evaluate", "fixture")
    (system.root / "EVAL.md").write_text("Changed metric")
    with pytest.raises(ContractError, match="Fixed contract changed"):
        system.status()


def test_resource_ranking_and_complete_review(system):
    candidates = [proposal("small"), proposal("large")]
    result = review(candidates)
    result["reviews"][1].update(pq_potential=1, vram_gb=24)
    ranked = rank(result, candidates, read_json(system.state_path)["config"])
    assert ranked[0]["shortcode"] == "small"
    assert ranked[1]["feasible"] is False
    result["reviews"].pop()
    with pytest.raises(ContractError, match="every candidate"):
        rank(result, candidates, read_json(system.state_path)["config"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_costs_rejected(system, value):
    candidates = [proposal("small"), proposal("large")]
    result = review(candidates)
    result["reviews"][0]["vram_gb"] = value
    with pytest.raises(ContractError):
        rank(result, candidates, read_json(system.state_path)["config"])


@pytest.mark.parametrize(
    "path", ["../escape.py", "/tmp/escape.py", "a/../../x.py", ".git/config.py", "x\\evil.py"]
)
def test_implementation_cannot_escape_short_code(path):
    result = code_result()
    result["files"][path] = "# contents"
    with pytest.raises(ContractError, match="Unsafe"):
        implementation(result)


def test_academic_source_metadata_required():
    result = {"approaches": [proposal("one"), proposal("two")]}
    result["approaches"][0]["sources"][0]["kind"] = "blog"
    with pytest.raises(ContractError, match="academic"):
        research(result, 2)


def test_full_training_cannot_skip_validation_and_holdout(system):
    cycle = advance(system)
    with pytest.raises(ContractError, match="requires tested"):
        remote.prepare(system, cycle["id"], "full_train", "fixture")
    with pytest.raises(ContractError, match="evaluated"):
        system.freeze(cycle["id"])


def test_stale_or_tampered_remote_artifacts_do_not_advance(system, tmp_path):
    cycle = advance(system)
    run = remote.prepare(system, cycle["id"], "evaluate", "fixture")
    path = fake_report(system, run, tmp_path / "result")
    report = read_json(path)
    bad = copy.deepcopy(report)
    bad["spec_hash"] = "wrong"
    write_json(path, bad)
    with pytest.raises(ContractError, match="another run"):
        remote.collect(system, run["run_id"], path)
    write_json(path, report)
    (path.parent / "model.keras").write_bytes(b"tampered")
    with pytest.raises(ContractError, match="Checkpoint integrity"):
        remote.collect(system, run["run_id"], path)
    assert system.status()["cycles"][cycle["id"]]["status"] == "implemented"


def test_holdout_cannot_be_repeated_even_after_failure(system, tmp_path):
    cycle = advance(system)
    run = remote.prepare(system, cycle["id"], "evaluate", "fixture")
    remote.collect(system, run["run_id"], fake_report(system, run, tmp_path / "result"))
    system.freeze(cycle["id"])
    held = remote.prepare(system, cycle["id"], "internal_test", "fixture")
    remote.fail(system, held["run_id"], "Synthetic failure")
    with pytest.raises(ContractError, match="reserved once"):
        remote.prepare(system, cycle["id"], "internal_test", "fixture")


def test_kernel_embeds_exact_sources_and_checkpoint_mount(system, tmp_path):
    cycle = advance(system)
    run = remote.prepare(system, cycle["id"], "evaluate", "fixture")
    source = (system.home / "runs" / run["run_id"] / "run.py").read_text()
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"payload", "hashes"}
    }
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(assignments["payload"]))) as archive:
        assert (
            archive.read("approach/pipeline.py")
            == (system.root / cycle["selected"] / "pipeline.py").read_bytes()
        )
        assert "src/sfsc/evaluation.py" in archive.namelist()
    remote.collect(system, run["run_id"], fake_report(system, run, tmp_path / "result"))
    system.freeze(cycle["id"])
    held = remote.prepare(system, cycle["id"], "internal_test", "fixture")
    meta = read_json(system.home / "runs" / held["run_id"] / "kernel-metadata.json")
    assert meta["kernel_sources"] == [run["kernel"]]
    assert meta["is_private"] is True and meta["enable_internet"] is False


def test_worker_failure_is_recorded_then_can_retry(system):
    cycle = system.cycle()
    job = cycle["research_jobs"][0]
    with pytest.raises(Exception, match="exit status 1"):
        system.work(job, [sys.executable, "-c", "raise SystemExit(1)"])
    assert system.status()["jobs"][job]["status"] == "failed"
    result = {"approaches": [proposal("one"), proposal("two")]}
    source = "import sys,json; json.load(sys.stdin); print(" + repr(json.dumps(result)) + ")"
    system.work(job, [sys.executable, "-c", source])
    assert system.status()["jobs"][job]["status"] == "completed"


def test_all_infeasible_starts_new_iteration_without_implementation(system):
    cycle = system.cycle()
    for index, job in enumerate(cycle["research_jobs"]):
        cycle = system.accept(job, {"approaches": [proposal(f"bad-{index}-{j}") for j in range(2)]})
    candidates = read_json(system.state_path)["cycles"][cycle["id"]]["candidates"]
    result = review(candidates)
    for row in result["reviews"]:
        row["compatible"] = False
    cycle = system.accept(cycle["review_job"], result)
    assert cycle["status"] == "rejected" and cycle["selected"] is None
    assert system.cycle("Find lower memory alternatives")["id"] == "cycle-002"


def test_fleet_runs_all_roles_with_a_real_stdio_process(system):
    worker = system.root / "fixture_worker.py"
    worker.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        f"template = {proposal('placeholder')!r}\n"
        "if request['role'] == 'research':\n"
        "    result = {'approaches': [dict(template, shortcode='fixture-' + "
        "str(request['inputs']['worker_index']) + '-' + str(i)) for i in range(2)]}\n"
        "elif request['role'] == 'review':\n"
        f"    row = {review([proposal('placeholder')])['reviews'][0]!r}\n"
        "    result = {'reviews': [dict(row, shortcode=c['shortcode']) "
        "for c in request['inputs']['candidates']]}\n"
        "else:\n"
        f"    result = {code_result()!r}\n"
        "print(json.dumps(result))\n"
    )
    cycle = system.cycle()
    result = system.fleet(cycle["id"], [sys.executable, str(worker)])
    assert result["status"] == "implemented"
    assert all(job["status"] == "completed" for job in system.status()["jobs"].values())
    assert len(system.status()["jobs"]) == 5


def test_modified_request_is_rejected_before_worker_or_import(system):
    cycle = system.cycle()
    job = cycle["research_jobs"][0]
    path = system.home / "jobs" / job / "request.json"
    value = read_json(path)
    value["instructions"] = "Modified request"
    write_json(path, value)
    with pytest.raises(ContractError, match="modified"):
        system.request(job)
    with pytest.raises(ContractError, match="modified"):
        system.accept(job, {"approaches": [proposal("one"), proposal("two")]})


def test_full_training_mounts_the_validation_selected_recipe(system, tmp_path):
    cycle = advance(system)
    evaluation = remote.prepare(system, cycle["id"], "evaluate", "fixture")
    remote.collect(
        system, evaluation["run_id"], fake_report(system, evaluation, tmp_path / "evaluation")
    )
    system.freeze(cycle["id"])
    held = remote.prepare(system, cycle["id"], "internal_test", "fixture")
    remote.collect(system, held["run_id"], fake_report(system, held, tmp_path / "held"))
    full = remote.prepare(system, cycle["id"], "full_train", "fixture")
    spec = read_json(system.home / "runs" / full["run_id"] / "run.json")
    meta = read_json(system.home / "runs" / full["run_id"] / "kernel-metadata.json")
    assert spec["upstream"]["run_id"] == evaluation["run_id"]
    assert meta["kernel_sources"] == [evaluation["kernel"]]
