"""Persistent three-fleet workflow, with immutable experiment handoffs."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sfsc.contracts import RESPONSE_CONTRACTS, implementation, number, rank, require, research

CONTRACT_FILES = (
    "DATA.md",
    "EVAL.md",
    "GPU.md",
    "splits/split_manifest.csv",
    "src/sfsc/evaluation.py",
    "system/config.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def now() -> str:
    return datetime.now(UTC).isoformat()


class System:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.home = self.root / ".sfsc"
        self.state_path = self.home / "state.json"

    @contextmanager
    def transaction(self):
        self.home.mkdir(parents=True, exist_ok=True)
        with (self.home / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read_json(self.state_path) if self.state_path.exists() else None
            yield state
            if state is not None:
                write_json(self.state_path, state)

    def check(self, state: dict) -> None:
        require(state is not None, "Run sfsc-system init first")
        for name, expected in state["contract"].items():
            require(
                (self.root / name).is_file() and digest(self.root / name) == expected,
                f"Fixed contract changed: {name}. Restore it before continuing this system.",
            )

    def init(self) -> dict:
        with self.transaction() as state:
            if state is not None:
                self.check(state)
                return state
            for name in CONTRACT_FILES:
                require((self.root / name).is_file(), f"Missing initialization input: {name}")
            config = read_json(self.root / "system/config.json")
            for key, low in (
                ("research_workers", 2),
                ("approaches_per_worker", 2),
                ("worker_timeout_seconds", 1),
            ):
                number(config.get(key), key, low, 100000)
                require(type(config[key]) is int, f"{key} must be an integer")
            for key in ("max_vram_gb", "max_training_hours", "reference_model_mb"):
                number(config.get(key), key, 0.000001)
            number(config.get("seed"), "seed", 0, 2**31 - 1)
            require(type(config["seed"]) is int, "seed must be an integer")
            expected = {"pq_potential", "complexity", "training_time", "vram", "model_size"}
            require(set(config.get("weights", {})) == expected, "Five ranking weights required")
            for key, value in config["weights"].items():
                number(value, key, 0, 1)
            require(math.isclose(sum(config["weights"].values()), 1), "Weights must sum to 1")
            with (self.root / "splits/split_manifest.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            counts = Counter(row["split"] for row in rows)
            require(set(counts) == {"train", "validation", "test"}, "Three splits required")
            require(len({r["filename"] for r in rows}) == len(rows), "Duplicate source filenames")
            for row in rows:
                require(Path(row["filename"]).name == row["filename"], "Unsafe source filename")
            require(
                all(
                    abs(counts[s] / len(rows) - f) <= max(0.01, 1 / len(rows))
                    for s, f in (("train", 0.70), ("validation", 0.15), ("test", 0.15))
                ),
                "Manifest must use the fixed 70/15/15 split",
            )
            state = {
                "version": 1,
                "created_at": now(),
                "config": config,
                "contract": {name: digest(self.root / name) for name in CONTRACT_FILES},
                "split_counts": dict(counts),
                "cycles": {},
                "jobs": {},
                "runs": {},
                "frozen_cycle": None,
                "events": [],
            }
            write_json(self.state_path, state)
            return state

    def event(self, state: dict, action: str, **details) -> None:
        state["events"].append({"at": now(), "action": action, **details})

    def job(self, state: dict, cycle_id: str, role: str, **inputs) -> str:
        job_id = f"{cycle_id}-{role}-{uuid.uuid4().hex[:8]}"
        documents = {
            name: (self.root / name).read_text() for name in ("DATA.md", "EVAL.md", "GPU.md")
        }
        if role == "implementation":
            documents["system/RUNTIME.md"] = (self.root / "system/RUNTIME.md").read_text()
            for name in ("dataset.py", "evaluation.py"):
                documents[f"src/sfsc/{name}"] = (self.root / "src/sfsc" / name).read_text()
        request = {
            "job_id": job_id,
            "cycle_id": cycle_id,
            "role": role,
            "instructions": (self.root / f"system/prompts/{role}.md").read_text(),
            "documents": documents,
            "contract_hashes": state["contract"],
            "config": state["config"],
            "inputs": inputs,
            "response_contract": RESPONSE_CONTRACTS[role],
        }
        write_json(self.home / "jobs" / job_id / "request.json", request)
        state["jobs"][job_id] = {
            "role": role,
            "cycle_id": cycle_id,
            "status": "pending",
            "request_hash": object_hash(request),
        }
        return job_id

    def cycle(self, feedback: str = "") -> dict:
        with self.transaction() as state:
            self.check(state)
            require(state["frozen_cycle"] is None, "Pipeline frozen; model selection is closed")
            require(
                all(c["status"] in {"evaluated", "rejected"} for c in state["cycles"].values()),
                "Finish or reject the active cycle before starting another",
            )
            cycle_id = f"cycle-{len(state['cycles']) + 1:03d}"
            cycle = {
                "id": cycle_id,
                "status": "research",
                "feedback": feedback,
                "research_jobs": [],
                "selected": None,
                "ranking": [],
            }
            state["cycles"][cycle_id] = cycle
            lanes = [
                "semantic probabilities with instance separation",
                "instance-aware segmentation",
                "thin-structure and multi-scale segmentation",
            ]
            prior = {
                key: {
                    "selected": c["selected"],
                    "ranking": c["ranking"],
                    "validation": c.get("validation"),
                    "feedback": c["feedback"],
                }
                for key, c in state["cycles"].items()
                if key != cycle_id
            }
            for i in range(state["config"]["research_workers"]):
                cycle["research_jobs"].append(
                    self.job(
                        state,
                        cycle_id,
                        "research",
                        lane=lanes[i % len(lanes)],
                        worker_index=i,
                        feedback=feedback,
                        previous=prior,
                        approaches_per_worker=state["config"]["approaches_per_worker"],
                    )
                )
            self.event(state, "cycle_created", cycle_id=cycle_id)
            return cycle

    def request(self, job_id: str) -> dict:
        with self.transaction() as state:
            self.check(state)
            require(job_id in state["jobs"], "Unknown job")
            request = read_json(self.home / "jobs" / job_id / "request.json")
            require(
                object_hash(request) == state["jobs"][job_id]["request_hash"],
                "Job request was modified",
            )
            return request

    def accept(self, job_id: str, result: dict) -> dict:
        require(isinstance(result, dict), "Worker result must be a JSON object")
        object_hash(result)  # Reject non-JSON values before writing implementation files.
        with self.transaction() as state:
            self.check(state)
            require(job_id in state["jobs"], "Unknown job")
            job = state["jobs"][job_id]
            cycle = state["cycles"][job["cycle_id"]]
            require(job["status"] in {"pending", "running", "failed"}, "Job already accepted")
            require(cycle["status"] == job["role"], "Job belongs to an inactive phase")
            request = read_json(self.home / "jobs" / job_id / "request.json")
            require(object_hash(request) == job["request_hash"], "Job request was modified")
            role = job["role"]
            if role == "research":
                candidates = research(result, state["config"]["approaches_per_worker"])
                existing = {c["shortcode"] for c in cycle.get("candidates", [])}
                require(
                    all(not (self.root / c["shortcode"]).exists() for c in candidates),
                    "Research short codes must use new directories for each iteration",
                )
                require(
                    not existing.intersection(c["shortcode"] for c in candidates),
                    "Short codes must be unique across research workers",
                )
                cycle.setdefault("candidates", []).extend(candidates)
            elif role == "review":
                cycle["ranking"] = rank(result, cycle["candidates"], state["config"])
                feasible = [r for r in cycle["ranking"] if r["feasible"]]
                if feasible:
                    cycle["selected"] = feasible[0]["shortcode"]
                    cycle["status"] = "implementation"
                    selected = next(
                        c for c in cycle["candidates"] if c["shortcode"] == cycle["selected"]
                    )
                    cycle["implementation_job"] = self.job(
                        state,
                        cycle["id"],
                        "implementation",
                        selected=selected,
                        review=feasible[0],
                        ranking=cycle["ranking"],
                    )
                else:
                    cycle["status"] = "rejected"
            else:
                files = implementation(result)
                destination = self.root / cycle["selected"]
                require(not destination.exists(), f"Approach directory exists: {destination}")
                staging = Path(tempfile.mkdtemp(dir=self.home, prefix="implementation-"))
                try:
                    for name, content in files.items():
                        target = staging / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(content, encoding="utf-8")
                    write_json(staging / "config.json", result["config"])
                    os.rename(staging, destination)
                finally:
                    if staging.exists():
                        import shutil

                        shutil.rmtree(staging)
                cycle["implementation_hashes"] = {
                    str(p.relative_to(destination)): digest(p)
                    for p in destination.rglob("*")
                    if p.is_file()
                }
                cycle["status"] = "implemented"
            job["status"] = "completed"
            job["result_hash"] = object_hash(result)
            write_json(self.home / "jobs" / job_id / "result.json", result)
            if role == "research" and all(
                state["jobs"][j]["status"] == "completed" for j in cycle["research_jobs"]
            ):
                cycle["status"] = "review"
                cycle["review_job"] = self.job(
                    state, cycle["id"], "review", candidates=cycle["candidates"]
                )
            self.event(state, "job_accepted", job_id=job_id, cycle_status=cycle["status"])
            return cycle

    def work(self, job_id: str, command: list[str] | None = None) -> dict:
        request = self.request(job_id)
        with self.transaction() as state:
            job = state["jobs"][job_id]
            command = command or state["config"]["workers"].get(job["role"])
            require(
                isinstance(command, list)
                and len(command) > 0
                and all(isinstance(v, str) for v in command),
                "Configure a worker argv in system/config.json before init, or pass --command",
            )
            require(job["status"] in {"pending", "failed"}, "Job is already running or complete")
            job["status"] = "running"
            job["started_at"] = now()
        try:
            result = subprocess.run(
                command,
                input=json.dumps(request),
                text=True,
                capture_output=True,
                cwd=self.root,
                timeout=request["config"]["worker_timeout_seconds"],
                check=True,
            )
            (self.home / "jobs" / job_id / "worker.log").write_text(result.stderr)
            return self.accept(job_id, json.loads(result.stdout))
        except Exception as error:
            with self.transaction() as state:
                job = state["jobs"][job_id]
                if job["status"] != "completed":
                    job.update(status="failed", error=str(error))
                    self.event(state, "job_failed", job_id=job_id, error=str(error))
            raise

    def fleet(self, cycle_id: str, command: list[str] | None = None) -> dict:
        """Run bounded parallel research, then review, then implementation."""
        for _ in range(3):
            state = self.status()
            require(cycle_id in state["cycles"], "Unknown cycle")
            role = state["cycles"][cycle_id]["status"]
            if role not in {"research", "review", "implementation"}:
                return state["cycles"][cycle_id]
            jobs = [
                key
                for key, job in state["jobs"].items()
                if job["cycle_id"] == cycle_id
                and job["role"] == role
                and job["status"] in {"pending", "failed"}
            ]
            require(bool(jobs), "Workers are already running; inspect status")
            errors = {}
            with ThreadPoolExecutor(max_workers=min(len(jobs), 3)) as pool:
                futures = {pool.submit(self.work, job, command): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        errors[futures[future]] = str(error)
            require(not errors, f"Fleet jobs failed; completed work is saved: {errors}")
        return self.status()["cycles"][cycle_id]

    def recover_job(self, job_id: str, reason: str) -> dict:
        """Operator recovery after confirming an interrupted worker has stopped."""
        with self.transaction() as state:
            self.check(state)
            require(job_id in state["jobs"] and bool(reason.strip()), "Job and reason required")
            job = state["jobs"][job_id]
            require(job["status"] == "running", "Only interrupted running jobs need recovery")
            job.update(status="failed", error=reason)
            self.event(state, "worker_recovered", job_id=job_id, reason=reason)
            return job

    def check_implementation(self, cycle: dict) -> None:
        for name, expected in cycle["implementation_hashes"].items():
            path = self.root / cycle["selected"] / name
            require(
                path.is_file() and digest(path) == expected,
                f"Accepted implementation changed: {path}; start a new cycle for revisions",
            )

    def freeze(self, cycle_id: str) -> dict:
        with self.transaction() as state:
            self.check(state)
            require(cycle_id in state["cycles"], "Unknown cycle")
            cycle = state["cycles"][cycle_id]
            require(
                state["frozen_cycle"] is None and cycle["status"] == "evaluated",
                "Freeze requires an evaluated cycle and an unused internal holdout",
            )
            require(
                all(c["status"] in {"evaluated", "rejected"} for c in state["cycles"].values()),
                "Finish all active model-selection work before freezing",
            )
            self.check_implementation(cycle)
            state["frozen_cycle"] = cycle_id
            cycle["status"] = "frozen"
            self.event(state, "pipeline_frozen", cycle_id=cycle_id)
            return cycle

    def reject(self, cycle_id: str, reason: str) -> dict:
        with self.transaction() as state:
            self.check(state)
            require(bool(reason.strip()), "A reason is required")
            require(cycle_id in state["cycles"], "Unknown cycle")
            cycle = state["cycles"][cycle_id]
            require(state["frozen_cycle"] is None, "Cannot reopen selection after freezing")
            require(
                not any(
                    j["status"] == "running" and j["cycle_id"] == cycle_id
                    for j in state["jobs"].values()
                ),
                "Wait for running workers",
            )
            require(
                not any(
                    r["status"] in {"prepared", "launching", "submitted"}
                    and r["cycle_id"] == cycle_id
                    for r in state["runs"].values()
                ),
                "Resolve outstanding GPU runs first",
            )
            cycle.update(status="rejected", feedback=reason)
            self.event(state, "cycle_rejected", cycle_id=cycle_id, reason=reason)
            return cycle

    def status(self) -> dict:
        with self.transaction() as state:
            self.check(state)
            return {
                "root": str(self.root),
                "split_counts": state["split_counts"],
                "frozen_cycle": state["frozen_cycle"],
                "cycles": {
                    k: {a: v for a, v in c.items() if a not in {"candidates"}}
                    for k, c in state["cycles"].items()
                },
                "jobs": state["jobs"],
                "runs": state["runs"],
            }
