"""Validate fleet handoffs before they can change experiment state."""

from __future__ import annotations

import ast
import math
import re
from pathlib import PurePosixPath
from urllib.parse import urlparse


class ContractError(ValueError):
    """An artifact does not satisfy the workflow contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def nonempty(value: object, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def number(value: object, name: str, low: float, high: float = math.inf) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and low <= value <= high,
        f"{name} must be finite and in [{low}, {high}]",
    )
    return float(value)


def shortcode(value: object) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9-]{2,47}", value) is not None,
        "shortcode must be 3–48 lowercase letters, digits or hyphens",
    )
    require(
        value not in {"src", "tests", "system", "splits", "reports", "kaggle", "data"},
        "shortcode is a reserved directory",
    )
    return value


def research(result: dict, minimum: int) -> list[dict]:
    candidates = result.get("approaches")
    require(
        isinstance(candidates, list) and len(candidates) >= minimum,
        f"Each research worker must provide at least {minimum} approaches",
    )
    seen = set()
    for candidate in candidates:
        require(isinstance(candidate, dict), "Each approach must be an object")
        code = shortcode(candidate.get("shortcode"))
        require(code not in seen, f"Duplicate approach: {code}")
        seen.add(code)
        for field in ("title", "mechanism", "data_fit", "runtime_plan", "risks", "experiment"):
            nonempty(candidate.get(field), field)
        sources = candidate.get("sources")
        require(isinstance(sources, list) and len(sources) > 0, "Academic sources are required")
        for source in sources:
            require(isinstance(source, dict), "Source must be an object")
            for field in ("title", "authors", "venue", "claim"):
                nonempty(source.get(field), field)
            number(source.get("year"), "year", 1900, 2100)
            require(
                source.get("kind") in {"journal", "conference", "preprint"},
                "Research sources must be academic papers",
            )
            url = urlparse(nonempty(source.get("url"), "source URL"))
            require(url.scheme == "https" and bool(url.hostname), "Source requires an HTTPS URL")
    return candidates


def rank(result: dict, candidates: list[dict], config: dict) -> list[dict]:
    reviews = result.get("reviews")
    require(isinstance(reviews, list), "reviews must be a list")
    codes = [item.get("shortcode") for item in reviews]
    require(
        len(codes) == len(set(codes)) and set(codes) == {c["shortcode"] for c in candidates},
        "Review must cover every candidate exactly once",
    )
    ranked = []
    weights = config["weights"]
    for item in reviews:
        for field in ("rationale", "assumptions", "uncertainty", "training_roi"):
            nonempty(item.get(field), field)
        require(type(item.get("compatible")) is bool, "compatible must be a boolean")
        potential = number(item.get("pq_potential"), "pq_potential", 0, 1)
        complexity = number(item.get("complexity"), "complexity", 1, 5)
        hours = number(item.get("training_hours"), "training_hours", 0.000001)
        vram = number(item.get("vram_gb"), "vram_gb", 0.000001)
        size = number(item.get("model_mb"), "model_mb", 0.000001)
        utility = {
            "pq_potential": potential,
            "complexity": (5 - complexity) / 4,
            "training_time": max(0, 1 - hours / config["max_training_hours"]),
            "vram": max(0, 1 - vram / config["max_vram_gb"]),
            "model_size": 1 / (1 + size / config["reference_model_mb"]),
        }
        feasible = (
            item["compatible"]
            and vram <= config["max_vram_gb"]
            and hours <= config["max_training_hours"]
        )
        ranked.append(
            {
                **item,
                "feasible": feasible,
                "utility": utility,
                "score": sum(weights[k] * v for k, v in utility.items()),
                "estimate_only": True,
            }
        )
    return sorted(ranked, key=lambda x: (not x["feasible"], -x["score"], x["shortcode"]))


def implementation(result: dict) -> dict[str, str]:
    nonempty(result.get("approach_markdown"), "approach_markdown")
    require(isinstance(result.get("config"), dict), "config must be a JSON object")
    files = result.get("files")
    require(isinstance(files, dict) and "pipeline.py" in files, "files must include pipeline.py")
    require(
        any(p.startswith("tests/test_") and p.endswith(".py") for p in files),
        "Implementation must include tests/test_*.py",
    )
    for name, content in files.items():
        path = PurePosixPath(name)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in name
            and str(path) == name
            and all(not p.startswith(".") for p in path.parts),
            f"Unsafe implementation path: {name}",
        )
        require(path.suffix in {".py", ".md", ".json"}, f"Unsupported source file: {name}")
        require(name not in {"APPROACH.md", "config.json"}, f"Reserved file: {name}")
        nonempty(content, name)
        if path.suffix == ".py":
            ast.parse(content, filename=name)
    tree = ast.parse(files["pipeline.py"])
    require(
        any(isinstance(node, ast.FunctionDef) and node.name == "run" for node in tree.body),
        "pipeline.py must define run(context)",
    )
    return {**files, "APPROACH.md": result["approach_markdown"]}


RESPONSE_CONTRACTS = {
    "research": {
        "approaches": [
            {
                "shortcode": "unique-code",
                "title": "text",
                "mechanism": "text",
                "data_fit": "text",
                "runtime_plan": "text",
                "risks": "text",
                "experiment": "text",
                "sources": [
                    {
                        "kind": "journal|conference|preprint",
                        "title": "text",
                        "authors": "text",
                        "year": 2026,
                        "venue": "text",
                        "url": "https://...",
                        "claim": "specific supporting claim",
                    }
                ],
            }
        ]
    },
    "review": {
        "reviews": [
            {
                "shortcode": "one per candidate",
                "pq_potential": 0.5,
                "complexity": 3,
                "training_hours": 4.0,
                "vram_gb": 10.0,
                "model_mb": 100.0,
                "compatible": True,
                "rationale": "text",
                "assumptions": "text",
                "uncertainty": "text",
                "training_roi": "text",
            }
        ]
    },
    "implementation": {
        "approach_markdown": "# Approach\n...",
        "config": {},
        "files": {"pipeline.py": "def run(context): ...", "tests/test_pipeline.py": "..."},
    },
}
