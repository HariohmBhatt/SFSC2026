"""Audit native-pilot artifacts for failures that Kaggle may still mark complete."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_number(value) -> bool:
    """Return whether an artifact value is a finite real number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonnegative_integer(value) -> int | None:
    """Parse a non-negative integer without accepting lossy conversions."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def audit_payload(report: dict, candidates: dict[str, list[dict]]) -> list[str]:
    """Return concrete integrity failures in a completed pilot payload."""
    if not isinstance(report, dict):
        return ["report payload is not an object"]
    issues: list[str] = []
    training = report.get("training", {})
    if not isinstance(training, dict):
        training = {}
        issues.append("training payload is not an object")
    required = report.get("status") == "complete"
    nonfinite_batches = nonnegative_integer(training.get("nonfinite_batches"))
    if required and nonfinite_batches is None:
        issues.append("training nonfinite_batches is missing or invalid")
    elif nonfinite_batches:
        issues.append(f"training recorded {nonfinite_batches} non-finite batches")
    for name in ("last_loss", "minimum_loss", "maximum_loss"):
        value = training.get(name)
        if (required or value is not None) and not finite_number(value):
            issues.append(f"training {name} is missing or non-finite")

    if not isinstance(candidates, dict) or any(
        not isinstance(image_rows, list) for image_rows in candidates.values()
    ):
        issues.append("candidate cache has invalid structure")
        rows = []
    else:
        rows = [candidate for image_rows in candidates.values() for candidate in image_rows]
    if not rows:
        issues.append("candidate cache is empty")
        return issues
    native_areas = []
    for index, candidate in enumerate(rows):
        if not isinstance(candidate, dict):
            issues.append(f"candidate {index} is not an object")
            native_areas.append(0)
            continue
        area = nonnegative_integer(candidate.get("native_area"))
        if area is None:
            issues.append(f"candidate {index} native_area is invalid")
            area = 0
        native_areas.append(area)
    nonempty = sum(area > 0 for area in native_areas)
    if nonempty == 0:
        issues.append("all reconstructed candidate masks are empty")
    elif nonempty / len(rows) < 0.05:
        issues.append(f"only {nonempty}/{len(rows)} reconstructed masks are non-empty")

    assessment = report.get("assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
        issues.append("assessment payload is not an object")
    for name in ("native", "native82"):
        metrics = assessment.get(name, {})
        pq = metrics.get("pq") if isinstance(metrics, dict) else None
        if (required or pq is not None) and not finite_number(pq):
            issues.append(f"assessment {name} PQ is missing or non-finite")
    if report.get("status") == "complete" and issues:
        issues.append("run was marked complete despite integrity failures")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("candidates", type=Path)
    args = parser.parse_args()
    issues = audit_payload(
        json.loads(args.report.read_text()), json.loads(args.candidates.read_text())
    )
    print(json.dumps({"valid": not issues, "issues": issues}, indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
