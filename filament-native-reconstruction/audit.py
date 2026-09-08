"""Audit native-pilot artifacts for failures that Kaggle may still mark complete."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def audit_payload(report: dict, candidates: dict[str, list[dict]]) -> list[str]:
    """Return concrete integrity failures in a completed pilot payload."""
    issues: list[str] = []
    training = report.get("training", {})
    if training.get("nonfinite_batches", 0):
        issues.append(f"training recorded {training['nonfinite_batches']} non-finite batches")
    for name in ("last_loss", "minimum_loss", "maximum_loss"):
        value = training.get(name)
        if value is not None and not math.isfinite(float(value)):
            issues.append(f"training {name} is non-finite")

    rows = [candidate for image_rows in candidates.values() for candidate in image_rows]
    if not rows:
        issues.append("candidate cache is empty")
        return issues
    native_areas = [int(candidate.get("native_area", 0)) for candidate in rows]
    nonempty = sum(area > 0 for area in native_areas)
    if nonempty == 0:
        issues.append("all reconstructed candidate masks are empty")
    elif nonempty / len(rows) < 0.05:
        issues.append(f"only {nonempty}/{len(rows)} reconstructed masks are non-empty")

    assessment = report.get("assessment", {})
    for name in ("native", "native82"):
        pq = assessment.get(name, {}).get("pq")
        if pq is not None and not math.isfinite(float(pq)):
            issues.append(f"assessment {name} PQ is non-finite")
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
