"""Deterministic, source-image-grouped dataset splitting."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SPLITS = ("train", "validation", "test")


def target_counts(total: int, fractions: tuple[float, float, float]) -> dict[str, int]:
    """Allocate integer capacities with the largest-remainder method."""
    if total < 0 or len(fractions) != len(SPLITS):
        raise ValueError("Expected a non-negative total and exactly three fractions")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1")
    if any(fraction < 0 for fraction in fractions):
        raise ValueError("Split fractions must be non-negative")

    raw = [total * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remainder_order = sorted(
        range(len(raw)), key=lambda index: (raw[index] - counts[index], -index), reverse=True
    )
    for index in remainder_order[: total - sum(counts)]:
        counts[index] += 1
    return dict(zip(SPLITS, counts, strict=True))


def _stable_rank(seed: int, filename: str) -> str:
    return hashlib.sha256(f"{seed}:{filename}".encode()).hexdigest()


def _quota_move_cost(
    year: str,
    *,
    rows_by_year: Mapping[str, Sequence[Mapping[str, Any]]],
    year_quotas: Mapping[str, Mapping[str, int]],
    fraction_by_split: Mapping[str, float],
    excess: str,
    deficit: str,
) -> tuple[float, str]:
    size = len(rows_by_year[year])
    excess_target = size * fraction_by_split[excess]
    deficit_target = size * fraction_by_split[deficit]
    excess_count = year_quotas[year][excess]
    deficit_count = year_quotas[year][deficit]
    before = (excess_count - excess_target) ** 2 + (deficit_count - deficit_target) ** 2
    after = (excess_count - 1 - excess_target) ** 2 + (deficit_count + 1 - deficit_target) ** 2
    return after - before, year


def _incremental_cost(
    split: str,
    *,
    item_features: Sequence[tuple[str, float]],
    feature_totals: Mapping[str, int],
    local_fractions: Mapping[str, float],
    feature_counts: Mapping[str, Counter[str]],
    quotas: Mapping[str, int],
    split_sizes: Mapping[str, int],
) -> tuple[float, str]:
    cost = 0.0
    for feature, weight in item_features:
        target = feature_totals[feature] * local_fractions[split]
        current = feature_counts[feature][split]
        cost += weight * ((current + 1 - target) ** 2 - (current - target) ** 2)
    remaining_ratio = (quotas[split] - split_sizes[split]) / max(quotas[split], 1)
    return cost - 0.01 * remaining_ratio, split


def assign_grouped_splits(
    groups: Iterable[Mapping[str, Any]],
    *,
    seed: int = 2026,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, str]:
    """Assign one split per source filename while balancing site and year.

    The global capacities are exact. A deterministic greedy objective minimizes
    squared deviation from the requested site, year, site-year, annotation-set,
    and instance-density distributions.
    """
    rows = [dict(group) for group in groups]
    filenames = [str(row["filename"]) for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Each source filename must appear exactly once")

    capacities = target_counts(len(rows), fractions)
    fraction_by_split = dict(zip(SPLITS, fractions, strict=True))

    def features(row: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
        annotation_sets = int(row.get("annotation_set_count", 1))
        mean_instances = float(row.get("instance_count_mean", 0.0))
        density = "low" if mean_instances < 5 else "medium" if mean_instances < 10 else "high"
        site = str(row["site"])
        year = str(row["year"])
        return (
            (f"year:{year}", 6.0),
            (f"site:{site}", 2.0),
            (f"site-year:{site}-{year}", 1.0),
            (f"annotation-sets:{annotation_sets}", 0.75),
            (f"instance-density:{density}", 0.5),
        )

    rows_by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_year[str(row["year"])].append(row)

    # Give every year its own proportional quota, then make the one or two
    # rounding corrections needed for exact global capacities.
    year_quotas = {
        year: target_counts(len(year_rows), fractions) for year, year_rows in rows_by_year.items()
    }
    quota_totals = Counter(
        {split: sum(quota[split] for quota in year_quotas.values()) for split in SPLITS}
    )
    while dict(quota_totals) != capacities:
        excess = next(split for split in SPLITS if quota_totals[split] > capacities[split])
        deficit = next(split for split in SPLITS if quota_totals[split] < capacities[split])

        year = min(
            (year for year in rows_by_year if year_quotas[year][excess] > 0),
            key=lambda candidate: _quota_move_cost(
                candidate,
                rows_by_year=rows_by_year,
                year_quotas=year_quotas,
                fraction_by_split=fraction_by_split,
                excess=excess,
                deficit=deficit,
            ),
        )
        year_quotas[year][excess] -= 1
        year_quotas[year][deficit] += 1
        quota_totals[excess] -= 1
        quota_totals[deficit] += 1

    split_sizes: Counter[str] = Counter()
    assignments: dict[str, str] = {}

    for year in sorted(rows_by_year):
        year_rows = sorted(
            rows_by_year[year], key=lambda row: _stable_rank(seed, str(row["filename"]))
        )
        quotas = year_quotas[year]
        local_fractions = {split: quotas[split] / len(year_rows) for split in SPLITS}
        row_features = {
            str(row["filename"]): tuple(
                (feature, weight)
                for feature, weight in features(row)
                if not feature.startswith("year:")
            )
            for row in year_rows
        }
        feature_totals: Counter[str] = Counter(
            feature for row in year_rows for feature, _ in row_features[str(row["filename"])]
        )
        feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
        year_split_sizes: Counter[str] = Counter()

        for row in year_rows:
            filename = str(row["filename"])
            candidates = [split for split in SPLITS if year_split_sizes[split] < quotas[split]]

            selected = min(
                candidates,
                key=lambda candidate: _incremental_cost(
                    candidate,
                    item_features=row_features[filename],
                    feature_totals=feature_totals,
                    local_fractions=local_fractions,
                    feature_counts=feature_counts,
                    quotas=quotas,
                    split_sizes=year_split_sizes,
                ),
            )
            assignments[filename] = selected
            split_sizes[selected] += 1
            year_split_sizes[selected] += 1
            for feature, _ in row_features[filename]:
                feature_counts[feature][selected] += 1

    if dict(split_sizes) != capacities:
        raise AssertionError(f"Split capacity invariant failed: {split_sizes} != {capacities}")
    return assignments
