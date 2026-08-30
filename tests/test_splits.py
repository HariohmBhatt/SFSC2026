from sfsc.splits import assign_grouped_splits, target_counts


def make_groups(count: int = 707) -> list[dict[str, object]]:
    return [
        {
            "filename": f"image-{index:04d}.jpeg",
            "site": ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")[index % 6],
            "year": 2011 + (index % 12),
            "annotation_set_count": 1 + (index % 3),
            "instance_count_mean": index % 16,
        }
        for index in range(count)
    ]


def test_largest_remainder_counts_match_requested_split() -> None:
    assert target_counts(707, (0.70, 0.15, 0.15)) == {
        "train": 495,
        "validation": 106,
        "test": 106,
    }


def test_grouped_assignment_is_exact_and_deterministic() -> None:
    groups = make_groups()

    first = assign_grouped_splits(groups, seed=2026)
    second = assign_grouped_splits(reversed(groups), seed=2026)

    assert first == second
    assert list(first.values()).count("train") == 495
    assert list(first.values()).count("validation") == 106
    assert list(first.values()).count("test") == 106
    assert len(first) == len(groups)

    for year in range(2011, 2023):
        year_splits = {first[str(group["filename"])] for group in groups if group["year"] == year}
        assert year_splits == {"train", "validation", "test"}


def test_duplicate_source_filename_is_rejected() -> None:
    groups = make_groups(2)
    groups[1]["filename"] = groups[0]["filename"]

    try:
        assign_grouped_splits(groups)
    except ValueError as error:
        assert "exactly once" in str(error)
    else:
        raise AssertionError("Expected duplicate filenames to be rejected")
