from pathlib import Path

import pytest

from sfsc.contracts import ContractError
from sfsc.kaggle_runtime import find_dataset_root


def make_dataset(root: Path):
    (root / "train/train_images").mkdir(parents=True)
    (root / "test/test_images").mkdir(parents=True)
    (root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text("{}")


@pytest.mark.parametrize("nested", [False, True])
def test_competition_mount_supports_flat_and_actual_nested_layout(tmp_path, nested):
    root = tmp_path / "MAGFiLO_1.0_Kaggle_2026" if nested else tmp_path
    make_dataset(root)
    assert find_dataset_root(tmp_path) == root


def test_dataset_discovery_rejects_ambiguous_and_incomplete_mounts(tmp_path):
    with pytest.raises(ContractError, match="found 0"):
        find_dataset_root(tmp_path)
    make_dataset(tmp_path / "first")
    make_dataset(tmp_path / "second")
    with pytest.raises(ContractError, match="found 2"):
        find_dataset_root(tmp_path)
