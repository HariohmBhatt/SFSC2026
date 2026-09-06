from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from sfsc.contracts import ContractError
from sfsc.dataset import Dataset, evaluate, label_masks, prediction_files, submission


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "train/train_images").mkdir(parents=True)
    (tmp_path / "test/test_images").mkdir(parents=True)
    images = [{"id": i, "file_name": "same.jpeg", "height": 8, "width": 8} for i in (1, 2)]
    annotations = [{"id": 1, "image_id": 1, "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]]}]
    annotation = {"images": images, "annotations": annotations}
    (tmp_path / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(
        json.dumps(annotation)
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,split,site,year,annotation_ids\nsame.jpeg,validation,Bh,2020,1;2\n"
    )
    Image.new("L", (8, 8)).save(tmp_path / "train/train_images/same.jpeg")
    Image.new("L", (8, 8)).save(tmp_path / "test/test_images/competition.jpeg")
    return Dataset(tmp_path, manifest)


def test_evaluation_scores_each_annotator_and_aggregates_global_counts(dataset, tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    np.save(predictions / "same.npy", dataset.truth(dataset.images[0])[0].astype(np.uint16))
    result = evaluate(dataset, predictions, "validation")
    assert result["images"] == 1 and result["annotation_records"] == 2
    assert result["true_positives"] == 1 and result["false_positives"] == 1
    assert result["pq"] == pytest.approx(2 / 3)
    assert result["groups"]["multiplicity:2"]["pq"] == result["pq"]


def test_submission_uses_compressed_coco_rle_and_exact_image_stem(dataset, tmp_path):
    folder = tmp_path / "predictions"
    folder.mkdir()
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[1:3, 2:5] = 7
    labels[5:7, 5:7] = 19
    np.save(folder / "competition.npy", labels)
    path = tmp_path / "submission.csv"
    result = submission(dataset.root, folder, path)
    assert result["instances"] == 2
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["filament_id"] for row in rows] == ["competition_1", "competition_2"]
    for row, value in zip(rows, (7, 19), strict=True):
        decoded = coco_mask.decode(
            {"size": [8, 8], "counts": row["segmentation_rle"].encode("ascii")}
        )
        assert np.array_equal(decoded.astype(bool), labels == value)


def test_empty_image_is_explicit_and_extra_predictions_are_rejected(dataset, tmp_path):
    folder = tmp_path / "predictions"
    folder.mkdir()
    np.save(folder / "competition.npy", np.zeros((8, 8), dtype=np.uint16))
    report = submission(dataset.root, folder, tmp_path / "submission.csv")
    assert report["empty_images"] == 1 and report["instances"] == 0
    np.save(folder / "unknown.npy", np.zeros((8, 8), dtype=np.uint16))
    with pytest.raises(ContractError, match="coverage mismatch"):
        prediction_files(folder, ["competition.jpeg"])


@pytest.mark.parametrize(
    "labels",
    [
        np.ones((4, 4), dtype=np.uint16),
        np.ones((8, 8)) / 2,
        -np.ones((8, 8), dtype=int),
        np.ones((8, 8, 1), dtype=int),
    ],
)
def test_invalid_label_maps_rejected(labels):
    with pytest.raises(ContractError):
        label_masks(labels, (8, 8))


def test_annotation_id_drift_rejected(dataset):
    path = dataset.root / "manifest.csv"
    path.write_text(path.read_text().replace("1;2", "1;3"))
    with pytest.raises(ContractError, match="Annotation set drift"):
        Dataset(dataset.root, path)
