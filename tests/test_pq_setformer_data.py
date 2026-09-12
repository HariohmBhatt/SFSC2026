"""Behavioral tests for coherent annotation sampling and rasterization."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "filament-pq-setformer/data.py"
spec = importlib.util.spec_from_file_location("pq_setformer_data", MODULE)
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)


def test_discover_dataset_root_accepts_private_train_only_mount(tmp_path) -> None:
    root = tmp_path / "sfsc2026-private-data"
    (root / "train_images").mkdir(parents=True)
    (root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text("{}")

    assert data.discover_dataset_root(tmp_path) == root.resolve()


def test_training_dataset_snapshot_detects_same_size_image_replacement(tmp_path) -> None:
    (tmp_path / "train_images").mkdir()
    annotation = tmp_path / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    image = tmp_path / "train_images/image.jpeg"
    annotation.write_bytes(b"annotations")
    image.write_bytes(b"first-image")

    first = data.training_dataset_snapshot(tmp_path)
    image.write_bytes(b"other-image")
    second = data.training_dataset_snapshot(tmp_path)

    assert first["files"] == second["files"] == 2
    assert first["bytes"] == second["bytes"] == 22
    assert first["annotation_sha256"] == second["annotation_sha256"]
    assert first["sha256"] != second["sha256"]


def test_real_index_keeps_complete_records_and_audits_query_capacity() -> None:
    annotation = ROOT / "SFSCDataset/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    index = data.load_annotation_index(annotation)

    assert len(index.filenames) == 707
    assert index.record_count == 1_154
    assert index.maximum_instances == 26
    assert len(index.records_by_file["20111116063134Lh.jpeg"]) == 3


def test_record_selection_is_deterministic_and_never_combines_annotators() -> None:
    records = (
        data.AnnotationRecord("a", ({"id": "a1"},)),
        data.AnnotationRecord("b", ({"id": "b1"}, {"id": "b2"})),
        data.AnnotationRecord("c", ({"id": "c1"},)),
    )
    first = data.select_annotation_record(records, "image.jpeg", presentation=7, seed=2026)
    second = data.select_annotation_record(records, "image.jpeg", presentation=7, seed=2026)

    assert first is second
    assert first in records


def test_native_rasterization_then_nearest_resize_preserves_instances_without_unioning() -> None:
    record = data.AnnotationRecord(
        "record",
        (
            {"segmentation": [[0, 0, 8, 0, 8, 8, 0, 8, 0, 0]]},
            {"segmentation": [[12, 12, 16, 12, 16, 16, 12, 16, 12, 12]]},
        ),
    )

    masks = data.rasterize_record(record, native_size=16, output_size=8)

    assert masks.shape == (2, 8, 8)
    assert masks.dtype == np.bool_
    assert masks[0].any() and masks[1].any()
    assert not np.logical_and(masks[0], masks[1]).any()


def test_native_rasterization_preserves_a_thin_mask_that_coordinate_scaling_loses() -> None:
    # This one-native-pixel stripe lands in a different 4x4 output cell from
    # its neighbour.  Rasterizing first is the contract; scaling vertices first
    # can change which output pixel owns the stripe at polygon boundaries.
    record = data.AnnotationRecord(
        "thin",
        ({"segmentation": [[2, 0, 3, 0, 3, 16, 2, 16, 2, 0]]},),
    )

    masks = data.rasterize_record(record, native_size=16, output_size=4)

    assert masks.shape == (1, 4, 4)
    assert masks[0, :, 0].all()
    assert masks[0].sum() == 4


def test_virtual_epoch_order_is_reproducible_and_covers_every_filename() -> None:
    names = [f"image-{index}.jpeg" for index in range(7)]
    first = [data.filename_for_presentation(names, index, seed=42) for index in range(14)]
    second = [data.filename_for_presentation(names, index, seed=42) for index in range(14)]

    assert first == second
    assert set(first[:7]) == set(names)
    assert set(first[7:]) == set(names)
    assert first[:7] != first[7:]


def test_annotation_index_serialization_is_small_and_hash_stable(tmp_path) -> None:
    payload = {
        "images": [{"id": "r1", "file_name": "a.jpeg"}],
        "annotations": [{"id": "m1", "image_id": "r1", "segmentation": [[0, 0, 1, 0, 1, 1]]}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload))

    first = data.load_annotation_index(path)
    second = data.load_annotation_index(path)

    assert first.sha256 == second.sha256
    assert first.maximum_instances == 1
