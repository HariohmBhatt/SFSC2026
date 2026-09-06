"""Unit tests for the competition watch tool."""

from __future__ import annotations

from sfsc import watch

SAMPLE_SOURCE = """
import numpy as np
import segmentation_models_pytorch as smp
import torch

CFG = {
    'seed': 42,
    'img_size': (512, 512),
    'batch_size': 8,
    'epochs': 5,
    'lr': 3e-4,
}

model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=1)
criterion = DiceBCELoss()

ann_entry = self.annotations.get(filename, []) if self.annotations else []
mask = create_mask_from_annotation(ann_entry, (2048, 2048))

if filament_count == 0:
    empty_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    empty_rle_str = mask_to_coco_rle(empty_mask)
"""


def test_tag_extraction() -> None:
    tags = watch.extract_tags(SAMPLE_SOURCE)
    assert "u-net" in tags
    assert "smp" in tags
    assert "pytorch" in tags
    assert "imagenet weights" in tags
    assert "bce loss" in tags
    assert "yolo" not in tags


def test_fact_extraction() -> None:
    facts = watch.extract_facts(SAMPLE_SOURCE)
    assert facts["epochs"] == "5"
    assert facts["image_size"] == "512"
    assert facts["batch_size"] == "8"
    assert facts["model"] == "smp.Unet"


def test_broken_annotation_flag() -> None:
    flags = watch.extract_flags(SAMPLE_SOURCE)
    assert "broken-annotation-lookup" in flags
    assert "empty-submission-path" in flags


def test_flag_patterns_absent_on_clean_source() -> None:
    clean = "model = smp.Unet(encoder_name='resnet34')\nthreshold = 0.5\n"
    assert watch.extract_flags(clean) == []


def test_extract_excerpts_prioritises_method_lines() -> None:
    excerpts = watch.extract_excerpts(SAMPLE_SOURCE, limit=5)
    assert 0 < len(excerpts) <= 5
    assert any("smp.Unet" in excerpt["text"] for excerpt in excerpts)
    assert all(excerpt["line"] >= 1 for excerpt in excerpts)


def test_external_inputs_filter_competition_slug() -> None:
    text = "/kaggle/input/filament-segmentation-2026/MAGFiLO\n/kaggle/input/some-model/weights.pt\n"
    assert watch.extract_external_inputs(text, "filament-segmentation-2026") == ["some-model"]


def test_parse_cli_csv_strips_preamble() -> None:
    text = "Next Page Token = abc123\nteamId,teamName,score\n1,alpha,0.56\n2,beta,0.55\n"
    rows = watch.parse_cli_csv(text)
    assert [row["teamName"] for row in rows] == ["alpha", "beta"]


def test_parse_cli_csv_empty() -> None:
    assert watch.parse_cli_csv("") == []
    assert watch.parse_cli_csv("Next Page Token = x\n\n") == []


def test_parse_topic_messages() -> None:
    text = (
        "id,title,authorName,commentCount,votes,postDate,id,authorName,postDate,votes\n"
        "728474,T,gr,6,18,2026-07-23,id,gr,2026-07-23,18\n"
        "id,authorName,votes,postDate,id,authorName,postDate,votes,content\n"
        '3503995,Reman,0,2026-07-26,3503995,Reman,2026-07-26,0,"my score jumped to 0.87"\n'
    )
    messages = watch.parse_topic_messages(text)
    assert len(messages) == 1
    assert messages[0]["authorName"] == "Reman"
    assert "0.87" in messages[0]["content"]


def test_notebook_code_cells_flattens_ipynb() -> None:
    import json

    notebook = {
        "cells": [
            {"cell_type": "code", "source": ["import torch\n", "print(torch.__version__)"]},
            {"cell_type": "markdown", "source": ["# Baseline"]},
        ]
    }
    text = watch.notebook_code_cells(json.dumps(notebook))
    assert "import torch" in text
    assert "# Baseline" in text


def test_votes_delta() -> None:
    record = {"votes_history": [["t1", 10], ["t2", 12]]}
    assert watch.votes_delta(record) == 2
    assert watch.votes_delta({"votes_history": [["t1", 5]]}) == 0


def test_analyze_source_shape() -> None:
    analysis = watch.analyze_source(SAMPLE_SOURCE, "filament-segmentation-2026")
    assert analysis["lines"] > 10
    assert "broken-annotation-lookup" in analysis["flags"]
    assert "resnet backbone" in analysis["tags"]


def test_flag_descriptions_cover_all_flags() -> None:
    for flag in watch.FLAG_PATTERNS:
        assert flag in watch.FLAG_DESCRIPTIONS
