"""Tests for merging annotation export samples into training split."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.merge_annotation_exports import merge_export


def test_merge_export_copies_positive_and_hard_negative_pairs(tmp_path):
    export_dir = tmp_path / "export"
    dataset_dir = tmp_path / "dataset"

    (export_dir / "images" / "positive").mkdir(parents=True)
    (export_dir / "labels" / "positive").mkdir(parents=True)
    (export_dir / "images" / "hard_negative").mkdir(parents=True)
    (export_dir / "labels" / "hard_negative").mkdir(parents=True)

    (export_dir / "images" / "positive" / "pos.jpg").write_bytes(b"pos-image")
    (export_dir / "labels" / "positive" / "pos.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (export_dir / "images" / "hard_negative" / "neg.jpg").write_bytes(b"neg-image")
    (export_dir / "labels" / "hard_negative" / "neg.txt").write_text("", encoding="utf-8")

    report = merge_export(export_dir, dataset_dir)

    assert report == {"positive_copied": 1, "hard_negative_copied": 1, "total_copied": 2}
    assert (dataset_dir / "images" / "train" / "pos.jpg").read_bytes() == b"pos-image"
    assert (dataset_dir / "labels" / "train" / "pos.txt").read_text(encoding="utf-8").startswith("0 ")
    assert (dataset_dir / "images" / "train" / "neg.jpg").read_bytes() == b"neg-image"
    assert (dataset_dir / "labels" / "train" / "neg.txt").read_text(encoding="utf-8") == ""
