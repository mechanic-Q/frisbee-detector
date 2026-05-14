"""Tests for frisbee binary classifier."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import pytest
import numpy as np
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_CSV = PROJECT_ROOT / "data" / "perbox_crops" / "review_results.csv"
CROP_DIR = PROJECT_ROOT / "data" / "perbox_crops"


@pytest.mark.integration
def test_load_labeled_dataset():
    """Load labeled dataset, return (path, label) list, 0=FP, 1=TP."""
    from tools.classifier_utils import load_labeled_dataset

    samples = load_labeled_dataset(REVIEW_CSV, CROP_DIR)

    assert len(samples) == 200
    paths, labels = zip(*samples)
    assert all(p.is_file() for p in paths)
    assert set(labels) == {0, 1}

    tp_count = sum(labels)
    fp_count = len(labels) - tp_count
    assert tp_count == 22
    assert fp_count == 178


def test_train_val_split():
    """Stratified train/val split preserves class ratio."""
    from tools.classifier_utils import split_train_val

    import tempfile, csv

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i in range(5):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp / f"crop_{i}.jpg"), img)

        csv_path = tmp / "review.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "result", "frame", "conf"])
            writer.writerow(["crop_0.jpg", "TP", "1", "0.5"])
            writer.writerow(["crop_1.jpg", "FP", "2", "0.4"])
            writer.writerow(["crop_2.jpg", "TP", "3", "0.6"])
            writer.writerow(["crop_3.jpg", "FP", "4", "0.3"])
            writer.writerow(["crop_4.jpg", "TP", "5", "0.7"])

        samples = [(tmp / row[0], 1 if row[1] == "TP" else 0) for row in [
            ("crop_0.jpg", "TP"), ("crop_1.jpg", "FP"), ("crop_2.jpg", "TP"),
            ("crop_3.jpg", "FP"), ("crop_4.jpg", "TP")
        ]]

        train, val = split_train_val(samples, val_ratio=0.4, seed=42)

        assert len(train) + len(val) == 5
        train_tp = sum(1 for _, l in train if l == 1)
        train_fp = sum(1 for _, l in train if l == 0)
        val_tp = sum(1 for _, l in val if l == 1)
        val_fp = sum(1 for _, l in val if l == 0)

        assert train_tp + val_tp == 3
        assert train_fp + val_fp == 2
        assert train_tp >= 1 and val_tp >= 1
        assert train_fp >= 1 and val_fp >= 1
