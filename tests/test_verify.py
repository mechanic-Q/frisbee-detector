"""Tests for dataset verification logic."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.verify_dataset import verify_yolo_dataset


def test_verify_clean_dataset():
    """Verify that a clean dataset passes verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "images" / "train").mkdir(parents=True)
        (tmp / "labels" / "train").mkdir(parents=True)

        # Create one valid image + label pair
        (tmp / "images" / "train" / "img001.jpg").write_text("")
        (tmp / "labels" / "train" / "img001.txt").write_text("0 0.5 0.5 0.1 0.1")

        # Create YAML config (val points to same dir to avoid dir-not-found issue)
        yaml_path = tmp / "data.yaml"
        yaml_path.write_text(
            f"path: {tmp}\n"
            "train: images/train\n"
            "val: images/train\n"
            "nc: 1\n"
            "names: ['frisbee']\n"
        )

        issues = verify_yolo_dataset(str(yaml_path))
        assert issues == 0


def test_verify_orphan_label():
    """Test that orphan labels are detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "images" / "train").mkdir(parents=True)
        (tmp / "labels" / "train").mkdir(parents=True)

        # Label without matching image
        (tmp / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1")

        yaml_path = tmp / "data.yaml"
        yaml_path.write_text(
            f"path: {tmp}\n"
            "train: images/train\n"
            "val: images/train\n"
            "nc: 1\n"
            "names: ['frisbee']\n"
        )

        issues = verify_yolo_dataset(str(yaml_path))
        assert issues > 0


def test_verify_invalid_bbox():
    """Test that out-of-range coordinates are detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "images" / "train").mkdir(parents=True)
        (tmp / "labels" / "train").mkdir(parents=True)

        (tmp / "images" / "train" / "img001.jpg").write_text("")
        # Invalid: width > 1.0
        (tmp / "labels" / "train" / "img001.txt").write_text("0 0.5 0.5 1.5 0.1")

        yaml_path = tmp / "data.yaml"
        yaml_path.write_text(
            f"path: {tmp}\n"
            "train: images/train\n"
            "val: images/train\n"
            "nc: 1\n"
            "names: ['frisbee']\n"
        )

        issues = verify_yolo_dataset(str(yaml_path))
        assert issues > 0


def test_verify_jpeg_orphan_detection():
    """Test that .jpeg files are matched to .txt labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "images" / "train").mkdir(parents=True)
        (tmp / "labels" / "train").mkdir(parents=True)

        # .jpeg image with matching label
        (tmp / "images" / "train" / "img001.jpeg").write_text("")
        (tmp / "labels" / "train" / "img001.txt").write_text("0 0.5 0.5 0.1 0.1")

        yaml_path = tmp / "data.yaml"
        yaml_path.write_text(
            f"path: {tmp}\n"
            "train: images/train\n"
            "val: images/train\n"
            "nc: 1\n"
            "names: ['frisbee']\n"
        )

        issues = verify_yolo_dataset(str(yaml_path))
        assert issues == 0  # .jpeg should be matched, no orphan detected


def test_verify_empty_label_is_valid_negative_sample():
    """Empty YOLO label files are valid negative/background samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "images" / "train").mkdir(parents=True)
        (tmp / "labels" / "train").mkdir(parents=True)

        (tmp / "images" / "train" / "background.jpg").write_text("")
        (tmp / "labels" / "train" / "background.txt").write_text("")

        yaml_path = tmp / "data.yaml"
        yaml_path.write_text(
            f"path: {tmp}\n"
            "train: images/train\n"
            "val: images/train\n"
            "nc: 1\n"
            "names: ['frisbee']\n"
        )

        issues = verify_yolo_dataset(str(yaml_path))
        assert issues == 0
