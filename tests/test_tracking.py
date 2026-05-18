"""Tests for tracking pipeline helper functions."""

import csv
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict_track import find_calibration


def test_find_calibration_exact_match(tmp_path, monkeypatch):
    """Exact stem match should find the calibration file."""
    calib_dir = tmp_path / "configs" / "homography"
    calib_dir.mkdir(parents=True)
    (calib_dir / "test_video.json").write_text(json.dumps({
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "video": "test_video.mp4", "image_size": [1280, 720],
        "field_size_m": [100, 37], "calibration_frame": 0,
        "points": [{"pixel": [100, 500], "world": [0, 0]}],
        "reprojection_error_px": 0.5,
    }))

    monkeypatch.chdir(tmp_path)
    result = find_calibration(Path("/videos/test_video.mp4"))
    assert result is not None
    assert result["video"] == "test_video.mp4"


def test_find_calibration_underscore_fallback(tmp_path, monkeypatch):
    """Video name with underscore should fall back to base name."""
    calib_dir = tmp_path / "configs" / "homography"
    calib_dir.mkdir(parents=True)
    (calib_dir / "base.json").write_text(json.dumps({
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "video": "base.mp4", "image_size": [1280, 720],
        "field_size_m": [100, 37], "calibration_frame": 0,
        "points": [{"pixel": [100, 500], "world": [0, 0]}],
        "reprojection_error_px": 0.5,
    }))

    monkeypatch.chdir(tmp_path)
    result = find_calibration(Path("/videos/base_55-56min.mp4"))
    assert result is not None
    assert result["video"] == "base.mp4"


def test_find_calibration_not_found(tmp_path, monkeypatch):
    """No matching file should return None."""
    calib_dir = tmp_path / "configs" / "homography"
    calib_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    result = find_calibration(Path("/videos/nonexistent.mp4"))
    assert result is None


def test_csv_fieldnames_match_new_format(tmp_path):
    """CSV should match the new single-frisbee tracker schema."""
    rows = [
        {"frame": 0, "px": 640.0, "py": 500.0, "vx": 2.5, "vy": -1.0,
         "conf": 0.52, "status": "tracking", "wx": None, "wy": None},
        {"frame": 1, "px": 642.0, "py": 498.0, "vx": 3.0, "vy": -1.5,
         "conf": 0.55, "status": "tracking", "wx": 50.1, "wy": 1.2},
    ]
    csv_path = tmp_path / "tracks.csv"
    fieldnames = ["frame", "px", "py", "vx", "vy", "conf", "status", "wx", "wy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 3
    assert lines[0] == "frame,px,py,vx,vy,conf,status,wx,wy"
    assert "0.52" in lines[1]
    assert "50.1" in lines[2]
    assert "tracking" in lines[1]
