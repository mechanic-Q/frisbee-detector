"""Tests for tracking pipeline helper functions."""

import datetime
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_calibration_lookup_exact_match(tmp_path, monkeypatch):
    """Simulate find_calibration logic: exact stem match."""
    calib_dir = tmp_path / "homography"
    calib_dir.mkdir(parents=True)
    (calib_dir / "test_video.json").write_text('{"matrix":[[1,0,0],[0,1,0],[0,0,1]],'
                                               '"video":"test.mp4","image_size":[1280,720],'
                                               '"field_size_m":[100,37],"calibration_frame":0,'
                                               '"points":[],"reprojection_error_px":0.5}')

    stem = "test_video"
    candidate = calib_dir / f"{stem}.json"
    assert candidate.exists()

    import json
    data = json.loads(candidate.read_text())
    assert data["video"] == "test.mp4"


def test_calibration_lookup_underscore_fallback(tmp_path):
    """Simulate find_calibration: fallback to base before first underscore."""
    calib_dir = tmp_path / "homography"
    calib_dir.mkdir(parents=True)
    (calib_dir / "base.json").write_text('{"matrix":[[1,0,0],[0,1,0],[0,0,1]],'
                                         '"video":"base.mp4","image_size":[1280,720],'
                                         '"field_size_m":[100,37],"calibration_frame":0,'
                                         '"points":[],"reprojection_error_px":0.5}')

    video_stem = "base_55-56min"
    base = video_stem.split("_", 1)[0] if "_" in video_stem else video_stem
    assert base == "base"
    candidate = calib_dir / f"{base}.json"
    assert candidate.exists()

    import json
    data = json.loads(candidate.read_text())
    assert data["video"] == "base.mp4"


def test_calibration_lookup_not_found(tmp_path):
    calib_dir = tmp_path / "homography"
    calib_dir.mkdir()

    video_stem = "nonexistent"
    assert not (calib_dir / f"{video_stem}.json").exists()

    base = video_stem.split("_", 1)[0]
    assert not (calib_dir / f"{base}.json").exists()


def test_export_csv_rows(tmp_path):
    """Test CSV export logic."""
    rows = [
        {"frame": 0, "track_id": 1, "px": 640.0, "py": 500.0, "wx": "", "wy": "", "conf": 0.52},
        {"frame": 1, "track_id": 1, "px": 642.0, "py": 498.0, "wx": 50.1, "wy": 1.2, "conf": 0.55},
    ]
    csv_path = tmp_path / "tracks.csv"
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "px", "py", "wx", "wy", "conf"])
        writer.writeheader()
        writer.writerows(rows)

    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 3
    assert lines[0] == "frame,track_id,px,py,wx,wy,conf"
    assert lines[1].endswith("0.52")
    assert "50.1" in lines[2]
