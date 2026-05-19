"""Tests for _label_utils: label IO, IoU, P2 cache."""

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from _label_utils import (
    read_label,
    write_label,
    compute_iou,
    load_p2_cache,
    save_p2_cache,
    run_p2_inference,
    load_review_state,
    save_review_state,
    sort_frames,
)


class TestReadLabel:
    def test_valid_label(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("0 0.500000 0.300000 0.040000 0.060000\n")
        result = read_label(txt)
        assert result is not None
        assert result["cx"] == pytest.approx(0.5)
        assert result["cy"] == pytest.approx(0.3)
        assert result["w"] == pytest.approx(0.04)
        assert result["h"] == pytest.approx(0.06)

    def test_empty_label(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("")
        assert read_label(txt) is None

    def test_missing_file(self, tmp_path):
        txt = tmp_path / "missing.txt"
        assert read_label(txt) is None

    def test_multiline_returns_first(self, tmp_path):
        txt = tmp_path / "multi.txt"
        txt.write_text("0 0.1 0.2 0.3 0.4\n0 0.5 0.6 0.1 0.1\n")
        result = read_label(txt)
        assert result["cx"] == pytest.approx(0.1)


class TestWriteLabel:
    def test_write_positive(self, tmp_path):
        txt = tmp_path / "out.txt"
        write_label(txt, {"cx": 0.5, "cy": 0.3, "w": 0.04, "h": 0.06})
        assert txt.read_text() == "0 0.500000 0.300000 0.040000 0.060000\n"

    def test_write_negative(self, tmp_path):
        txt = tmp_path / "out.txt"
        write_label(txt, None)
        assert txt.read_text() == ""


class TestComputeIoU:
    def test_same_box(self):
        assert compute_iou(0.5, 0.5, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_iou(0.2, 0.5, 0.1, 0.1, 0.8, 0.5, 0.1, 0.1) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = compute_iou(0.5, 0.5, 0.2, 0.2, 0.55, 0.55, 0.2, 0.2)
        assert 0.0 < iou < 1.0

    def test_zero_size_box(self):
        assert compute_iou(0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.1, 0.1) == pytest.approx(0.0)


class TestP2Cache:
    def test_save_and_load(self, tmp_path):
        cache = {"frame_0001": {"cx": 0.5, "cy": 0.3, "w": 0.04, "h": 0.06, "conf": 0.8}}
        path = tmp_path / "p2_cache.json"
        save_p2_cache(path, cache)
        loaded = load_p2_cache(path)
        assert loaded["frame_0001"]["conf"] == pytest.approx(0.8)

    def test_load_missing_returns_empty(self, tmp_path):
        path = tmp_path / "missing.json"
        assert load_p2_cache(path) == {}


class TestReviewState:
    def test_save_and_load(self, tmp_path):
        state = {
            "accept": ["frame_0001", "frame_0002"],
            "reject": ["frame_0003"],
            "skip": [],
        }
        path = tmp_path / "review_result.json"
        save_review_state(path, state)
        loaded = load_review_state(path)
        assert "frame_0001" in loaded["accept"]
        assert "frame_0003" in loaded["reject"]

    def test_load_missing_returns_default(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        loaded = load_review_state(path)
        assert loaded == {"accept": [], "reject": [], "skip": []}


class TestSortFrames:
    def test_sort_by_iou_desc(self):
        p2 = {
            "f1": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "conf": 0.8},
            "f2": {"cx": 0.6, "cy": 0.6, "w": 0.1, "h": 0.1, "conf": 0.9},
        }
        src = {
            "f1": {"cx": 0.51, "cy": 0.51, "w": 0.1, "h": 0.1},
        }
        from pathlib import Path
        f1 = Path("/frames/f1.jpg")
        f2 = Path("/frames/f2.jpg")
        f3 = Path("/frames/f3.jpg")
        result = sort_frames([f3, f2, f1], p2, src, "iou_desc")
        assert result[0] == f1
        assert result[-1] in (f2, f3)

    def test_sort_by_bbox_size(self):
        src = {
            "large": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.3},
            "small": {"cx": 0.5, "cy": 0.5, "w": 0.01, "h": 0.01},
        }
        from pathlib import Path
        large = Path("/frames/large.jpg")
        small = Path("/frames/small.jpg")
        result = sort_frames([small, large], {}, src, "bbox_size")
        assert result[0] == large
        assert result[1] == small

    def test_sort_unlabeled_last(self):
        src = {
            "labeled": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
        }
        from pathlib import Path
        lab = Path("/frames/labeled.jpg")
        unlab = Path("/frames/unlabeled.jpg")
        result = sort_frames([unlab, lab], {}, src, "iou_desc")
        assert result[0] == lab
        assert result[1] == unlab
