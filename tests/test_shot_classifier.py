"""shot_classifier 单元测试（§13.19 镜头分类器）。

运行: python -m pytest tests/test_shot_classifier.py -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.shot_classifier import classify_sequence, classify_shot  # noqa: E402

FH = 480.0


def _player(y1=300, y2=340):
    return {"bbox": [100.0, y1, 130.0, y2]}


def test_wide_shot_many_small_players():
    dets = [_player(300 + i, 330 + i) for i in range(10)]  # 10 人小框
    assert classify_shot(dets, FH) == "wide"


def test_close_shot_few_large_players():
    dets = [_player(100, 400)]  # 1 人半身大框（框高 62%）
    assert classify_shot(dets, FH) == "close"


def test_empty_shot_no_players():
    assert classify_shot([], FH) == "empty"


def test_mid_counts_fall_to_wide():
    dets = [_player() for _ in range(5)]  # 5 人小框 → 宁进 wide 分母
    assert classify_shot(dets, FH) == "wide"


def test_sequence_stats():
    per_frame = {
        "0": [_player() for _ in range(10)],
        "1": [_player(100, 400)],
        "2": [],
        "3": [_player() for _ in range(12)],
    }
    out = classify_sequence(per_frame, FH)
    assert out["counts"] == {"wide": 2, "close": 1, "empty": 1}
    assert out["wide_frac"] == 0.5
    assert out["wide_frames"] == [0, 3]
