"""frisbee_analyzer 持盘补全单元测试（纯几何，无 CV 依赖）。

运行: python -m pytest tests/test_possession_impute.py -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.disc_fusion import DiscFrame  # noqa: E402
from frisbee_analyzer.possession_impute import (  # noqa: E402
    IMPUTE_BBOX_SIZE,
    IMPUTE_CONF,
    impute_possession,
)


def _df(status, cx=100.0, cy=200.0):
    return DiscFrame([cx - 5, cy - 5, cx + 5, cy + 5], 0.9, cx, cy, status)


def _player(bbox, tid=7):
    return {"track_id": tid, "bbox": list(bbox), "conf": 0.9, "cls": 0}


def test_basic_imputation_on_static_holder():
    # 帧0 tracking 于 (100,200)，帧1-5 searching，锚点静止；球员 #7 框含锚点且不动
    fused = [_df("tracking")] + [_df("searching")] * 5
    players = {i: [_player([80, 150, 130, 260])] for i in range(6)}
    extra, stats = impute_possession(fused, players, min_streak=3)
    assert stats.streaks == 1
    assert stats.frames_imputed == 5  # 帧1-5 全补
    assert set(extra) == {1, 2, 3, 4, 5}
    for fi, dets in extra.items():
        assert dets[0]["source"] == "possession_imputed"
        assert dets[0]["conf"] == IMPUTE_CONF
        x1, y1, x2, y2 = dets[0]["bbox"]
        assert x2 - x1 == IMPUTE_BBOX_SIZE


def test_two_players_containing_anchor_is_ambiguous():
    fused = [_df("tracking")] + [_df("searching")] * 4
    players = {i: [_player([80, 150, 130, 260], tid=7),
                   _player([90, 160, 140, 270], tid=8)] for i in range(5)}
    extra, stats = impute_possession(fused, players)
    assert extra == {} and stats.streaks == 0


def test_track_id_change_terminates_streak():
    # 帧3 起换成另一名球员含锚点 → 段在帧2 截止（不足 min_streak=3 则无输出）
    fused = [_df("tracking")] + [_df("searching")] * 6
    players = {i: [_player([80, 150, 130, 260], tid=7)] for i in range(3)}
    players.update({i: [_player([80, 150, 130, 260], tid=9)] for i in range(3, 6)})
    extra, stats = impute_possession(fused, players, min_streak=3)
    assert stats.streaks == 0 and extra == {}


def test_speed_gate_stops_on_runner():
    # 球员框每帧右移 40px（> 6px/帧）→ 速度门在第二帧终止 → 段不足 3 帧
    fused = [_df("tracking")] + [_df("searching")] * 6
    players = {i: [_player([80 + 40 * i, 150, 130 + 40 * i, 260], tid=7)] for i in range(6)}
    extra, stats = impute_possession(fused, players, min_streak=3)
    assert stats.streaks == 0 and extra == {}


def test_anchor_outside_any_box_no_imputation():
    fused = [_df("tracking", cx=500, cy=400)] + [_df("searching", cx=500, cy=400)] * 5
    players = {i: [_player([80, 150, 130, 260])] for i in range(6)}
    extra, stats = impute_possession(fused, players)
    assert extra == {}


def test_world_roundtrip_hand_pixel():
    # 提供恒等投影时，手部像素 = 脚点世界 + 1.3m 的反投影（此处 identity: px==wx）
    fused = [_df("tracking")] + [_df("searching")] * 4
    players = {i: [_player([80, 150, 130, 260])] for i in range(5)}
    extra, _ = impute_possession(
        fused, players, world_to_pixel=lambda wx, wy: (wx, wy),
        pixel_to_world=lambda px, py: (px, py), min_streak=3)
    x1, y1, x2, y2 = extra[1][0]["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    assert (cx, cy) == (105.0, 260.0 + 1.3)  # 脚点(105,260) + 手高


def test_max_streak_cap():
    fused = [_df("tracking")] + [_df("searching")] * 200
    players = {i: [_player([80, 150, 130, 260])] for i in range(201)}
    extra, stats = impute_possession(fused, players, min_streak=3, max_streak=10)
    assert stats.frames_imputed == 10
    assert set(extra) == set(range(1, 11))
