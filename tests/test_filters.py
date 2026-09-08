"""filters.py 单测：场内过滤（启发式 + 标定多边形）与队伍过滤。"""

import numpy as np

from frisbee_analyzer.filters import filter_dets, is_field_det
from utils.homography import compute_homography


def _det(bbox, team=0):
    return {"track_id": 1, "bbox": bbox, "team_id": team, "conf": 0.9}


# ── 启发式 ───────────────────────────────────────────────

def test_field_player_kept_crowd_dropped():
    frame_h = 1080
    player = [800, 600, 900, 950]      # 框高 350，脚点在下部 → 场内
    crowd_small = [300, 250, 340, 300]   # 框高 50 → 太小
    crowd_high = [1200, 100, 1280, 320]  # 脚点 y2=320 < 0.45*1080 → 画面上部
    assert is_field_det(player, frame_h)
    assert not is_field_det(crowd_small, frame_h)
    assert not is_field_det(crowd_high, frame_h)
    out = filter_dets([_det(player), _det(crowd_small), _det(crowd_high)], frame_h)
    assert [d["bbox"] for d in out] == [player]


# ── 队伍过滤 ─────────────────────────────────────────────

def test_require_team_drops_unassigned():
    frame_h = 1080
    dets = [_det([800, 600, 900, 950], team=None), _det([100, 600, 200, 950], team=1)]
    out = filter_dets(dets, frame_h, require_team=True)
    assert len(out) == 1 and out[0]["team_id"] == 1
    assert len(filter_dets(dets, frame_h, require_team=False)) == 2


# ── 标定多边形过滤 ────────────────────────────────────────

def _matrix_shift_left_30m():
    """像素→世界：画面横向 1000px 映射到场外 -30m…70m，脚点 px<300 世界 x<0（场外）。"""
    pts = [(0, 0, -30, 0), (1000, 0, 70, 0), (1000, 370, 70, 37), (0, 370, -30, 37)]
    matrix, rmse = compute_homography(pts)
    assert matrix is not None
    return matrix


def test_polygon_filter_drops_off_field():
    from frisbee_analyzer.filters import FIELD_POLYGON_M

    matrix = _matrix_shift_left_30m()
    frame_h = 370
    inside = [700, 200, 760, 360]   # 脚点 (730,360) → 世界 (43,36) 场内
    outside = [100, 200, 160, 360]  # 脚点 (130,360) → 世界 (-17,36) 场外（左边的车队区）
    assert is_field_det(inside, frame_h, matrix=matrix, polygon=FIELD_POLYGON_M)
    assert not is_field_det(outside, frame_h, matrix=matrix, polygon=FIELD_POLYGON_M)
    # margin=2m 仍救不了 -17m 的场外点，但能容纳压线球员（margin 边界验证）
    edge = [298, 200, 302, 360]     # 脚点 (300,360) → 世界 (0,36) 正好在场线上
    assert is_field_det(edge, frame_h, matrix=matrix, polygon=FIELD_POLYGON_M)


def test_np_variance_guard():
    np.zeros(3)  # 确认测试环境依赖可用（filters 不直接用 numpy，防止误删依赖）
