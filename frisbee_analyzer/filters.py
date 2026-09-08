"""显示过滤：从检出中筛出场地球员（纯函数，可单测）。

两层规则（方案 §8.1 场地过滤的近似）：
1. 启发式：框高 ≥ min_height 且框底边 y2 ≥ H*min_y_frac（观众席在画面上部且框小）；
2. 场地多边形：有标定矩阵时，脚点反投影到世界坐标，落在场地矩形（含 margin）内才保留。
过滤只作用于显示视图，tracks.json 原始数据不动——统计侧可按自己的口径再过滤。
"""

from __future__ import annotations

FIELD_POLYGON_M = [(0.0, 0.0), (100.0, 0.0), (100.0, 37.0), (0.0, 37.0)]  # WFDF 场地矩形


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _polygon_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def is_field_det(bbox: list[float], frame_h: float, matrix=None,
                 polygon: list[tuple[float, float]] | None = None,
                 margin_m: float = 2.0, min_height: float = 90,
                 min_y_frac: float = 0.45) -> bool:
    x1, y1, x2, y2 = bbox
    if (y2 - y1) < min_height:
        return False
    if y2 < frame_h * min_y_frac:
        return False
    if matrix is not None and polygon:
        from utils.homography import pixel_to_world

        wx, wy = pixel_to_world(matrix, (x1 + x2) / 2, y2)
        minx, miny, maxx, maxy = _polygon_bbox(polygon)
        return (minx - margin_m) <= wx <= (maxx + margin_m) and \
               (miny - margin_m) <= wy <= (maxy + margin_m)
    return True


def filter_dets(dets: list[dict], frame_h: float, matrix=None,
                polygon: list[tuple[float, float]] | None = None,
                margin_m: float = 2.0, min_height: float = 90,
                min_y_frac: float = 0.45, require_team: bool = False) -> list[dict]:
    """按场内规则 + 可选"必须有队伍"过滤。require_team 用于隐藏观众/未分配轨迹。"""
    out = []
    for det in dets:
        if require_team and det.get("team_id") is None:
            continue
        if is_field_det(det["bbox"], frame_h, matrix, polygon, margin_m,
                        min_height, min_y_frac):
            out.append(det)
    return out
