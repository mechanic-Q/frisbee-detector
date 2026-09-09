"""自动标定 v2：球员脚点分布反推场地矩形（players-extent 路线，零训练零人工）。

与 E5-v0（Hough 线检测，负结果）的区别：不检测白线，而是利用"球员只能在场地内跑动"
这一强约束——把脚点云的内接矩形拟合问题转化为 4 角点参数优化，目标函数 = 内点率。

方法：
  1. 从未过滤 tracks.json 取脚点 (cx, y2)，按框高过滤掉小目标（观众/远处人头）；
  2. 分位数初始化 4 角点（near/far × left/right）；
  3. 坐标下降：每个角点在邻域网格内移动，最大化内点率（点落在场地多边形 ±margin 内）；
  4. 产物 = compute_homography(4 角点↔场地角) 的矩阵 + 内点率自检。

用法：
    python3 tools/auto_calibrate_v2.py --tracks <raw tracks.json> [--out <json>]
自检门槛：内点率 ≥0.85 视为可用；否则报告负结果（等待主会话 pitch_kp 路线）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer import protocol  # noqa: E402
from frisbee_analyzer.filters import point_in_polygon  # noqa: E402
from utils.homography import FIELD_LINES_WORLD, compute_homography, pixel_to_world  # noqa: E402

FIELD = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 37.0], [0.0, 37.0]])
# 角点角色：0=左下(near-left) 1=右下(near-right) 2=右上(far-right) 3=左上(far-left)
CORNERS_WORLD = FIELD


def load_foot_points(path: str, min_height_frac: float = 0.05) -> np.ndarray:
    doc = json.loads(Path(protocol.win_to_wsl(path)).read_text(encoding="utf-8"))
    h = doc.get("height") or 1080
    pts = []
    for dets in doc["frames"].values():
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            if (y2 - y1) < h * min_height_frac:
                continue
            pts.append(((x1 + x2) / 2, y2))
    return np.asarray(pts, dtype=np.float64)


def corners_to_homo(corners: np.ndarray):
    """4 个图像角点 → 场地 4 角的单应性。corners 顺序 = CORNERS_WORLD。"""
    pts = [(float(c[0]), float(c[1]), float(w[0]), float(w[1]))
           for c, w in zip(corners, CORNERS_WORLD)]
    return compute_homography(pts)


def inlier_ratio(corners: np.ndarray, pts: np.ndarray, margin: float = 3.0) -> float:
    m, _ = corners_to_homo(corners)
    if m is None:
        return 0.0
    wx = np.empty(len(pts))
    wy = np.empty(len(pts))
    for i, (px, py) in enumerate(pts):
        wx[i], wy[i] = pixel_to_world(m, px, py)
    minx, miny = FIELD.min(axis=0) - margin
    maxx, maxy = FIELD.max(axis=0) + margin
    inside = (wx >= minx) & (wx <= maxx) & (wy >= miny) & (wy <= maxy)
    return float(inside.mean())


def init_corners(pts: np.ndarray, h: float, w: float) -> np.ndarray:
    """分位数初始化：near 带取 y2 大的 25% 分位附近，far 带取 y2 小的 25% 分位附近。"""
    ys = pts[:, 1]
    near_band = pts[ys >= np.quantile(ys, 0.75)]
    far_band = pts[ys <= np.quantile(ys, 0.45)]
    near_left = near_band[near_band[:, 0].argmin()]
    near_right = near_band[near_band[:, 0].argmax()]
    far_left = far_band[far_band[:, 0].argmin()]
    far_right = far_band[far_band[:, 0].argmax()]
    # 顺序：左下、右下、右上、左上
    return np.array([near_left, near_right, far_right, far_left], dtype=np.float64)


def optimize(corners: np.ndarray, pts: np.ndarray, steps=(40.0, 20.0, 10.0),
             clamp: float = 60.0, margin: float = 1.5) -> tuple[np.ndarray, float]:
    """坐标下降：逐角点在邻域网格内移动最大化内点率。

    关键约束：角点被 clamp 在初始化分位数位置 ±clamp px 内——无约束优化会把矩形
    膨胀到覆盖全画面骗取内点率（实测 0.893→1.0 但多边形盖住观众区，保留 95% 框）。
    margin 1.5m：容忍压线球员，但不容纳球场外区域。
    """
    best = corners.copy()
    best_score = inlier_ratio(best, pts, margin=margin)
    lo = best - clamp
    hi = best + clamp
    hi[:, 1] = np.minimum(hi[:, 1], 1080.0)
    hi[:, 0] = np.minimum(hi[:, 0], 1920.0)
    lo[:, 0] = np.maximum(lo[:, 0], 0.0)
    lo[:, 1] = np.maximum(lo[:, 1], 0.0)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for ci in range(4):
                for axis in (0, 1):
                    for sign in (-1, 1):
                        cand = best.copy()
                        cand[ci, axis] += sign * step
                        if not (lo[ci, 0] <= cand[ci, 0] <= hi[ci, 0]
                                and lo[ci, 1] <= cand[ci, 1] <= hi[ci, 1]):
                            continue
                        s = inlier_ratio(cand, pts, margin=margin)
                        if s > best_score + 1e-6:
                            best, best_score = cand, s
                            improved = True
    return best, best_score


def main() -> int:
    parser = argparse.ArgumentParser(description="自动标定 v2（球员脚点分布反推场地矩形）")
    parser.add_argument("--tracks", required=True, help="未过滤 tracks.json")
    parser.add_argument("--out", default=None, help="输出 json（默认与 tracks 同目录 auto_calib.json）")
    parser.add_argument("--video", default=None, help="视频名（写入标定元数据）")
    args = parser.parse_args()

    doc = json.loads(Path(protocol.win_to_wsl(args.tracks)).read_text(encoding="utf-8"))
    h = doc.get("height") or 1080
    w = doc.get("width") or 1920
    pts = load_foot_points(args.tracks)
    if len(pts) < 500:
        print(json.dumps({"error": f"too few foot points: {len(pts)}"}))
        return 1

    corners0 = init_corners(pts, h, w)
    s0 = inlier_ratio(corners0, pts)
    corners, score = optimize(corners0, pts)
    m, _rmse = corners_to_homo(corners)

    report = {
        "points": len(pts),
        "init_inlier_ratio": round(s0, 4),
        "optimized_inlier_ratio": round(score, 4),
        "gate": ">=0.85",
        "pass": bool(score >= 0.85),
        "corners_px": {name: corners[i].tolist() for i, name in
                       enumerate(["near-left", "near-right", "far-right", "far-left"])},
    }
    out = args.out or str(Path(protocol.win_to_wsl(args.tracks)).parent / "auto_calib.json")
    calib = {
        "video": args.video or doc.get("video", ""),
        "image_size": [w, h],
        "field_size_m": [100.0, 37.0],
        "calibration_frame": "auto(players-extent)",
        "points": [{"pixel": [float(c[0]), float(c[1])],
                    "world": [float(CORNERS_WORLD[i][0]), float(CORNERS_WORLD[i][1])]}
                   for i, c in enumerate(corners)],
        "matrix": m.tolist(),
        "reprojection_error_px": 0.0,
        "method": "players-extent v2",
        "inlier_ratio": report["optimized_inlier_ratio"],
    }
    Path(protocol.win_to_wsl(out)).write_text(json.dumps(calib, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
