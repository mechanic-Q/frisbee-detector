"""段内自动标定（pipeline 内联版，F1 后续）。

复用 auto_calibrate_v2 的 players-extent 算法（脚点云 → 4 角点内点率优化），
但以可导入函数形态供 pipeline 在检测完成后、过滤前调用——消除"标定文件与
素材时间段错位"这类系统性坑（§13.2/§13.9 三次实证）。

纯 numpy 实现（T5 同款纯度）：只依赖 numpy + utils.homography 的纯函数。
"""

from __future__ import annotations

import numpy as np

FIELD = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 37.0], [0.0, 37.0]])


def _compute_homography(points):
    """与 utils.homography.compute_homography 同签名的轻量 DLT（避免循环依赖）。"""
    A = []
    for px, py, wx, wy in points:
        A.append([px, py, 1, 0, 0, 0, -wx * px, -wx * py, -wx])
        A.append([0, 0, 0, px, py, 1, -wy * px, -wy * py, -wy])
    A = np.asarray(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    h = vt[-1]
    if abs(h[8]) < 1e-12:
        return None
    return (h / h[8]).reshape(3, 3)


def _pixel_to_world(m, px, py):
    p = m @ np.array([px, py, 1.0])
    if abs(p[2]) < 1e-12:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def collect_foot_points(frames: dict, frame_h: float, min_height_frac: float = 0.05) -> np.ndarray:
    """从 frames（未过滤检测）收集脚点 (cx, y2)，过滤小目标（观众/远处人头）。"""
    pts = []
    for dets in frames.values():
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            if (y2 - y1) < frame_h * min_height_frac:
                continue
            pts.append(((x1 + x2) / 2, y2))
    return np.asarray(pts, dtype=np.float64)


def _corners_to_homo(corners: np.ndarray):
    pts = [(float(c[0]), float(c[1]), float(w[0]), float(w[1]))
           for c, w in zip(corners, FIELD)]
    return _compute_homography(pts)


def _inlier_ratio(corners: np.ndarray, pts: np.ndarray, margin: float = 3.0) -> float:
    m = _corners_to_homo(corners)
    if m is None:
        return 0.0
    wx = np.empty(len(pts))
    wy = np.empty(len(pts))
    for i, (px, py) in enumerate(pts):
        r = _pixel_to_world(m, px, py)
        if r is None:
            wx[i], wy[i] = -999.0, -999.0
        else:
            wx[i], wy[i] = r
    lo = FIELD.min(axis=0) - margin
    hi = FIELD.max(axis=0) + margin
    inside = (wx >= lo[0]) & (wx <= hi[0]) & (wy >= lo[1]) & (wy <= hi[1])
    return float(inside.mean())


def _init_corners(pts: np.ndarray) -> np.ndarray:
    ys = pts[:, 1]
    near = pts[ys >= np.quantile(ys, 0.75)]
    far = pts[ys <= np.quantile(ys, 0.45)]
    if len(near) < 2 or len(far) < 2:
        return np.empty((0, 2))
    return np.array([near[near[:, 0].argmin()], near[near[:, 0].argmax()],
                     far[far[:, 0].argmax()], far[far[:, 0].argmin()]], dtype=np.float64)


def _corners_sane(corners: np.ndarray, min_sep: float = 5.0) -> bool:
    """几何合法性：near 带 y > far 带 y（梯形不倒置），左角 x < 右角 x。

    防两类退化骗局：压扁（共线点上矩形塌成线骗满内点）与左右翻转。
    """
    near_l, near_r, far_r, far_l = corners
    return (near_l[1] > far_l[1] + min_sep and near_r[1] > far_r[1] + min_sep
            and near_l[0] < near_r[0] - min_sep and far_l[0] < far_r[0] - min_sep)


def _optimize(corners: np.ndarray, pts: np.ndarray, w: float, h: float,
              steps=(40.0, 20.0, 10.0), clamp: float = 60.0, margin: float = 1.5):
    best = corners.copy()
    best_score = _inlier_ratio(best, pts, margin) if _corners_sane(best) else -1.0
    lo = np.maximum(best - clamp, [0.0, 0.0])
    hi = np.minimum(best + clamp, [w, h])
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
                        if not _corners_sane(cand):
                            continue
                        s = _inlier_ratio(cand, pts, margin=margin)
                        if s > best_score + 1e-6:
                            best, best_score = cand, s
                            improved = True
    return best, best_score


def auto_calibrate_segment(frames: dict, width: float, height: float,
                           min_points: int = 500,
                           gate: float = 0.85) -> tuple[dict | None, dict]:
    """对一次 run 的未过滤 frames 做段内标定。

    返回 (calib_dict | None, report)。calib_dict 结构与 utils.homography.load_calibration
    兼容（含 matrix/image_size/inlier_ratio），None = 未过 0.85 门（报告带原因）。
    """
    pts = collect_foot_points(frames, height)
    report = {"points": len(pts), "gate": gate}
    if len(pts) < min_points:
        report["reason"] = f"too few foot points: {len(pts)} < {min_points}"
        return None, report
    corners0 = _init_corners(pts)
    if corners0.size == 0 or not _corners_sane(corners0):
        report["reason"] = "degenerate corner init (collinear or inverted)"
        return None, report
    s0 = _inlier_ratio(corners0, pts)
    corners, score = _optimize(corners0, pts, width, height)
    if not _corners_sane(corners) or score < 0:
        report["reason"] = "no geometrically sane solution"
        return None, report
    m = _corners_to_homo(corners)
    if m is None:
        report["reason"] = "homography degenerate"
        return None, report
    report.update(init_inlier_ratio=round(s0, 4), optimized_inlier_ratio=round(score, 4),
                  passed=bool(score >= gate))
    if score < gate:
        report["reason"] = f"inlier ratio {score:.3f} < gate {gate}"
        return None, report
    calib = {
        "image_size": [width, height],
        "field_size_m": [100.0, 37.0],
        "calibration_frame": "auto(segment players-extent)",
        "points": [{"pixel": [float(c[0]), float(c[1])],
                    "world": [float(FIELD[i][0]), float(FIELD[i][1])]}
                   for i, c in enumerate(corners)],
        "matrix": m.tolist(),
        "reprojection_error_px": 0.0,
        "method": "players-extent segment-inline",
        "inlier_ratio": round(score, 4),
    }
    return calib, report


# ── 恢复扫窗（§13.15）：整段 FAIL 时自动找最佳过门子窗 ──
RECOVER_WINDOW = 1200   # 扫窗窗长（帧）；实际取 min(此值, 总帧数//2)
RECOVER_STEP_DIV = 2    # 步长 = 窗长/2


def auto_calibrate_segment_recover(frames: dict, width: float, height: float,
                                   min_points: int = 500,
                                   gate: float = 0.85) -> tuple[dict | None, dict]:
    """auto_calibrate_segment 的带恢复版：整段 FAIL 时滑窗找最佳过门子窗。

    实证（§13.13/13.15 chunk0）：跨摇镜头长段整段内点率 0.826，但 [1800,3000)
    子窗 0.881 过门——摇镜头/开段 lineup 只污染局部。返回结构与
    auto_calibrate_segment 一致；恢复成功时 creport 带 recovery_window=[a,b)。
    """
    calib, report = auto_calibrate_segment(frames, width, height, min_points, gate)
    if calib is not None:
        return calib, report

    n = max((int(k) for k in frames), default=-1) + 1
    if n <= 0:
        return None, report
    win = min(RECOVER_WINDOW, max(n // 2, 1))
    step = max(win // RECOVER_STEP_DIV, 1)
    best = None
    for a in range(0, n - win // 2 + 1, step):
        b = min(a + win, n)
        if b - a < win // 2:
            continue
        sub = {k: v for k, v in frames.items() if a <= int(k) < b}
        c, r = auto_calibrate_segment(sub, width, height,
                                      min_points=max(min_points // 2, 200), gate=gate)
        inl = r.get("optimized_inlier_ratio", 0.0)
        if c is not None and (best is None or inl > best[2]):
            best = (a, b, inl, c)
    if best is None:
        report["reason"] = "whole-segment and all recovery windows failed"
        return None, report
    a, b, inl, calib = best
    report = dict(report)
    report.update(passed=True, recovery_window=[a, b],
                  optimized_inlier_ratio=round(inl, 4))
    report.pop("reason", None)
    return calib, report
