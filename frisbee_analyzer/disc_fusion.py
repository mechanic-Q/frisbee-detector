"""盘轨迹多维融合（纯函数模块，F1 Phase A）。

多维互证：检测置信度 × 帧间运动关联（Kalman 预测 + 候选评分）× 马氏门控
（野值拒绝）× 世界坐标速度物理上限 × 断轨外推续接。

设计约束（T5）：只依赖 numpy + 标准库 + 可选注入的世界坐标投影函数——
不 import cv2/torch/ultralytics，保证可单测可移植。

输入：逐帧盘检测（list[Detections]），每帧候选 {bbox:[x1,y1,x2,y2], conf:float}
输出：逐帧 DiscFrame（status: tracking|predicting|gated|rejected|lost）+ 汇总统计。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ── 可调常量（与 predict_track.py / tracker_utils.py 主线口径一致）──
MAX_EXPECTED_DISPLACEMENT = 50.0   # px，帧间运动搜索半径（25fps）
MIN_DISPLACEMENT = 5.0             # px/帧，"在动"阈值
MIN_SCORE = 0.3                    # 候选最低分（低于直接丢弃）
GATE_THRESHOLD = 13.8155           # chi²₀.₉₉₉(df=2)——事后马氏门控（v3 设计）
MAX_SPEED_MS = 25.0                # 世界坐标物理上限（飞盘不可能超此速度）
LOST_TRACK_THRESHOLD = 15          # 连续丢失/降级超过此帧数 → 轨迹结束重找
MIN_TRACK_QUALITY = 3              # 开轨迹所需最少连续 tracking 帧


# ── 数据结构 ──

@dataclass
class DiscObservation:
    bbox: list[float]           # [x1,y1,x2,y2] 像素
    conf: float


@dataclass
class DiscFrame:
    bbox: list[float] | None
    conf: float
    cx: float
    cy: float
    status: str                 # tracking|predicting|gated|rejected|lost|searching
    d2: float | None = None     # 马氏距离²（tracking 时对所选候选）
    speed_ms: float | None = None
    world_xy: tuple[float, float] | None = None


@dataclass
class FusionStats:
    n_frames: int = 0
    n_tracking: int = 0
    n_predicting: int = 0
    n_gated: int = 0            # 马氏门控拒绝的候选帧
    n_rejected_speed: int = 0   # 速度物理拒绝
    n_lost: int = 0
    n_searching: int = 0
    n_tracks: int = 0           # 开过的轨迹数
    longest_track_frames: int = 0
    field_rejected: int = 0     # 场外投影剔除（有投影函数时）
    details: dict = field(default_factory=dict)


# ── Kalman（4 态 px,py,vx,vy；轻量实现替代 cv2.KalmanFilter，保证 T5）──

class Kalman2D:
    """常数速度模型 + 每帧正确；processNoise/measurementNoise 与主线同量级。"""

    def __init__(self):
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.eye(4) * 0.003
        self.R = np.eye(2) * 0.1
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 100.0
        self.initialized = False

    def predict(self) -> tuple[float, float]:
        if not self.initialized:
            return 0.0, 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def correct(self, cx: float, cy: float) -> None:
        if not self.initialized:
            self.x[:] = 0.0
            self.x[0, 0], self.x[1, 0] = cx, cy
            self.initialized = True
            return
        y = np.array([[cx], [cy]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def mahalanobis2(self, cx: float, cy: float) -> float:
        """观测相对当前预测的马氏距离²（调用方先 predict）。"""
        S = self.H @ self.P @ self.H.T + self.R
        innov = np.array([[cx - self.x[0, 0]], [cy - self.x[1, 0]]])
        return float((innov.T @ np.linalg.inv(S) @ innov)[0, 0])

    def velocity(self) -> tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])


# ── 候选评分（与 tracker_utils.score_candidates 同口径的轻量版）──

def score_candidates(candidates: list[tuple[float, float, float, float, float]],
                     prev_pt: tuple[float, float] | None,
                     pred_pt: tuple[float, float] | None,
                     prev_area: float | None) -> int:
    """candidates: [(cx, cy, area, aspect_ratio, conf), ...]。返回最优索引或 -1。

    跟踪态（pred_pt 非 None）不做 MIN_SCORE 硬淘汰——门控/速度维度负责拒野值，
    评分只负责"选最优"；搜索态（无预测）才用 MIN_SCORE 卡准入。
    """
    if not candidates:
        return -1
    best_idx, best_score = 0, -1.0
    for i, (cx, cy, area, aspect, conf) in enumerate(candidates):
        if prev_pt is None or pred_pt is None:
            area_score = max(0.0, min(1.0, 1.0 - abs(math.log2(area / 200.0))))
            score = 0.7 * conf + 0.3 * area_score
            if score < MIN_SCORE:
                continue
        else:
            dist = math.hypot(cx - pred_pt[0], cy - pred_pt[1])
            motion = max(0.0, min(1.0, 1.0 - dist / MAX_EXPECTED_DISPLACEMENT))
            area_cons = 1.0 if prev_area in (None, 0) else max(0.0, 1.0 - abs(math.log2(area / max(prev_area, 1.0))))
            score = 0.45 * motion + 0.35 * conf + 0.20 * area_cons
        if score > best_score:
            best_score, best_idx = score, i
    if best_score < 0:
        return -1
    return best_idx


# ── 主融合函数 ──

def fuse_disc_detections(
    per_frame_dets: list[list[dict]],
    fps: float = 25.0,
    world_projector=None,          # callable(cx, cy) -> (wx, wy) | None
    gate_threshold: float = GATE_THRESHOLD,
    max_speed_ms: float = MAX_SPEED_MS,
    width: float = 1920.0,
    height: float = 1080.0,
) -> tuple[list[DiscFrame], FusionStats]:
    """把逐帧盘检测序列融合成带状态标注的轨迹输出。

    world_projector: 像素→世界投影（无标定时传 None，跳过速度/场内维度）。
    """
    frames_out: list[DiscFrame] = []
    stats = FusionStats()
    kf = Kalman2D()
    track_len = 0
    cur_track = 0
    lost_counter = 0
    last_world: tuple[float, float] | None = None
    prev_area: float | None = None

    for dets in per_frame_dets:
        stats.n_frames += 1
        pred = kf.predict() if kf.initialized else None
        # 门控/丢失降级期间（lost_counter ≤ THRESHOLD 且未破产）仍是"活轨迹"，
        # 保留门控判定资格——否则一次门控就永久哑火只能 predicting。
        is_tracking = kf.initialized and track_len > 0
        is_confident = is_tracking and lost_counter == 0  # 连续可信（用于面积参考等）

        # 候选整理
        cands = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            cands.append(((x1 + x2) / 2, (y1 + y2) / 2, bw * bh, bw / max(bh, 1.0), float(d["conf"])))

        pred_pt = pred if (is_tracking and pred is not None) else None
        best = score_candidates(cands, pred_pt, pred_pt, prev_area if is_confident else None)

        if best < 0:
            # 本帧无可信候选
            if is_tracking:
                lost_counter += 1
                if lost_counter > LOST_TRACK_THRESHOLD:
                    kf = Kalman2D(); track_len = 0; lost_counter = 0
                    stats.n_lost += 1
                    frames_out.append(DiscFrame(None, 0.0, 0.0, 0.0, "searching"))
                    stats.n_searching += 1
                    continue
                px, py = pred
                in_bounds = 0 <= px <= width and 0 <= py <= height
                if in_bounds:
                    frames_out.append(DiscFrame([px - 8, py - 8, px + 8, py + 8], 0.0, px, py, "predicting"))
                    stats.n_predicting += 1
                else:
                    kf = Kalman2D(); track_len = 0; lost_counter = 0
                    frames_out.append(DiscFrame(None, 0.0, 0.0, 0.0, "searching"))
                    stats.n_searching += 1
            else:
                frames_out.append(DiscFrame(None, 0.0, 0.0, 0.0, "searching"))
                stats.n_searching += 1
            continue

        cx, cy, area, _aspect, conf = cands[best]
        bbox = dets[best]["bbox"]

        # ── 维度1：世界坐标速度物理拒绝（先于马氏门控——物理定律优先）──
        wx = wy = None
        if world_projector is not None:
            try:
                proj = world_projector(cx, cy)
                if proj is not None:
                    wx, wy = float(proj[0]), float(proj[1])
            except Exception:
                wx = wy = None
        speed = None
        if wx is not None and last_world is not None and is_tracking:
            speed = math.hypot(wx - last_world[0], wy - last_world[1]) / max(fps, 1e-6)
            if speed > max_speed_ms:
                stats.n_rejected_speed += 1
                # 速度物理不可能 → 视同野值：降级纯预测，不更新 last_world
                lost_counter += 1
                px, py = pred
                frames_out.append(DiscFrame([px - 8, py - 8, px + 8, py + 8], conf, px, py, "rejected",
                                            speed_ms=round(speed, 1), world_xy=(wx, wy)))
                if lost_counter > LOST_TRACK_THRESHOLD:
                    kf = Kalman2D(); track_len = 0; lost_counter = 0
                continue

        # ── 维度2：马氏门控（仅跟踪态；野值拒绝→降级纯预测，不污染滤波器）──
        d2 = None
        if is_tracking and pred_pt is not None and track_len >= 2:
            d2 = kf.mahalanobis2(cx, cy)
            if d2 > gate_threshold:
                lost_counter += 1
                stats.n_gated += 1
                px, py = pred
                frames_out.append(DiscFrame([px - 8, py - 8, px + 8, py + 8], conf, px, py, "gated", d2=round(d2, 1)))
                if lost_counter > LOST_TRACK_THRESHOLD:
                    kf = Kalman2D(); track_len = 0; lost_counter = 0
                prev_area = None  # 门控后面积参考失效
                continue

        # ── 接受观测 ──
        kf.correct(cx, cy)
        lost_counter = 0
        track_len += 1
        cur_track = max(cur_track, 1)
        prev_area = area
        if wx is not None:
            last_world = (wx, wy)
        vx, vy = kf.velocity()
        speed_out = math.hypot(vx, vy) / max(fps, 1e-6) * (1.0)  # px/帧→px/s，仅参考
        frames_out.append(DiscFrame(bbox, conf, cx, cy, "tracking",
                                    d2=round(d2, 1) if d2 is not None else None,
                                    speed_ms=round(speed, 1) if speed is not None else None,
                                    world_xy=(wx, wy) if wx is not None else None))
        stats.n_tracking += 1
        stats.longest_track_frames = max(stats.longest_track_frames, track_len)

    stats.n_tracks = 1 if stats.n_tracking > 0 else 0
    stats.details["fps"] = fps
    stats.details["width"] = width
    stats.details["height"] = height
    return frames_out, stats
