"""Single-frisbee tracker: Kalman filter + candidate scoring + trajectory.

Functions:
    init_kalman          — create and configure cv2.KalmanFilter(4, 2)
    score_candidates     — pick the best candidate box per frame
    Trajectory           — ring buffer of tracked positions + areas
"""

from collections import deque

import cv2
import numpy as np


MAX_EXPECTED_DISPLACEMENT = 50.0  # px, at 25fps
MIN_DISPLACEMENT = 5.0  # px/frame, threshold for "moving"
REFERENCE_AREA = 200.0  # px², rough frisbee box area in 1280×720 video
MIN_SCORE = 0.3  # minimum score to accept any candidate


def init_kalman() -> cv2.KalmanFilter:
    """Create a 6-state Kalman filter (px,py,vx,vy,ax,ay) for position+velocity+acceleration tracking."""
    kf = cv2.KalmanFilter(6, 2)
    dt = 1.0
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0.5*dt*dt, 0],
        [0, 1, 0, dt, 0, 0.5*dt*dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 100.0
    return kf



H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32)

GATE_THRESHOLD = 500.0   # catch only extreme outliers   # empirically safe for 60fps frisbee motion  # chi²_{0.999}(df=2) — only reject extreme outliers


def mahalanobis_gate(
    kf: cv2.KalmanFilter,
    prediction: tuple[float, float],
    measurement: tuple[float, float],
) -> float:
    innov = np.array([
        [measurement[0] - prediction[0]],
        [measurement[1] - prediction[1]],
    ], dtype=np.float32)
    P = kf.errorCovPost
    S = H @ P @ H.T + kf.measurementNoiseCov
    S_inv = np.linalg.inv(S)
    return float((innov.T @ S_inv @ innov)[0, 0])

def score_candidates(
    candidates: list[dict],
    trajectory: 'Trajectory | None',
    prediction: tuple[float, float] | None,
) -> int:
    """Return index of the best candidate, or -1 if empty."""
    if not candidates:
        return -1

    best_idx = 0
    best_score = -1.0

    for i, cand in enumerate(candidates):
        conf = cand.get("conf", 0.0)
        box = cand.get("box", [0, 0, 0, 0])
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        bw = float(box[2]) - float(box[0])
        bh = float(box[3]) - float(box[1])
        area = bw * bh

        if trajectory is None or prediction is None:
            area_size_score = max(0.0, min(1.0, 1.0 - abs(np.log2(area / 200.0))))
            score = 0.7 * conf + 0.3 * area_size_score
        else:
            dist = np.sqrt((cx - prediction[0]) ** 2 + (cy - prediction[1]) ** 2)
            motion_score = max(0.0, min(1.0, 1.0 - dist / MAX_EXPECTED_DISPLACEMENT))

            if trajectory.areas:
                prev_area = float(trajectory.areas[-1])
                area_consistent_score = max(0.0, 1.0 - abs(np.log2(area / max(prev_area, 1.0))))
            else:
                area_consistent_score = 1.0

            aspect_score = max(0.0, 1.0 - abs(np.log2(bw / max(bh, 1.0))))

            pts_list = list(trajectory._pts)
            if len(pts_list) >= 3:
                recent = pts_list[-3:]
                dx = recent[-1][0] - recent[0][0]
                dy = recent[-1][1] - recent[0][1]
                avg_speed = np.sqrt(dx ** 2 + dy ** 2) / max(len(recent) - 1, 1)
                trajectory_speed_bonus = min(1.0, avg_speed / MIN_DISPLACEMENT)
            else:
                trajectory_speed_bonus = 0.5
            speed_score = min(1.0, dist / MIN_DISPLACEMENT)

            score = (0.15 * motion_score + 0.15 * conf
                     + 0.15 * area_consistent_score + 0.10 * aspect_score
                     + 0.25 * speed_score + 0.20 * trajectory_speed_bonus)

        if score > best_score:
            best_score = score
            best_idx = i

    if best_score < MIN_SCORE:
        return -1
    return best_idx


class Trajectory:
    """Ring buffer of tracked positions and areas."""

    def __init__(self, maxlen: int = 2000):
        self._pts = deque(maxlen=maxlen)
        self.areas: list[float] = []

    def push(self, px: float, py: float, area: float) -> None:
        self._pts.append((px, py))
        self.areas.append(area)
        if len(self.areas) > len(self._pts):
            self.areas.pop(0)

    def get_window(self, n: int = 50) -> list[tuple[float, float]]:
        return list(self._pts)[-n:]

    def last_position(self) -> tuple[float, float] | None:
        return self._pts[-1] if self._pts else None
