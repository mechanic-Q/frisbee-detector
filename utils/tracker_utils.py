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


def init_kalman() -> cv2.KalmanFilter:
    """Create a 4-state Kalman filter for position + velocity tracking."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.003
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
    return kf


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
                speed_score = min(5.0, dist / MIN_DISPLACEMENT)
            else:
                speed_score = 0.5

            score = (0.30 * motion_score + 0.20 * conf
                     + 0.20 * area_consistent_score + 0.15 * aspect_score
                     + 0.15 * speed_score)

        if score > best_score:
            best_score = score
            best_idx = i

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
