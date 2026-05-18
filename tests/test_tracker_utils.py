"""Tests for single-frisbee tracker utilities."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tracker_utils import init_kalman, score_candidates, Trajectory


def test_kalman_init_state():
    kf = init_kalman()
    state = kf.statePost  # (4, 1) after init
    assert state.shape == (4, 1), f"Expected (4, 1), got {state.shape}"


def test_kalman_predict_update():
    kf = init_kalman()
    obs = np.array([[320.0], [240.0]], dtype=np.float32)
    for _ in range(20):
        kf.predict()
        kf.correct(obs)
    state = kf.statePost
    assert abs(float(state[0, 0]) - 320.0) < 10.0, f"px drifted: {float(state[0, 0])}"
    assert abs(float(state[1, 0]) - 240.0) < 10.0, f"py drifted: {float(state[1, 0])}"


def test_trajectory_push_window():
    traj = Trajectory(maxlen=100)
    for i in range(60):
        traj.push(float(i), float(i), 200.0)
    window = traj.get_window(50)
    assert len(window) == 50
    assert window[0] == (10.0, 10.0)
    assert window[-1] == (59.0, 59.0)


def test_trajectory_ring_behavior():
    traj = Trajectory(maxlen=10)
    for i in range(20):
        traj.push(float(i), float(i), 200.0)
    assert len(traj.get_window(50)) == 10
    assert traj.last_position() == (19.0, 19.0)
    assert traj.areas == [200.0] * 10


def test_trajectory_empty():
    traj = Trajectory()
    assert traj.last_position() is None
    assert traj.get_window() == []


def test_score_prioritizes_confidence():
    cands = [
        {"box": [100, 200, 120, 230], "conf": 0.9},
        {"box": [300, 400, 320, 420], "conf": 0.3},
    ]
    best = score_candidates(cands, None, None)
    assert best == 0, "should pick higher confidence"


def test_score_uses_motion():
    traj = Trajectory()
    traj.push(100, 100, 200)
    cands = [
        {"box": [105, 105, 125, 125], "conf": 0.5},
        {"box": [500, 500, 520, 520], "conf": 0.9},
    ]
    best = score_candidates(cands, traj, traj.last_position())
    assert best == 0, "should pick motion-consistent even with lower conf"


def test_aspect_score_favors_wide_boxes():
    traj = Trajectory()
    traj.push(100, 100, 200)
    tall_box = {"box": [100, 100, 110, 140], "conf": 0.5}
    wide_box = {"box": [100, 100, 120, 120], "conf": 0.5}
    cands = [tall_box, wide_box]
    best = score_candidates(cands, traj, (105, 105))
    assert best == 1, "wide (square-like) box should win over tall box"


def test_score_empty_candidates():
    best = score_candidates([], None, None)
    assert best == -1, "empty candidates should return -1"
