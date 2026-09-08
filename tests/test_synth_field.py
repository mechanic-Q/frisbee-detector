"""synth_field 合成渲染包单元测试（纯 CPU，无渲染执行）。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from synth_field import camera as cam_mod  # noqa: E402
from synth_field import field_model as fm  # noqa: E402


def test_field_model_invariants():
    segs = fm.line_segments()
    assert len(segs) == 6  # 边线4 + 得分区线2
    kps = fm.keypoints()
    assert len(kps) == 8
    bricks = fm.brick_marks()
    assert bricks == [(36.0, 0.0), (36.0, 37.0), (64.0, 0.0), (64.0, 37.0)]
    assert all(0 <= x <= fm.FIELD_W and 0 <= y <= fm.FIELD_H for x, y in kps)


def test_camera_deterministic_with_same_seed():
    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(7)
    c1 = cam_mod.sample_camera(r1)
    c2 = cam_mod.sample_camera(r2)
    assert set(c1) == set(c2)
    for k in c1:
        if isinstance(c1[k], np.ndarray):
            assert np.array_equal(c1[k], c2[k]), k
        else:
            assert c1[k] == c2[k], k


def test_project_shape_and_finiteness():
    rng = np.random.default_rng(0)
    cam = cam_mod.sample_camera(rng)
    K = cam_mod.build_k(rng, cam)
    rvec, tvec = cam_mod.build_rt(cam)
    pts = np.array(fm.corners(), dtype=np.float64)
    out = cam_mod.project(pts, K, rvec, tvec, cam)
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


def test_homography_consistency_direct_vs_indirect():
    """地面点经直接投影 与 经四角单应性变换 必须一致（同一平面射影映射）。"""
    rng = np.random.default_rng(3)
    cam = cam_mod.sample_camera(rng)
    K = cam_mod.build_k(rng, cam)
    rvec, tvec = cam_mod.build_rt(cam)
    corners_w = np.array(fm.corners(), dtype=np.float64)
    corners_px = cam_mod.project(corners_w, K, rvec, tvec, cam)
    H, _ = cv2.findHomography(corners_w, corners_px, 0)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        pytest.skip("degenerate camera configuration")
    interior = np.array([[50, 18.5], [10, 5], [90, 30], [36, 18.5]], dtype=np.float64)
    direct = cam_mod.project(interior, K, rvec, tvec, cam)
    wrapped = cv2.perspectiveTransform(interior.reshape(-1, 1, 2), H).reshape(-1, 2)
    # 尺度容差：极端相机下像素坐标可达数千，半像素级一致即视为同一射影映射
    assert np.allclose(direct, wrapped, rtol=1e-4, atol=0.5)
