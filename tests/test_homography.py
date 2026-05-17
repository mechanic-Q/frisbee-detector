"""Tests for homography calibration utilities."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.homography import compute_homography, pixel_to_world, world_to_pixel, draw_field_overlay, warp_to_birdseye


def test_roundtrip_known_transform():
    pixel_pts = np.array([
        [100, 500],
        [1180, 500],
        [1180, 50],
        [100, 50],
    ], dtype=np.float64)

    world_pts = np.array([
        [0, 0],
        [100, 0],
        [100, 37],
        [0, 37],
    ], dtype=np.float64)

    points = [(p[0], p[1], w[0], w[1]) for p, w in zip(pixel_pts, world_pts)]
    matrix, error = compute_homography(points)

    assert matrix is not None
    assert error < 1.0

    for px, py, wx, wy in points:
        result_wx, result_wy = pixel_to_world(matrix, px, py)
        assert abs(result_wx - wx) < 0.5, f"wx: {result_wx} != {wx}"
        assert abs(result_wy - wy) < 0.5, f"wy: {result_wy} != {wy}"

        result_px, result_py = world_to_pixel(matrix, wx, wy)
        assert abs(result_px - px) < 1.0, f"px: {result_px} != {px}"
        assert abs(result_py - py) < 1.0, f"py: {result_py} != {py}"


def test_collinear_points_return_none():
    collinear = [
        (100, 100, 0, 0),
        (200, 200, 50, 18.5),
        (300, 300, 100, 37),
        (400, 400, 50, 18.5),
    ]
    matrix, error = compute_homography(collinear)
    assert matrix is None or error > 50.0


def test_ransac_five_points():
    pixel_pts = [
        (100, 500),
        (640, 500),
        (1180, 500),
        (1180, 50),
        (100, 50),
    ]
    world_pts = [
        (0, 0),
        (50, 0),
        (100, 0),
        (100, 37),
        (0, 37),
    ]
    points = [(p[0], p[1], w[0], w[1]) for p, w in zip(pixel_pts, world_pts)]
    matrix, error = compute_homography(points)
    assert matrix is not None
    assert error < 2.0

    for px, py, wx, wy in points:
        rwx, rwy = pixel_to_world(matrix, px, py)
        assert abs(rwx - wx) < 1.0
        assert abs(rwy - wy) < 1.0


def _make_test_matrix():
    points = [
        (100, 500, 0, 0),
        (1180, 500, 100, 0),
        (1180, 50, 100, 37),
        (100, 50, 0, 37),
    ]
    matrix, _ = compute_homography(points)
    return matrix


def test_draw_field_overlay_returns_same_size():
    matrix = _make_test_matrix()
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = draw_field_overlay(img, matrix)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert np.any(result > 0)


def test_warp_to_birdseye_returns_fixed_size():
    matrix = _make_test_matrix()
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = warp_to_birdseye(img, matrix)
    assert result.shape == (370, 1000, 3)
    assert result.dtype == np.uint8

