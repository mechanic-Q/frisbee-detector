"""Homography calibration: pixel-to-field coordinate mapping.

Functions:
    compute_homography  — compute 3x3 transform from matched points
    pixel_to_world      — image pixel → field coordinate (meters)
    world_to_pixel      — field coordinate → image pixel
    draw_field_overlay  — draw standard field lines on image
    warp_to_birdseye    — generate top-down view of the field
    save_calibration    — persist calibration to JSON
    load_calibration    — load calibration from JSON
"""

import json
from pathlib import Path

import cv2
import numpy as np


def compute_homography(points: list[tuple[float, float, float, float]]) -> tuple[np.ndarray | None, float]:
    """Compute homography from matched pixel/world points.

    Args:
        points: list of (pixel_x, pixel_y, world_x, world_y) tuples.
                4 points → exact solve; 5+ → RANSAC.

    Returns:
        (3x3 matrix, reprojection_rmse_px) or (None, inf) on failure.
    """
    if len(points) < 4:
        return None, float("inf")

    pixel_arr = np.array([[p[0], p[1]] for p in points], dtype=np.float64)
    world_arr = np.array([[p[2], p[3]] for p in points], dtype=np.float64)

    if len(points) == 4:
        matrix = cv2.getPerspectiveTransform(
            pixel_arr.astype(np.float32),
            world_arr.astype(np.float32),
        )
    else:
        matrix, mask = cv2.findHomography(
            pixel_arr, world_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

    if matrix is None:
        return None, float("inf")

    projected = cv2.perspectiveTransform(pixel_arr.reshape(1, -1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - world_arr, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return matrix, rmse


def pixel_to_world(matrix: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Convert pixel coordinates to world coordinates."""
    pt = np.array([[[px, py]]], dtype=np.float64)
    result = cv2.perspectiveTransform(pt, matrix).reshape(2)
    return float(result[0]), float(result[1])


def world_to_pixel(matrix: np.ndarray, wx: float, wy: float) -> tuple[float, float]:
    """Convert world coordinates to pixel coordinates."""
    inv = np.linalg.inv(matrix)
    pt = np.array([[[wx, wy]]], dtype=np.float64)
    result = cv2.perspectiveTransform(pt, inv).reshape(2)
    return float(result[0]), float(result[1])
