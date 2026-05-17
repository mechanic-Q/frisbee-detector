"""Homography calibration: pixel-to-field coordinate mapping.

Functions:
    compute_homography  — compute 3x3 transform from matched points
    pixel_to_world      — image pixel → field coordinate (meters)
    world_to_pixel      — field coordinate → image pixel
"""

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
        matrix, _ = cv2.findHomography(
            pixel_arr, world_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

    if matrix is None:
        return None, float("inf")

    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None, float("inf")
    reprojected_pixels = cv2.perspectiveTransform(world_arr.reshape(1, -1, 2), inv_matrix).reshape(-1, 2)
    errors = np.linalg.norm(reprojected_pixels - pixel_arr, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return matrix, rmse


def pixel_to_world(matrix: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Convert pixel coordinates to world coordinates."""
    pt = np.array([[[px, py]]], dtype=np.float64)
    result = cv2.perspectiveTransform(pt, matrix).reshape(2)
    return float(result[0]), float(result[1])


def world_to_pixel(matrix: np.ndarray, wx: float, wy: float) -> tuple[float, float]:
    """Convert world coordinates to pixel coordinates."""
    try:
        inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    pt = np.array([[[wx, wy]]], dtype=np.float64)
    result = cv2.perspectiveTransform(pt, inv).reshape(2)
    return float(result[0]), float(result[1])


FIELD_LINES_WORLD = [
    [(0, 0), (100, 0)],
    [(100, 0), (100, 37)],
    [(100, 37), (0, 37)],
    [(0, 37), (0, 0)],
    [(0, 18.5), (100, 18.5)],
]

BIRDSEYE_SIZE = (1000, 370)


def draw_field_overlay(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    pts_per_line = 100
    for line in FIELD_LINES_WORLD:
        pixels = []
        for i in range(pts_per_line + 1):
            t = i / pts_per_line
            wx = line[0][0] + t * (line[1][0] - line[0][0])
            wy = line[0][1] + t * (line[1][1] - line[0][1])
            px, py = world_to_pixel(matrix, wx, wy)
            if not (np.isnan(px) or np.isnan(py)):
                pixels.append([int(round(px)), int(round(py))])
        if len(pixels) > 1:
            cv2.polylines(overlay, [np.array(pixels)], False, (0, 255, 0), 2, cv2.LINE_AA)
    return overlay


def warp_to_birdseye(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    inv_matrix = np.linalg.inv(matrix)
    return cv2.warpPerspective(image, inv_matrix, BIRDSEYE_SIZE)
