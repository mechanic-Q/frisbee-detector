"""Homography calibration: pixel-to-field coordinate mapping.

Functions:
    compute_homography  — compute 3x3 transform from matched points
    pixel_to_world      — image pixel → field coordinate (meters)
    world_to_pixel      — field coordinate → image pixel
    draw_field_overlay  — project standard field lines onto image
    warp_to_birdseye    — generate top-down bird's-eye view
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
LINE_SAMPLES = 100


def draw_field_overlay(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return overlay
    for line in FIELD_LINES_WORLD:
        pixels = []
        for i in range(LINE_SAMPLES + 1):
            t = i / LINE_SAMPLES
            wx = line[0][0] + t * (line[1][0] - line[0][0])
            wy = line[0][1] + t * (line[1][1] - line[0][1])
            px, py = world_to_pixel(matrix, wx, wy)
            if not (np.isnan(px) or np.isnan(py)):
                pixels.append([int(round(px)), int(round(py))])
        if len(pixels) > 1:
            cv2.polylines(overlay, [np.array(pixels)], False, (0, 255, 0), 2, cv2.LINE_AA)
    return overlay


def warp_to_birdseye(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return image
    return cv2.warpPerspective(image, inv_matrix, BIRDSEYE_SIZE)


_CALIBRATION_SCHEMA_KEYS = {
    "video", "image_size", "field_size_m", "calibration_frame",
    "points", "matrix", "reprojection_error_px",
}


def save_calibration(
    path: Path | str,
    video: str,
    image_size: list[int],
    field_size_m: list[float],
    calibration_frame: int,
    points: list[dict],
    matrix: np.ndarray,
    reprojection_error_px: float,
) -> None:
    """Save calibration data to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video": video,
        "image_size": image_size,
        "field_size_m": field_size_m,
        "calibration_frame": calibration_frame,
        "points": points,
        "matrix": matrix.tolist(),
        "reprojection_error_px": reprojection_error_px,
    }
    path.write_text(json.dumps(data, indent=2))


def load_calibration(path: Path | str) -> dict:
    """Load and validate calibration data from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    missing = _CALIBRATION_SCHEMA_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in {path}: {missing}")

    if not data["points"]:
        raise ValueError(f"Empty points list in {path}")

    mat = np.array(data["matrix"])
    if mat.shape != (3, 3):
        raise ValueError(f"Matrix must be 3x3, got {mat.shape}")

    data["matrix"] = mat
    return data
