# Homography 场地标定工具 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 创建 Streamlit 标定工具 + utils/homography.py 核心库，将视频帧像素坐标映射到飞盘场地真实坐标。

**架构：** `utils/homography.py` 提供纯函数（计算、坐标变换、可视化），`tools/calibrate_field.py` 是 Streamlit UI 调用这些函数。JSON 持久化到 `configs/homography/`。

**技术栈：** Python 3, OpenCV (cv2), NumPy, Streamlit, pytest

**规格文档：** `docs/superpowers/specs/2026-05-18-homography-calibration-design.md`

---

## 文件结构

| 文件 | 职责 | 状态 |
|------|------|------|
| `utils/homography.py` | Homography 计算、坐标变换、场地线绘制、鸟瞰图、JSON 加载/保存 | 新建 |
| `tools/calibrate_field.py` | Streamlit UI：加载帧、点选控制点、输入坐标、计算、可视化、保存 | 新建 |
| `tests/test_homography.py` | 核心函数单元测试 | 新建 |
| `configs/homography/` | 标定结果 JSON 存储目录 | 新建（空） |

---

### 任务 1：compute_homography + pixel_to_world + world_to_pixel

**文件：**
- 创建：`utils/homography.py`
- 创建：`tests/test_homography.py`

- [ ] **步骤 1：编写失败的测试 — 正反变换互逆**

创建 `tests/test_homography.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils.homography import compute_homography, pixel_to_world, world_to_pixel


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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_homography.py::test_roundtrip_known_transform -v`
预期：FAIL — `ImportError: cannot import name 'compute_homography'`

- [ ] **步骤 3：实现 compute_homography、pixel_to_world、world_to_pixel**

创建 `utils/homography.py`：

```python
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
        matrix = cv2.getPerspectiveTransform(pixel_arr, world_arr)
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_homography.py::test_roundtrip_known_transform -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add utils/homography.py tests/test_homography.py
git commit -m "feat: add compute_homography, pixel_to_world, world_to_pixel"
```

---

### 任务 2：共线点检测 + RANSAC 多点测试

**文件：**
- 修改：`tests/test_homography.py`
- 修改：`utils/homography.py`（如果需要加强错误处理）

- [ ] **步骤 1：编写失败测试 — 共线点报错**

追加到 `tests/test_homography.py`：

```python
def test_collinear_points_return_none():
    collinear = [
        (100, 100, 0, 0),
        (200, 200, 50, 18.5),
        (300, 300, 100, 37),
        (400, 400, 50, 18.5),
    ]
    matrix, error = compute_homography(collinear)
    assert matrix is None or error > 50.0
```

- [ ] **步骤 2：运行测试验证通过/失败**

运行：`python3 -m pytest tests/test_homography.py::test_collinear_points_return_none -v`
预期：可能 PASS（OpenCV 可能返回退化矩阵）。如果 PASS 继续；如果 FAIL 则加强 `compute_homography` 的退化检测。

- [ ] **步骤 3：编写 RANSAC 多点测试**

追加到 `tests/test_homography.py`：

```python
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
```

- [ ] **步骤 4：运行全部测试**

运行：`python3 -m pytest tests/test_homography.py -v`
预期：3 个测试全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add tests/test_homography.py
git commit -m "test: add collinear and RANSAC tests for homography"
```

---

### 任务 3：draw_field_overlay + warp_to_birdseye

**文件：**
- 修改：`utils/homography.py`

- [ ] **步骤 1：编写失败测试 — 场地线绘制**

追加到 `tests/test_homography.py`：

```python
from utils.homography import draw_field_overlay, warp_to_birdseye


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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_homography.py::test_draw_field_overlay_returns_same_size tests/test_homography.py::test_warp_to_birdseye_returns_fixed_size -v`
预期：FAIL — `ImportError: cannot import name 'draw_field_overlay'`

- [ ] **步骤 3：实现 draw_field_overlay 和 warp_to_birdseye**

追加到 `utils/homography.py`：

```python
FIELD_LINES_WORLD = [
    [(0, 0), (100, 0)],
    [(100, 0), (100, 37)],
    [(100, 37), (0, 37)],
    [(0, 37), (0, 0)],
    [(0, 18.5), (100, 18.5)],
]


def draw_field_overlay(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Draw standard field lines projected onto the image."""
    overlay = image.copy()
    pts_per_line = 100
    for line in FIELD_LINES_WORLD:
        pixels = []
        for i in range(pts_per_line + 1):
            t = i / pts_per_line
            wx = line[0][0] + t * (line[1][0] - line[0][0])
            wy = line[0][1] + t * (line[1][1] - line[0][1])
            px, py = world_to_pixel(matrix, wx, wy)
            pixels.append([int(round(px)), int(round(py))])
        cv2.polylines(overlay, [np.array(pixels)], False, (0, 255, 0), 2)
    return overlay


BIRDSEYE_SIZE = (1000, 370)


def warp_to_birdseye(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Warp image to bird's-eye view of the field."""
    inv_matrix = np.linalg.inv(matrix)
    return cv2.warpPerspective(image, inv_matrix, BIRDSEYE_SIZE)
```

注意：`warp_to_birdseye` 使用逆矩阵（world→pixel），因为 `warpPerspective` 从目标坐标查源像素。`draw_field_overlay` 的场地线定义包含 4 条边线 + 1 条中线，不含得分线（标准飞盘场地得分线位置不固定，中线足够验证标定质量）。

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_homography.py -v`
预期：全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add utils/homography.py tests/test_homography.py
git commit -m "feat: add draw_field_overlay and warp_to_birdseye"
```

---

### 任务 4：save_calibration + load_calibration

**文件：**
- 修改：`utils/homography.py`
- 修改：`tests/test_homography.py`

- [ ] **步骤 1：编写失败测试 — JSON 保存/加载往返**

追加到 `tests/test_homography.py`：

```python
import tempfile

from utils.homography import save_calibration, load_calibration


def test_save_load_roundtrip(tmp_path):
    matrix = _make_test_matrix()
    points = [
        {"pixel": [100, 500], "world": [0, 0]},
        {"pixel": [1180, 500], "world": [100, 0]},
        {"pixel": [1180, 50], "world": [100, 37]},
        {"pixel": [100, 50], "world": [0, 37]},
    ]
    save_calibration(
        path=tmp_path / "test.json",
        video="test.mp4",
        image_size=[1280, 720],
        field_size_m=[100, 37],
        calibration_frame=0,
        points=points,
        matrix=matrix,
        reprojection_error_px=0.5,
    )

    data = load_calibration(tmp_path / "test.json")
    assert data["video"] == "test.mp4"
    assert data["image_size"] == [1280, 720]
    assert data["field_size_m"] == [100, 37]
    assert len(data["points"]) == 4
    assert np.allclose(data["matrix"], matrix)
    assert abs(data["reprojection_error_px"] - 0.5) < 0.01


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_calibration(Path("/tmp/nonexistent_homography.json"))


def test_load_invalid_json_raises(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json")
    with pytest.raises(ValueError):
        load_calibration(bad_file)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_homography.py::test_save_load_roundtrip -v`
预期：FAIL — `ImportError: cannot import name 'save_calibration'`

- [ ] **步骤 3：实现 save_calibration 和 load_calibration**

追加到 `utils/homography.py`：

```python
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
```

- [ ] **步骤 4：运行全部测试**

运行：`python3 -m pytest tests/test_homography.py -v`
预期：全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add utils/homography.py tests/test_homography.py
git commit -m "feat: add save_calibration and load_calibration with JSON validation"
```

---

### 任务 5：创建 configs/homography/ 目录

**文件：**
- 创建：`configs/homography/.gitkeep`

- [ ] **步骤 1：创建目录并添加 .gitkeep**

```bash
mkdir -p configs/homography
touch configs/homography/.gitkeep
```

- [ ] **步骤 2：Commit**

```bash
git add configs/homography/.gitkeep
git commit -m "chore: add configs/homography/ directory for calibration data"
```

---

### 任务 6：Streamlit 标定 UI

**文件：**
- 创建：`tools/calibrate_field.py`

这是最大的任务。Streamlit UI 没有传统意义的单元测试——通过手动运行验证。

- [ ] **步骤 1：创建 calibrate_field.py — 骨架 + 参数解析 + 帧提取**

创建 `tools/calibrate_field.py`：

```python
"""Field calibration tool using Streamlit.

Click field line intersections on a video frame, enter real-world coordinates,
compute homography, verify with overlay and bird's-eye view.

Usage:
    streamlit run tools/calibrate_field.py -- --video movie/25866279684-1-192.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import cv2
import numpy as np
import streamlit as st

from utils.homography import (
    compute_homography,
    draw_field_overlay,
    load_calibration,
    pixel_to_world,
    save_calibration,
    warp_to_birdseye,
    world_to_pixel,
)

st.set_page_config(page_title="Field Calibration", layout="wide")

FIELD_W, FIELD_H = 100, 37
BIRDSEYE_PX_PER_M = 10
```

- [ ] **步骤 2：添加参数解析和帧提取函数**

追加到 `tools/calibrate_field.py`：

```python
def get_args():
    parser = argparse.ArgumentParser(description="Field calibration tool")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to use for calibration (default: 0)",
    )
    args, _ = parser.parse_known_args()
    return args


@st.cache_data
def extract_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

- [ ] **步骤 3：实现核心 UI 逻辑**

追加到 `tools/calibrate_field.py`：

```python
def main():
    args = get_args()
    video_path = Path(args.video)
    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        st.info("Usage: streamlit run tools/calibrate_field.py -- --video <path>")
        return

    st.title("Field Calibration Tool")
    st.caption(f"Video: {video_path.name}")

    frame = extract_frame(str(video_path), args.frame)
    if frame is None:
        st.error("Failed to extract frame from video")
        return

    h, w = frame.shape[:2]
    st.caption(f"Frame size: {w}x{h} | Field: {FIELD_W}m x {FIELD_H}m | Origin: bottom-left")

    if "points" not in st.session_state:
        st.session_state.points = []

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("Calibration Frame")
        annotated = frame.copy()

        for i, pt in enumerate(st.session_state.points):
            px, py = int(pt["pixel"][0]), int(pt["pixel"][1])
            cv2.circle(annotated, (px, py), 8, (0, 255, 0), 2)
            cv2.putText(annotated, str(i + 1), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        clicked = st.image(annotated, use_container_width=True)

        st.subheader("Add Control Point")
        input_cols = st.columns([1, 1, 1, 1, 1])
        with input_cols[0]:
            px_in = st.number_input("Pixel X", min_value=0, max_value=w, value=w // 2, key="px")
        with input_cols[1]:
            py_in = st.number_input("Pixel Y", min_value=0, max_value=h, value=h // 2, key="py")
        with input_cols[2]:
            wx_in = st.number_input("World X (m)", min_value=0.0, max_value=float(FIELD_W), value=0.0, step=1.0, key="wx")
        with input_cols[3]:
            wy_in = st.number_input("World Y (m)", min_value=0.0, max_value=float(FIELD_H), value=0.0, step=1.0, key="wy")
        with input_cols[4]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add Point", type="primary"):
                st.session_state.points.append({
                    "pixel": [float(px_in), float(py_in)],
                    "world": [float(wx_in), float(wy_in)],
                })
                st.rerun()

        btn_cols = st.columns([1, 1, 1, 1])
        with btn_cols[0]:
            if st.button("Undo Last"):
                if st.session_state.points:
                    st.session_state.points.pop()
                    st.rerun()
        with btn_cols[1]:
            if st.button("Clear All"):
                st.session_state.points = []
                st.rerun()
        with btn_cols[2]:
            compute_clicked = st.button("Compute Homography", type="primary")
        with btn_cols[3]:
            save_clicked = st.button("Save Calibration")

    with col_right:
        st.subheader("Points")
        if not st.session_state.points:
            st.info("No points added yet. Enter pixel/world coordinates and click 'Add Point'.")
        else:
            for i, pt in enumerate(st.session_state.points):
                st.text(f"#{i+1}: px=({pt['pixel'][0]:.0f},{pt['pixel'][1]:.0f}) "
                        f"→ w=({pt['world'][0]:.1f},{pt['world'][1]:.1f})")

    if compute_clicked:
        pts = st.session_state.points
        if len(pts) < 4:
            st.error("Need at least 4 points to compute homography")
            return

        raw_points = [
            (p["pixel"][0], p["pixel"][1], p["world"][0], p["world"][1])
            for p in pts
        ]
        matrix, error = compute_homography(raw_points)

        if matrix is None:
            st.error("Homography computation failed. Points may be nearly collinear.")
            return

        st.session_state.matrix = matrix
        st.session_state.reproj_error = error

        quality = "Excellent" if error < 3 else ("Acceptable" if error < 8 else "Poor — consider recalibrating")
        st.success(f"Homography computed — RMSE: {error:.2f}px ({quality})")

        overlay = draw_field_overlay(frame, matrix)
        birdseye = warp_to_birdseye(frame, matrix)

        vis_cols = st.columns(2)
        with vis_cols[0]:
            st.subheader("Field Line Overlay")
            st.image(overlay, use_container_width=True)
        with vis_cols[1]:
            st.subheader("Bird's-Eye View")
            st.image(birdseye, use_container_width=True)

        st.subheader("Per-Point Errors")
        for i, pt in enumerate(pts):
            px, py = pt["pixel"][0], pt["pixel"][1]
            rwx, rwy = pixel_to_world(matrix, px, py)
            err = ((rwx - pt["world"][0]) ** 2 + (rwy - pt["world"][1]) ** 2) ** 0.5
            st.text(f"#{i+1}: projected=({rwx:.2f},{rwy:.2f}) expected=({pt['world'][0]:.1f},{pt['world'][1]:.1f}) err={err:.2f}m")

    if save_clicked:
        if "matrix" not in st.session_state:
            st.error("Compute homography first")
            return

        output_dir = Path("configs/homography")
        output_path = output_dir / f"{video_path.stem}.json"

        point_errors = []
        matrix = st.session_state.matrix
        for pt in st.session_state.points:
            px, py = pt["pixel"][0], pt["pixel"][1]
            rwx, rwy = pixel_to_world(matrix, px, py)
            err_px = ((rwx - pt["world"][0]) ** 2 + (rwy - pt["world"][1]) ** 2) ** 0.5
            point_errors.append(round(err_px, 4))

        points_with_errors = []
        for pt, err in zip(st.session_state.points, point_errors):
            p = dict(pt)
            p["error_px"] = err
            points_with_errors.append(p)

        save_calibration(
            path=output_path,
            video=video_path.name,
            image_size=[w, h],
            field_size_m=[FIELD_W, FIELD_H],
            calibration_frame=args.frame,
            points=points_with_errors,
            matrix=matrix,
            reprojection_error_px=round(st.session_state.reproj_error, 4),
        )
        st.success(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：手动验证 UI 可启动**

运行：`streamlit run tools/calibrate_field.py -- --video movie/25866279684-1-192.mp4`
预期：浏览器打开，显示视频第一帧，可输入坐标并添加点

- [ ] **步骤 5：手动端到端测试**

1. 添加 4 个控制点（根据帧中可见的场地标线交叉点，输入对应的真实坐标）
2. 点击 "Compute Homography"
3. 检查 RMSE < 3px
4. 检查网格线叠加与实际标线是否重合
5. 检查鸟瞰图是否呈现长方形
6. 点击 "Save Calibration"
7. 检查 `configs/homography/25866279684-1-192.json` 已生成
8. 验证 JSON 内容完整

- [ ] **步骤 6：Commit**

```bash
git add tools/calibrate_field.py
git commit -m "feat: add Streamlit field calibration tool"
```

---

### 任务 7：全部测试回归

**文件：** 无修改

- [ ] **步骤 1：运行完整测试套件**

运行：`python3 -m pytest tests/ -v`
预期：全部 PASS（包括已有的 test_paths.py、test_utils.py、test_verify.py）

- [ ] **步骤 2：验证最终文件结构**

```bash
ls -la utils/homography.py tools/calibrate_field.py tests/test_homography.py configs/homography/
```

预期：所有文件存在
