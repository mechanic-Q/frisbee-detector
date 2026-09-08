# review_labels.py 增强 + 合并 label_frames.py 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `label_frames.py`（手标）和 `review_labels.py`（审核）合并为一个通用标签审核工具，增加 P2 模型对照、进度追踪、撤销、排序、结果导出功能，同时保持向后兼容（不传 `--model` 时行为接近原有 review_labels.py）。从零手标功能由 yololabeler 承担（`tools/review_desktop.py` 桥接），review_labels.py 聚焦审核。

**架构：** 单个 Streamlit 应用 `tools/review_labels.py`，两种审核场景：（1）纯标签审核、（2）P2 模型对照审核。P2 推理结果缓存到 JSON 避免重复计算。输出写入独立目录 `labels/`，保持源标签目录只读。

**技术栈：** Python 3, Streamlit 1.57, OpenCV, Ultralytics YOLO, pytest

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `tools/review_labels.py` | **重写** | 通用帧标注/审核 Streamlit 工具 |
| `tools/label_frames.py` | **删除** | 功能已并入 review_labels.py |
| `tools/_label_utils.py` | **新建** | 标签读写、P2 缓存、IoU 计算等纯函数 |
| `tests/test_label_utils.py` | **新建** | _label_utils 的单元测试 |

---

## 关键设计决策

### 参数接口

```bash
# 模式1：审核自动标签（原有 review_labels.py 用法，向后兼容）
streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames

# 模式2：审核 + P2 对照
streamlit run tools/review_labels.py -- \
  --frames-dir data/datasets/game1080/frames \
  --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt

# 从零手标用 yololabeler（tools/review_desktop.py 桥接）
#   yololabeler data/datasets/<frames_dir>
```

### 目录约定

```
data/datasets/<name>/
  frames/          ← 输入图片
  labels_gsam/     ← 自动标注源（只读，不改）
  labels/          ← 审核输出（accept/reject 写这里）
  p2_cache.json    ← P2 推理缓存
  review_result.json ← 审核完成后的汇总
```

### 输出行为

- **Accept**：复制源标签到 `labels/`（如果 P2 有结果且 IoU>0.3，用 P2 的 bbox）
- **Reject**：写空文件到 `labels/`
- **Skip**：不写任何文件到 `labels/`
- **源标签目录永远不被修改**

---

### 任务 1：创建 `_label_utils.py` 工具函数库

**文件：**
- 创建：`tools/_label_utils.py`
- 创建：`tests/test_label_utils.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_label_utils.py`：

```python
"""Tests for _label_utils: label IO, IoU, P2 cache."""

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from _label_utils import (
    read_label,
    write_label,
    compute_iou,
    load_p2_cache,
    save_p2_cache,
    run_p2_inference,
    load_review_state,
    save_review_state,
)


class TestReadLabel:
    def test_valid_label(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("0 0.500000 0.300000 0.040000 0.060000\n")
        result = read_label(txt)
        assert result is not None
        assert result["cx"] == pytest.approx(0.5)
        assert result["cy"] == pytest.approx(0.3)
        assert result["w"] == pytest.approx(0.04)
        assert result["h"] == pytest.approx(0.06)

    def test_empty_label(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("")
        assert read_label(txt) is None

    def test_missing_file(self, tmp_path):
        txt = tmp_path / "missing.txt"
        assert read_label(txt) is None

    def test_multiline_returns_first(self, tmp_path):
        txt = tmp_path / "multi.txt"
        txt.write_text("0 0.1 0.2 0.3 0.4\n0 0.5 0.6 0.1 0.1\n")
        result = read_label(txt)
        assert result["cx"] == pytest.approx(0.1)


class TestWriteLabel:
    def test_write_positive(self, tmp_path):
        txt = tmp_path / "out.txt"
        write_label(txt, {"cx": 0.5, "cy": 0.3, "w": 0.04, "h": 0.06})
        assert txt.read_text() == "0 0.500000 0.300000 0.040000 0.060000\n"

    def test_write_negative(self, tmp_path):
        txt = tmp_path / "out.txt"
        write_label(txt, None)
        assert txt.read_text() == ""


class TestComputeIoU:
    def test_same_box(self):
        assert compute_iou(0.5, 0.5, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_iou(0.2, 0.5, 0.1, 0.1, 0.8, 0.5, 0.1, 0.1) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = compute_iou(0.5, 0.5, 0.2, 0.2, 0.55, 0.55, 0.2, 0.2)
        assert 0.0 < iou < 1.0

    def test_zero_size_box(self):
        assert compute_iou(0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.1, 0.1) == pytest.approx(0.0)


class TestP2Cache:
    def test_save_and_load(self, tmp_path):
        cache = {"frame_0001": {"cx": 0.5, "cy": 0.3, "w": 0.04, "h": 0.06, "conf": 0.8}}
        path = tmp_path / "p2_cache.json"
        save_p2_cache(path, cache)
        loaded = load_p2_cache(path)
        assert loaded["frame_0001"]["conf"] == pytest.approx(0.8)

    def test_load_missing_returns_empty(self, tmp_path):
        path = tmp_path / "missing.json"
        assert load_p2_cache(path) == {}


class TestReviewState:
    def test_save_and_load(self, tmp_path):
        state = {
            "accept": ["frame_0001", "frame_0002"],
            "reject": ["frame_0003"],
            "skip": [],
        }
        path = tmp_path / "review_result.json"
        save_review_state(path, state)
        loaded = load_review_state(path)
        assert "frame_0001" in loaded["accept"]
        assert "frame_0003" in loaded["reject"]


class TestSortFrames:
    def test_sort_by_iou_desc(self):
        p2 = {
            "f1": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "conf": 0.8},
            "f2": {"cx": 0.6, "cy": 0.6, "w": 0.1, "h": 0.1, "conf": 0.9},
        }
        src = {
            "f1": {"cx": 0.51, "cy": 0.51, "w": 0.1, "h": 0.1},
        }
        from pathlib import Path
        f1 = Path("/frames/f1.jpg")
        f2 = Path("/frames/f2.jpg")
        f3 = Path("/frames/f3.jpg")
        result = sort_frames([f3, f2, f1], p2, src, "iou_desc")
        # f1 has IoU > 0, f2 has only P2 (no src), f3 has nothing
        assert result[0] == f1
        assert result[-1] in (f2, f3)

    def test_sort_by_bbox_size(self):
        src = {
            "large": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.3},
            "small": {"cx": 0.5, "cy": 0.5, "w": 0.01, "h": 0.01},
        }
        from pathlib import Path
        large = Path("/frames/large.jpg")
        small = Path("/frames/small.jpg")
        result = sort_frames([small, large], {}, src, "bbox_size")
        assert result[0] == large
        assert result[1] == small

    def test_sort_unlabeled_last(self):
        src = {
            "labeled": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
        }
        from pathlib import Path
        lab = Path("/frames/labeled.jpg")
        unlab = Path("/frames/unlabeled.jpg")
        result = sort_frames([unlab, lab], {}, src, "iou_desc")
        assert result[0] == lab
        assert result[1] == unlab
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_label_utils.py -v`
预期：FAIL — `ModuleNotFoundError: No module named '_label_utils'`

- [ ] **步骤 3：编写实现代码**

创建 `tools/_label_utils.py`：

```python
"""Shared utilities for label review tools.

Pure functions for label I/O, IoU computation, P2 cache, and review state.
No Streamlit dependency — safe to unit test.
"""

import json
from pathlib import Path


def read_label(txt_path: Path) -> dict | None:
    """Read first YOLO bbox from a label file. Returns None if empty/missing."""
    if not txt_path.exists():
        return None
    content = txt_path.read_text().strip()
    if not content:
        return None
    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0].isdigit():
            return {
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            }
    return None


def write_label(txt_path: Path, box: dict | None) -> None:
    """Write a YOLO label. box=None writes empty (negative sample)."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if box:
        txt_path.write_text(f"0 {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}\n")
    else:
        txt_path.write_text("")


def compute_iou(cx1, cy1, w1, h1, cx2, cy2, w2, h2) -> float:
    """Compute IoU between two YOLO-format bboxes (normalized cx,cy,w,h)."""
    x1 = max(cx1 - w1 / 2, cx2 - w2 / 2)
    y1 = max(cy1 - h1 / 2, cy2 - h2 / 2)
    x2 = min(cx1 + w1 / 2, cx2 + w2 / 2)
    y2 = min(cy1 + h1 / 2, cy2 + h2 / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def load_p2_cache(cache_path: Path) -> dict:
    """Load P2 inference cache. Returns {} if not found."""
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_p2_cache(cache_path: Path, data: dict) -> None:
    """Save P2 inference cache to JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2))


def run_p2_inference(model_path: str, frames: list[Path], cache_path: Path) -> dict:
    """Run P2 model on frames, cache results. Returns {stem: {cx,cy,w,h,conf}}."""
    cached = load_p2_cache(cache_path)
    if cached:
        return cached

    from ultralytics import YOLO

    model = YOLO(model_path)
    results = {}
    for i, f in enumerate(frames):
        r = model.predict(str(f), conf=0.25, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            best = r.boxes[0]
            results[f.stem] = {
                "cx": float(best.xywhn[0][0]),
                "cy": float(best.xywhn[0][1]),
                "w": float(best.xywhn[0][2]),
                "h": float(best.xywhn[0][3]),
                "conf": float(best.conf[0]),
            }
        if (i + 1) % 20 == 0:
            print(f"  P2 inference: {i + 1}/{len(frames)}")

    save_p2_cache(cache_path, results)
    return results


def load_review_state(state_path: Path) -> dict:
    """Load review result JSON. Returns empty structure if not found."""
    if not state_path.exists():
        return {"accept": [], "reject": [], "skip": []}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"accept": [], "reject": [], "skip": []}


def save_review_state(state_path: Path, state: dict) -> None:
    """Save review result JSON."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_label_utils.py -v`
预期：全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add tools/_label_utils.py tests/test_label_utils.py
git commit -m "feat: add _label_utils shared functions for label review tools"
```

---

### 任务 2：重写 `review_labels.py` — 参数解析 + 模式切换

**文件：**
- 重写：`tools/review_labels.py`

- [ ] **步骤 1：重写 review_labels.py 的参数和初始化部分**

将 `tools/review_labels.py` 全部内容替换为以下代码（这是完整的重写，后续任务在此基础上迭代添加 UI 功能）：

```python
"""Universal frame label review tool for YOLO format.

Accept/Reject auto-generated labels (GSAM, pseudo, etc.) with optional
P2 model overlay. Manual labeling uses yololabeler directly.

Usage:
    # Review auto-labels (backward compatible with old review_labels.py)
    streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames

    # Review with P2 model overlay
    streamlit run tools/review_labels.py -- \
        --frames-dir data/datasets/game1080/frames \
        --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt

    # Force rebuild P2 cache
    streamlit run tools/review_labels.py -- \\
        --frames-dir data/datasets/game1080/frames \\
        --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt \\
        --rebuild-cache
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import cv2
import streamlit as st

from _label_utils import (
    read_label,
    write_label,
    compute_iou,
    load_p2_cache,
    run_p2_inference,
    load_review_state,
    save_review_state,
)

st.set_page_config(page_title="Label Review", layout="wide")


def get_args():
    parser = argparse.ArgumentParser(description="Universal frame label reviewer")
    parser.add_argument("--frames-dir", default="data/datasets/game1080/frames")
    parser.add_argument("--labels-dir", default=None,
                        help="Source labels dir (auto-detected: labels_gsam, then labels)")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir for reviewed labels (default: <parent>/labels)")
    parser.add_argument("--model", default=None,
                        help="YOLO model for overlay comparison (optional)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for model overlay")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild P2 inference cache")
    return parser.parse_known_args()[0]


def auto_detect_dirs(frames_dir: Path, args):
    """Resolve labels_dir and output_dir based on frames_dir and args."""
    parent = frames_dir.parent

    if args.labels_dir:
        labels_dir = Path(args.labels_dir)
    else:
        candidates = [parent / "labels_gsam", parent / "labels"]
        labels_dir = next((d for d in candidates if d.exists()), parent / "labels")

    output_dir = Path(args.output_dir) if args.output_dir else parent / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    return labels_dir, output_dir


def load_frames(frames_dir: Path, labels_dir: Path) -> list[Path]:
    """Load frames that have source labels. Falls back to all frames if none labeled."""
    all_frames = sorted(frames_dir.glob("*.jpg"))
    if not all_frames:
        return []

    frames = []
    for f in all_frames:
        txt = labels_dir / f"{f.stem}.txt"
        if txt.exists():
            content = txt.read_text().strip()
            if content and content[0].isdigit():
                frames.append(f)
    return frames if frames else all_frames


def load_p2_data(model_path: str | None, frames: list[Path],
                 frames_dir: Path, rebuild: bool) -> dict:
    """Load or compute P2 inference results."""
    if not model_path:
        return {}

    cache_path = frames_dir.parent / "p2_cache.json"
    if rebuild and cache_path.exists():
        cache_path.unlink()

    return run_p2_inference(model_path, frames, cache_path)


def draw_bbox(img, box, color, label, h, w):
    """Draw a single YOLO bbox on image."""
    x1 = int((box["cx"] - box["w"] / 2) * w)
    y1 = int((box["cy"] - box["h"] / 2) * h)
    x2 = int((box["cx"] + box["w"] / 2) * w)
    y2 = int((box["cy"] + box["h"] / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def render_frame(frame_path: Path,
                 source_box: dict | None, p2_box: dict | None,
                 p2_conf: float | None, iou: float | None):
    """Load image, draw bboxes, return display image."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None, 0, 0
    h, w = img.shape[:2]
    if max(w, h) > 1200:
        scale = 1200 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        h, w = img.shape[:2]

    display = img.copy()

    if source_box:
        draw_bbox(display, source_box, (0, 255, 0), "SRC", h, w)

    if p2_box:
        label = f"P2 {p2_conf:.2f}"
        if iou is not None:
            label += f" IoU={iou:.2f}"
        draw_bbox(display, p2_box, (255, 100, 0), label, h, w)

    return display, h, w


def get_frame_info(stem, source_box, p2_data):
    """Compute P2 overlay info for a frame."""
    p2_entry = p2_data.get(stem)
    p2_box = None
    p2_conf = None
    iou = None
    if p2_entry:
        p2_box = {k: p2_entry[k] for k in ("cx", "cy", "w", "h")}
        p2_conf = p2_entry["conf"]
        if source_box:
            iou = compute_iou(
                source_box["cx"], source_box["cy"], source_box["w"], source_box["h"],
                p2_box["cx"], p2_box["cy"], p2_box["w"], p2_box["h"],
            )
    return p2_box, p2_conf, iou


def sort_frames(frames: list[Path], p2_data: dict, source_boxes: dict,
                sort_mode: str) -> list[Path]:
    """Sort frames by chosen mode."""
    if sort_mode == "filename":
        return sorted(frames)

    def score(f):
        stem = f.stem
        src = source_boxes.get(stem)
        p2 = p2_data.get(stem)
        if sort_mode == "iou_desc":
            if src and p2:
                return -compute_iou(
                    src["cx"], src["cy"], src["w"], src["h"],
                    p2["cx"], p2["cy"], p2["w"], p2["h"],
                )
            if p2:
                return -p2["conf"]
            return float('inf')
        if sort_mode == "bbox_size":
            if src:
                return -(src["w"] * src["h"])
            return float('inf')
        return float('inf')

    return sorted(frames, key=score)


def main():
    args = get_args()
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        st.error(f"Frames directory not found: {frames_dir}")
        return

    labels_dir, output_dir = auto_detect_dirs(frames_dir, args)

    frames = load_frames(frames_dir, labels_dir)
    if not frames:
        st.error("No frames found")
        return

    source_boxes = {}
    for f in frames:
        txt = labels_dir / f"{f.stem}.txt"
        source_boxes[f.stem] = read_label(txt)

    p2_data = {}
    if args.model:
        with st.spinner("Running P2 inference (cached after first run)..."):
            p2_data = load_p2_data(args.model, frames, frames_dir, args.rebuild_cache)

    # review_result.json 保存在 <frames_parent>/review_result.json
    # 和 frames/、labels/ 平级，方便查找
    state_path = output_dir.parent / "review_result.json"
    review_state = load_review_state(state_path)

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        sort_mode = st.selectbox("Sort by", [
            ("File name", "filename"),
            ("P2 IoU (high first)", "iou_desc"),
            ("BBox size (large first)", "bbox_size"),
        ], format_func=lambda x: x[0], index=0)[1]

        n_accept = len(review_state.get("accept", []))
        n_reject = len(review_state.get("reject", []))
        n_skip = len(review_state.get("skip", []))
        st.metric("Accept", n_accept)
        st.metric("Reject", n_reject)
        st.metric("Skip", n_skip)
        st.metric("Remaining", len(frames) - n_accept - n_reject - n_skip)

        if st.button("Export Results"):
            save_review_state(state_path, review_state)
            st.success(f"Saved to {state_path}")

    frames = sort_frames(frames, p2_data, source_boxes, sort_mode)
    total = len(frames)

    # Skip already-reviewed frames
    reviewed = set(review_state.get("accept", []) + review_state.get("reject", []) + review_state.get("skip", []))
    unreviewed = [f for f in frames if f.stem not in reviewed]
    if unreviewed:
        current_idx = frames.index(unreviewed[0])
    else:
        current_idx = total - 1

    if st.session_state.idx >= total:
        st.session_state.idx = current_idx

    idx = min(st.session_state.idx, total - 1)
    frame_path = frames[idx]
    stem = frame_path.stem
    source_box = source_boxes.get(stem)

    p2_box, p2_conf, iou = get_frame_info(stem, source_box, p2_data)

    display, _, _ = render_frame(frame_path, source_box, p2_box, p2_conf, iou)
    if display is None:
        st.error(f"Cannot load image: {frame_path}")
        return

    st.image(display, channels="BGR", use_container_width=True)

    # --- Buttons ---
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    with col1:
        if st.button("Accept", type="primary"):
            out_txt = output_dir / f"{stem}.txt"
            if p2_box and iou is not None and iou > 0.3:
                write_label(out_txt, p2_box)
            elif source_box:
                write_label(out_txt, source_box)
            review_state.setdefault("accept", []).append(stem)
            save_review_state(state_path, review_state)
            st.session_state.idx = idx + 1
            st.rerun()
    with col2:
        if st.button("Reject"):
            out_txt = output_dir / f"{stem}.txt"
            write_label(out_txt, None)
            review_state.setdefault("reject", []).append(stem)
            save_review_state(state_path, review_state)
            st.session_state.idx = idx + 1
            st.rerun()
    with col3:
        if st.button("Skip"):
            review_state.setdefault("skip", []).append(stem)
            save_review_state(state_path, review_state)
            st.session_state.idx = idx + 1
            st.rerun()
    with col4:
    # 已审核帧自动跳过，但仍可手动导航回顾
    with col5:
        if st.button("<< Prev"):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

    # --- Info bar ---
    src_status = "has label" if source_box else "no label"
    p2_status = f"P2 conf={p2_conf:.2f}" if p2_conf else "P2: no detection"
    iou_status = f"IoU={iou:.2f}" if iou is not None else ""
    st.caption(
        f"{idx + 1}/{total} | {stem}.jpg | SRC: {src_status} | {p2_status} {iou_status} | "
        f"src: {labels_dir.name} -> out: {output_dir.name}"
    )

    if idx >= total - 1:
        st.success(
            f"All frames reviewed! Accept: {n_accept}, Reject: {n_reject}, Skip: {n_skip}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：手动验证基础功能**

运行：`python3 -c "from tools._label_utils import read_label, write_label, compute_iou; print('import ok')"`
预期：`import ok`

- [ ] **步骤 3：Commit**

```bash
git add tools/review_labels.py
git commit -m "feat: rewrite review_labels.py with P2 overlay, progress tracking, undo, sort"
```

---

### 任务 3：预计算 P2 缓存

**文件：**
- 生成：`data/datasets/game1080/p2_cache.json`

这一步在 Streamlit 启动前预跑 P2 推理，避免首次加载时等太久。实际执行时由 `review_labels.py` 自动触发，但为了审核体验，建议提前跑。

- [ ] **步骤 1：用 Python 脚本预跑 P2**

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
from pathlib import Path
from _label_utils import run_p2_inference

frames_dir = Path('data/datasets/game1080/frames')
labels_gsam = frames_dir.parent / 'labels_gsam'
cache_path = frames_dir.parent / 'p2_cache.json'

labeled = []
for f in sorted(frames_dir.glob('*.jpg')):
    txt = labels_gsam / f'{f.stem}.txt'
    if txt.exists() and txt.read_text().strip().startswith('0'):
        labeled.append(f)

print(f'Running P2 on {len(labeled)} labeled frames...')
model = 'runs/detect/frisbee_det_p2_game_v2/weights/best.pt'
result = run_p2_inference(model, labeled, cache_path)
print(f'Cache saved: {cache_path} ({len(result)} detections)')
"
```

预期：生成 `data/datasets/game1080/p2_cache.json`，包含约 30-40 帧的 P2 检测结果

---

### 任务 4：更新 AGENTS.md 和 session handover

**文件：**
- 修改：`AGENTS.md`
- 修改：`docs/2026-05-20-session-handover.md`

- [ ] **步骤 1：更新 AGENTS.md 的 Quick Start 部分**

在 `AGENTS.md` 的 Quick Start 区块中，更新 Streamlit 审核工具的说明。找到现有的 `# Interactive detection review (Web — Streamlit)` 区块后面，添加：

```markdown
# Review auto-labels (GSAM, pseudo, etc.)
streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames
streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt
```
同时更新 Architecture 部分的工具描述，将 `label_frames.py` 删除，从零手标说明指向 yololabeler。

- [ ] **步骤 2：更新 session handover**

在 `docs/2026-05-20-session-handover.md` 的 `game1080 Labeling Status` 部分，更新审核工具说明：

```markdown
- **Review tool:** `tools/review_labels.py` (Streamlit) — universal label reviewer
  - Run: `streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt`
  - Supports: Accept/Reject/Skip/Prev, P2 overlay, sort by IoU, progress tracking
  - Output: `labels/` (never modifies `labels_gsam/`)
  - Result: `review_result.json`
```

- [ ] **步骤 3：Commit**

```bash
git add AGENTS.md docs/2026-05-20-session-handover.md
git commit -m "docs: update review_labels.py usage in AGENTS.md and session handover"
```

---

### 任务 5：删除 `label_frames.py`

**文件：**
- 删除：`tools/label_frames.py`

从零手标功能由 yololabeler 承担（`tools/review_desktop.py` 桥接），`review_labels.py` 不再包含 label 模式。`label_frames.py` 直接删除。

- [ ] **步骤 1：删除 label_frames.py**

```bash
git rm tools/label_frames.py
git commit -m "refactor: remove label_frames.py — use yololabeler for manual labeling"
```

---

### 任务 6：运行完整测试套件验证

- [ ] **步骤 1：运行所有测试**

```bash
python3 -m pytest tests/ -v
```

预期：全部 PASS（原有 37 个测试 + 新增 test_label_utils 测试）

- [ ] **步骤 2：Streamlit 烟测**

```bash
# 无模型模式（向后兼容）
streamlit run tools/review_labels.py -- --frames-dir data/datasets/game1080/frames &
sleep 5
curl -s http://localhost:8501 | head -5
kill %1

# 有模型模式
streamlit run tools/review_labels.py -- \
  --frames-dir data/datasets/game1080/frames \
  --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt &
sleep 5
curl -s http://localhost:8501 | head -5
kill %1
```

预期：两次都能启动成功，无报错

---

### 任务 7：用户执行 GSAM 标签审核

这一步由用户在浏览器中手动完成，AI 不执行。

- [ ] **步骤 1：启动审核工具**

```bash
streamlit run tools/review_labels.py -- \
  --frames-dir data/datasets/game1080/frames \
  --model runs/detect/frisbee_det_p2_game_v2/weights/best.pt
```

- [ ] **步骤 2：在浏览器中审核 113 帧**

操作：
- 看到 GSAM bbox（绿框）和 P2 bbox（蓝框）对照
- 飞盘标签正确 → 点击 Accept
- GSAM 标签是误检 → 点击 Reject
- 不确定 → 点击 Skip
- 点错了 → 点击 Prev 回退，但需手动修改 review_result.json

建议审核策略：
- Sidebar 选择 "Sort by: P2 IoU (high first)" — P2+GSAM 一致的先看，快速过
- 巨型 bbox（>25% 图片）直接 Reject
- P2 没检出但 GSAM 标了的帧，看图判断

- [ ] **步骤 3：完成后 Export Results**

点击 Sidebar 的 "Export Results" 按钮，结果保存到 `data/datasets/game1080/review_result.json`。

---

### 任务 8：合并审核结果到训练集

**文件：**
- 读取：`data/datasets/game1080/review_result.json`
- 读取：`data/datasets/game1080/labels/`（审核输出）
- 复制到：`data/datasets/frisbee_merged/images/train_game/` + `labels/train_game/`

- [ ] **步骤 1：验证审核结果**

```bash
python3 -c "
import json
from pathlib import Path

result = json.loads(Path('data/datasets/game1080/review_result.json').read_text())
print(f'Accept: {len(result[\"accept\"])}')
print(f'Reject: {len(result[\"reject\"])}')
print(f'Skip: {len(result[\"skip\"])}')
"
```

- [ ] **步骤 2：复制 accepted 帧和标签到训练集**

```bash
python3 -c "
import json, shutil
from pathlib import Path

result = json.loads(Path('data/datasets/game1080/review_result.json').read_text())
accepted = result['accept']

img_dst = Path('data/datasets/frisbee_merged/images/train_game')
lbl_dst = Path('data/datasets/frisbee_merged/labels/train_game')
img_dst.mkdir(parents=True, exist_ok=True)
lbl_dst.mkdir(parents=True, exist_ok=True)

frames_dir = Path('data/datasets/game1080/frames')
labels_dir = Path('data/datasets/game1080/labels')

copied = 0
for stem in accepted:
    src_img = frames_dir / f'{stem}.jpg'
    src_lbl = labels_dir / f'{stem}.txt'
    if src_img.exists() and src_lbl.exists():
        shutil.copy2(src_img, img_dst / src_img.name)
        shutil.copy2(src_lbl, lbl_dst / src_lbl.name)
        copied += 1

print(f'Copied {copied} accepted frames to training set')
"
```

- [ ] **步骤 3：Commit**

```bash
git add tools/review_labels.py tools/_label_utils.py tests/test_label_utils.py
git commit -m "feat: universal label review tool with P2 overlay, merge label_frames"
```

---

## 不在此计划中的范围

- `review_web.py`（TP/FP 检测框审核）— 不同粒度，独立保留
- `review_desktop.py`（yololabeler 桥接）— 不同工具链，独立保留
- `auto_label_gsam.py`（GSAM 自动标注）— 上游工具，不改动
- `auto_label.py`（YOLO 伪标注）— 上游工具，不改动
- `analyze_labels.py`（标签统计）— 分析工具，不改动
- 点击画框功能（label_frames.py 的 components.html 交互）— 由 yololabeler 替代
- Streamlit 键盘快捷键 — 可作为后续增强
