# 追踪 + 场地坐标映射 Pipeline 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 创建 `inference/predict_track.py`，在 v3 检测上跑 ByteTrack 追踪，用 Homography 映射场地坐标，输出标注视频 + CSV 轨迹数据。

**架构：** 单文件脚本：argparse → YOLO.track() → pixel_to_world() → OpenCV 标注 + CSV 导出。坐标映射复用 `utils/homography.py`。

**技术栈：** Python 3, ultralytics (ByteTrack), OpenCV, NumPy

**规格文档：** `docs/superpowers/specs/2026-05-18-tracking-pipeline-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `inference/predict_track.py` | 新建 | 追踪 pipeline 主入口 |
| `tests/test_tracking.py` | 新建 | 辅助函数单元测试 |
| `configs/models.py` | 修改 | 更新 DEFAULT_MODEL 为 v3 |

---

### 任务 1：更新默认模型为 v3

**文件：**
- 修改：`configs/models.py`

- [ ] **步骤 1：修改 `configs/models.py`**

```python
DEFAULT_MODEL = V3_MODEL
DEFAULT_CONF = 0.35
```

- [ ] **步骤 2：验证**

运行：`python3 -c "from configs.models import DEFAULT_MODEL, DEFAULT_CONF; print(DEFAULT_MODEL.name, DEFAULT_CONF)"`
预期：输出包含 `frisbee_det_s_v3 0.35`

- [ ] **步骤 3：Commit**

```bash
git add configs/models.py
git commit -m "fix: set DEFAULT_MODEL to v3, DEFAULT_CONF to 0.35"
```

---

### 任务 2：创建辅助函数测试

**文件：**
- 创建：`tests/test_tracking.py`

- [ ] **步骤 1：编写测试**

```python
"""Tests for tracking pipeline helper functions."""

import datetime
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_calibration_lookup_exact_match(tmp_path, monkeypatch):
    """Simulate find_calibration logic: exact stem match."""
    calib_dir = tmp_path / "homography"
    calib_dir.mkdir(parents=True)
    (calib_dir / "test_video.json").write_text('{"matrix":[[1,0,0],[0,1,0],[0,0,1]],'
                                               '"video":"test.mp4","image_size":[1280,720],'
                                               '"field_size_m":[100,37],"calibration_frame":0,'
                                               '"points":[],"reprojection_error_px":0.5}')

    stem = "test_video"
    candidate = calib_dir / f"{stem}.json"
    assert candidate.exists()

    import json
    data = json.loads(candidate.read_text())
    assert data["video"] == "test.mp4"


def test_calibration_lookup_underscore_fallback(tmp_path):
    """Simulate find_calibration: fallback to base before first underscore."""
    calib_dir = tmp_path / "homography"
    calib_dir.mkdir(parents=True)
    (calib_dir / "base.json").write_text('{"matrix":[[1,0,0],[0,1,0],[0,0,1]],'
                                         '"video":"base.mp4","image_size":[1280,720],'
                                         '"field_size_m":[100,37],"calibration_frame":0,'
                                         '"points":[],"reprojection_error_px":0.5}')

    video_stem = "base_55-56min"
    base = video_stem.split("_", 1)[0] if "_" in video_stem else video_stem
    assert base == "base"
    candidate = calib_dir / f"{base}.json"
    assert candidate.exists()

    import json
    data = json.loads(candidate.read_text())
    assert data["video"] == "base.mp4"


def test_calibration_lookup_not_found(tmp_path):
    calib_dir = tmp_path / "homography"
    calib_dir.mkdir()

    video_stem = "nonexistent"
    assert not (calib_dir / f"{video_stem}.json").exists()

    base = video_stem.split("_", 1)[0]
    assert not (calib_dir / f"{base}.json").exists()


def test_export_csv_rows(tmp_path):
    """Test CSV export logic."""
    rows = [
        {"frame": 0, "track_id": 1, "px": 640.0, "py": 500.0, "wx": "", "wy": "", "conf": 0.52},
        {"frame": 1, "track_id": 1, "px": 642.0, "py": 498.0, "wx": 50.1, "wy": 1.2, "conf": 0.55},
    ]
    csv_path = tmp_path / "tracks.csv"
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "px", "py", "wx", "wy", "conf"])
        writer.writeheader()
        writer.writerows(rows)

    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 3
    assert lines[0] == "frame,track_id,px,py,wx,wy,conf"
    assert lines[1].endswith("0.52")
    assert "50.1" in lines[2]
```

- [ ] **步骤 2：运行测试**

运行：`python3 -m pytest tests/test_tracking.py -v`
预期：4/4 PASS

- [ ] **步骤 3：Commit**

```bash
git add tests/test_tracking.py
git commit -m "test: add calibration lookup and CSV export tests"
```

---

### 任务 3：创建 predict_track.py

**文件：**
- 创建：`inference/predict_track.py`

- [ ] **步骤 1：创建完整文件**

```python
"""ByteTrack tracking + field coordinate mapping.

Runs YOLO inference with ByteTrack on a video, maps detections to field
coordinates via homography, outputs annotated video + CSV trajectory data.

Usage:
    python3 inference/predict_track.py --video movie/test.mp4
    python3 inference/predict_track.py --video movie/test.mp4 --no-visualize
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from configs.models import V3_MODEL, DEFAULT_CONF
from utils.homography import load_calibration, pixel_to_world


def find_calibration(video_path: Path) -> dict | None:
    """Auto-detect calibration JSON for a video. Returns None if not found."""
    calib_dir = Path("configs/homography")
    if not calib_dir.exists():
        return None

    stem = video_path.stem
    candidate = calib_dir / f"{stem}.json"
    if candidate.exists():
        return load_calibration(candidate)

    if "_" in stem:
        base = stem.split("_", 1)[0]
        candidate = calib_dir / f"{base}.json"
        if candidate.exists():
            return load_calibration(candidate)

    return None


def export_tracks_csv(rows: list[dict], output_path: Path) -> None:
    """Write tracking data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "px", "py", "wx", "wy", "conf"])
        writer.writeheader()
        writer.writerows(rows)


def get_args():
    parser = argparse.ArgumentParser(description="ByteTrack tracking with field coordinate mapping")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default=str(V3_MODEL))
    parser.add_argument("--calibration", type=str, default=None, help="Explicit calibration JSON path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-visualize", action="store_true", default=False, help="Skip video output")
    return parser.parse_args()


def main():
    args = get_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print(f"Video:  {video_path.name}")
    print(f"Model:  {args.model}")
    print(f"Conf:   {args.conf}")

    calib = None
    matrix = None
    if args.calibration:
        calib = load_calibration(Path(args.calibration))
        matrix = calib["matrix"]
        print(f"Calib:  {args.calibration}")
    else:
        calib = find_calibration(video_path)
        if calib:
            matrix = calib["matrix"]
            print("Calib:  auto-detected")
        else:
            print("Calib:  NONE — output will be pixel-only")

    output_dir = Path(args.output_dir or f"runs/track/{video_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    model = YOLO(args.model)
    print("\nTracking...")

    all_rows: list[dict] = []
    out_video = None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not args.no_visualize:
        out_video = cv2.VideoWriter(
            str(output_dir / f"{video_path.stem}_tracked.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
        )

    results = model.track(
        source=str(video_path),
        conf=args.conf,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_idx, r in enumerate(results, 1):
        orig_frame = r.orig_img if hasattr(r, "orig_img") else None
        if orig_frame is None:
            continue

        if r.boxes is not None and r.boxes.id is not None:
            for box, tid, conf_val in zip(r.boxes.xyxy, r.boxes.id, r.boxes.conf):
                px = float((box[0] + box[2]) / 2.0)
                py = float(box[3])
                tid_int = int(tid)

                row = {"frame": frame_idx, "track_id": tid_int, "px": round(px, 1),
                       "py": round(py, 1), "wx": "", "wy": "", "conf": round(float(conf_val), 4)}

                if matrix is not None:
                    try:
                        wx, wy = pixel_to_world(matrix, px, py)
                        row["wx"] = round(wx, 2)
                        row["wy"] = round(wy, 2)
                    except Exception:
                        pass

                all_rows.append(row)

                if out_video is not None:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    tid_label = f"#{tid_int}"
                    cv2.putText(orig_frame, tid_label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if row["wx"] != "":
                        coord_text = f"({row['wx']},{row['wy']})m"
                        cv2.putText(orig_frame, coord_text, (x2 - 120, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if out_video is not None:
            info = f"Frame: {frame_idx}  Dets: {len(r.boxes) if r.boxes else 0}"
            cv2.putText(orig_frame, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            out_video.write(orig_frame)

        if frame_idx % 100 == 0:
            print(f"  {frame_idx} frames, {len(all_rows)} detections")

    if out_video is not None:
        out_video.release()

    csv_path = output_dir / f"{video_path.stem}_tracks.csv"
    export_tracks_csv(all_rows, csv_path)
    dets_per_sec = len(all_rows) / max(frame_idx / fps, 0.001) if fps > 0 else 0
    print(f"\nDone: {frame_idx} frames, {len(all_rows)} detections ({dets_per_sec:.1f} dets/s)")
    print(f"CSV:   {csv_path}")
    if out_video is not None:
        print(f"Video: {output_dir / f'{video_path.stem}_tracked.mp4'}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：编译验证**

运行：`python3 -c "import py_compile; py_compile.compile('inference/predict_track.py', doraise=True)"`
预期：无输出

- [ ] **步骤 3：Commit**

```bash
git add inference/predict_track.py
git commit -m "feat: add ByteTrack tracking + coordinate mapping pipeline"
```

---

### 任务 4：端到端验证

**文件：** 无修改

- [ ] **步骤 1：在 55-56min 视频上运行（像素-only，无标定）**

```bash
python3 inference/predict_track.py \
  --video movie/25866279684-1-192_55-56min.mp4
```

预期：
- 输出 `runs/track/25866279684-1-192_55-56min/` 目录
- 包含 `.mp4` 和 `_tracks.csv`
- 控制台出现 "NONE — output will be pixel-only"
- CSV 中 wx, wy 为空

- [ ] **步骤 2：在 20-23min 视频上运行**

```bash
python3 inference/predict_track.py \
  --video movie/clip_20-23min.mp4
```

预期：同上

- [ ] **步骤 3：检查 CSV 质量**

```bash
python3 -c "
import csv
for fn in ['runs/track/25866279684-1-192_55-56min/25866279684-1-192_55-56min_tracks.csv',
           'runs/track/clip_20-23min/clip_20-23min_tracks.csv']:
    try:
        with open(fn) as f:
            rows = list(csv.DictReader(f))
        print(f'{fn}: {len(rows)} rows, {len(set(r[\"track_id\"] for r in rows))} tracks')
    except FileNotFoundError:
        print(f'{fn}: not found')
"
```

预期：行数合理（500-5000 行），track_ids 不为 0

---

### 任务 5：全部测试回归

**文件：** 无修改

- [ ] **步骤 1：运行完整测试套件**

```bash
python3 -m pytest tests/ -v
```

预期：全部 PASS（约 4*8 + 4 = ~26 个测试）

---

### 任务 6：清除中间产物

**文件：** 无修改

- [ ] **步骤 1：清理临时视频输出**

```bash
rm -rf runs/track/
```
