# 单飞盘追踪器 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 替换当前 ByteTrack 多目标追踪为单目标 Kalman 滤波 + 候选评分，每帧只追踪一个飞盘，画出滑动窗口轨迹线。

**架构：** `utils/tracker_utils.py` 提供 Kalman 初始化 + 候选评分 + Trajectory 管理。`inference/predict_track.py` 的 main loop 从 `model.track()` 改为逐帧 `model()` + Kalman 更新 + OpenCV 轨迹绘制。

**技术栈：** Python 3, OpenCV (cv2.KalmanFilter), NumPy, ultralytics (YOLO), pytest

**规格文档：** `docs/superpowers/specs/2026-05-18-single-frisbee-tracker-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `utils/tracker_utils.py` | 新建 | `init_kalman()`, `score_candidates()`, `Trajectory` 类 |
| `tests/test_tracker_utils.py` | 新建 | 7 个测试覆盖 tracker_utils |
| `inference/predict_track.py` | 修改 | 主循环从 ByteTrack 改为单目标 Kalman + 轨迹绘制 |

---

### 任务 1：utils/tracker_utils.py — Kalman + 评分器 + Trajectory

**文件：**
- 创建：`utils/tracker_utils.py`
- 创建：`tests/test_tracker_utils.py`

- [ ] **步骤 1：编写失败测试 — Kalman 初始化状态为 4 维**

创建 `tests/test_tracker_utils.py`：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_tracker_utils.py::test_kalman_init_state -v`
预期：FAIL — `ImportError: cannot import name 'init_kalman'`

- [ ] **步骤 3：实现 init_kalman**

创建 `utils/tracker_utils.py`：

```python
"""Single-frisbee tracker: Kalman filter + candidate scoring + trajectory.

Functions:
    init_kalman          — create and configure cv2.KalmanFilter(4, 2)
    score_candidates     — pick the best candidate box per frame
    Trajectory           — ring buffer of tracked positions + areas
"""

import cv2
import numpy as np


MAX_EXPECTED_DISPLACEMENT = 50.0  # px, at 25fps


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
    return kf


def score_candidates(
    candidates: list[dict],
    trajectory: Trajectory | None,
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

            aspect_ratio = max(0.0, min(1.0, bw / max(bh, 1.0) / 2.0))
            score = (0.35 * motion_score + 0.25 * conf
                     + 0.25 * area_consistent_score + 0.15 * aspect_ratio)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


class Trajectory:
    """Ring buffer of tracked positions and areas."""

    def __init__(self, maxlen: int = 2000):
        self._pts: list[tuple[float, float]] = []
        self.areas: list[float] = []
        self._maxlen = maxlen

    def push(self, px: float, py: float, area: float) -> None:
        self._pts.append((px, py))
        self.areas.append(area)
        if len(self._pts) > self._maxlen:
            self._pts.pop(0)
            self.areas.pop(0)

    def get_window(self, n: int = 50) -> list[tuple[float, float]]:
        return self._pts[-n:]

    def last_position(self) -> tuple[float, float] | None:
        return self._pts[-1] if self._pts else None
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python3 -m pytest tests/test_tracker_utils.py::test_kalman_init_state -v`
预期：PASS

- [ ] **步骤 5：编写 Kalman predict/update 测试**

追加到 `tests/test_tracker_utils.py`：

```python
def test_kalman_predict_update():
    kf = init_kalman()
    # Feed same observation for 3 iterations — output should converge toward observation
    obs = np.array([[320.0], [240.0]], dtype=np.float32)
    for _ in range(3):
        kf.predict()
        kf.correct(obs)
    state = kf.statePost
    assert abs(float(state[0]) - 320.0) < 10.0, f"px drifted: {float(state[0])}"
    assert abs(float(state[1]) - 240.0) < 10.0, f"py drifted: {float(state[1])}"
```

- [ ] **步骤 6：运行测试**

运行：`python3 -m pytest tests/test_tracker_utils.py::test_kalman_predict_update -v`
预期：PASS (2 passed)

- [ ] **步骤 7：编写 Trajectory 测试**

追加到 `tests/test_tracker_utils.py`：

```python
def test_trajectory_push_window():
    traj = Trajectory(maxlen=100)
    for i in range(60):
        traj.push(float(i), float(i), 200.0)
    window = traj.get_window(50)
    assert len(window) == 50
    assert window[0] == (10.0, 10.0)  # first of the last 50
    assert window[-1] == (59.0, 59.0)  # last pushed


def test_trajectory_ring_behavior():
    traj = Trajectory(maxlen=10)
    for i in range(20):
        traj.push(float(i), float(i), 200.0)
    assert len(traj.get_window(50)) == 10  # capped by maxlen
    assert traj.last_position() == (19.0, 19.0)
    assert traj.areas == [200.0] * 10


def test_trajectory_empty():
    traj = Trajectory()
    assert traj.last_position() is None
    assert traj.get_window() == []
```

- [ ] **步骤 8：运行测试**

运行：`python3 -m pytest tests/test_tracker_utils.py::test_trajectory_push_window tests/test_tracker_utils.py::test_trajectory_ring_behavior tests/test_tracker_utils.py::test_trajectory_empty -v`
预期：3 PASS

- [ ] **步骤 9：编写 score_candidates 测试**

追加到 `tests/test_tracker_utils.py`：

```python
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
        {"box": [105, 105, 125, 125], "conf": 0.5},  # close to last pos
        {"box": [500, 500, 520, 520], "conf": 0.9},  # far but high conf
    ]
    best = score_candidates(cands, traj, traj.last_position())
    assert best == 0, "should pick motion-consistent even with lower conf"


def test_aspect_score_favors_wide_boxes():
    """Aspect ratio 15% should help wide boxes (frisbee) over tall boxes (person)."""
    traj = Trajectory()
    traj.push(100, 100, 200)
    tall_box = {"box": [100, 100, 110, 140], "conf": 0.5}    # w=10 h=40 ar=0.25
    wide_box = {"box": [100, 100, 120, 120], "conf": 0.5}    # w=20 h=20 ar=1.0
    cands = [tall_box, wide_box]
    best = score_candidates(cands, traj, (105, 105))
    assert best == 1, "wide box should win"
```

- [ ] **步骤 10：运行全部测试**

运行：`python3 -m pytest tests/test_tracker_utils.py -v`
预期：7/7 PASS

- [ ] **步骤 11：Commit**

```bash
git add utils/tracker_utils.py tests/test_tracker_utils.py
git commit -m "feat: add Kalman filter, candidate scorer, Trajectory class"
```

---

### 任务 2：重写 predict_track.py 主循环

**文件：**
- 修改：`inference/predict_track.py`（替换 main 循环 + 更新 CSV 导出 + 添加轨迹绘制）

这个任务是全部替换 predict_track.py 的核心逻辑。完整新文件内容：

- [ ] **步骤 1：重写 predict_track.py**

```python
"""Single-frisbee tracker with trajectory visualization.

Runs YOLO per-frame, picks the most-likely frisbee candidate via Kalman
filter + weighted scoring, draws a sliding-window trajectory line.

Usage:
    python3 inference/predict_track.py --video movie/test.mp4
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from configs.models import V3_MODEL, DEFAULT_CONF
from utils.homography import load_calibration, pixel_to_world
from utils.tracker_utils import init_kalman, score_candidates, Trajectory

LOST_TRACK_THRESHOLD = 15  # frames


def find_calibration(video_path: Path) -> dict | None:
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


def get_args():
    parser = argparse.ArgumentParser(description="Single-frisbee tracker with trajectory")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", type=str, default=str(V3_MODEL))
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-visualize", action="store_true", default=False)
    return parser.parse_args()


def draw_trajectory(frame: np.ndarray, trajectory: Trajectory, window: int = 50) -> None:
    pts = trajectory.get_window(window)
    if len(pts) < 2:
        return
    n = len(pts)
    for i in range(n - 1):
        p0 = (int(round(pts[i][0])), int(round(pts[i][1])))
        p1 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
        t = i / max(n - 1, 1)
        # BGR: old=orange(0,127,255), new=red(0,0,255)
        g = int(255 * (1.0 - t))
        b = int(127 * (1.0 - t))
        cv2.line(frame, p0, p1, (b, g, 255), 2, cv2.LINE_AA)
    last = (int(round(pts[-1][0])), int(round(pts[-1][1])))
    cv2.circle(frame, last, 5, (0, 255, 0), -1)  # green dot on current position


def main():
    args = get_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print(f"Video:  {video_path.name}")
    print(f"Model:  {args.model}")
    print(f"Conf:   {args.conf}")

    matrix = None
    if args.calibration:
        matrix = load_calibration(Path(args.calibration))["matrix"]
        print(f"Calib:  {args.calibration}")
    else:
        calib = find_calibration(video_path)
        if calib:
            matrix = calib["matrix"]
            print("Calib:  auto-detected")
        else:
            print("Calib:  NONE — pixel-only")

    output_dir = Path(args.output_dir or f"runs/track/{video_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    model = YOLO(args.model)
    kf = init_kalman()
    trajectory = Trajectory()
    status = "searching"  # searching | predicting | tracking
    lost_counter = 0
    print(f"\nTracking (single-frisbee)...")

    all_rows: list[dict] = []
    out_video = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not args.no_visualize:
        out_video = cv2.VideoWriter(
            str(output_dir / f"{video_path.stem}_tracked.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
        )

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=args.conf, verbose=False)
        r = results[0]
        candidates = []
        if r.boxes is not None:
            for box, conf_val in zip(r.boxes.xyxy, r.boxes.conf):
                candidates.append({
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "conf": float(conf_val),
                })

        current_status = status
        if not candidates:
            # No detections this frame
            lost_counter += 1
            if lost_counter > LOST_TRACK_THRESHOLD:
                status = "searching"
                kf = init_kalman()
            else:
                status = "predicting"
                prediction = kf.predict()
                px = float(prediction[0])
                py = float(prediction[1])
                if 0 <= px <= w and 0 <= py <= h:
                    trajectory.push(px, py, 0)
                current_status = "predicting"
        else:
            prediction = (float(kf.statePost[0]), float(kf.statePost[1])) if lost_counter > 0 else None
            prediction_arg = kf.predict() if status == "tracking" else None
            pred_pt = (float(prediction_arg[0]), float(prediction_arg[1])) if prediction_arg is not None else None

            best_idx = score_candidates(candidates, trajectory if status == "tracking" else None, pred_pt)
            if best_idx >= 0:
                best = candidates[best_idx]
                bx = best["box"]
                cx = (bx[0] + bx[2]) / 2.0
                cy = (bx[1] + bx[3]) / 2.0
                bw = bx[2] - bx[0]
                bh = bx[3] - bx[1]
                area = bw * bh
                meas = np.array([[cx], [cy]], dtype=np.float32)
                kf.correct(meas)
                status = "tracking"
                lost_counter = 0
                trajectory.push(cx, cy, area)
                current_status = "tracking"

                vx = float(kf.statePost[2])
                vy = float(kf.statePost[3])
                row = {
                    "frame": frame_idx, "px": round(cx, 1), "py": round(cy, 1),
                    "vx": round(vx, 2), "vy": round(vy, 2),
                    "conf": round(float(best["conf"]), 4), "status": status,
                }
                if matrix is not None:
                    try:
                        wx, wy = pixel_to_world(matrix, cx, cy)
                        row["wx"] = round(wx, 2)
                        row["wy"] = round(wy, 2)
                    except Exception:
                        row["wx"] = None
                        row["wy"] = None
                all_rows.append(row)

                if out_video is not None:
                    x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Frisbee", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if out_video is not None:
            draw_trajectory(frame, trajectory)
            info_line = f"Frame: {frame_idx}  Status: {current_status}"
            cv2.putText(frame, info_line, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            det_info = f"Dets: {len(candidates)}"
            cv2.putText(frame, det_info, (w - 150, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            out_video.write(frame)

        if frame_idx % 100 == 0:
            print(f"  {frame_idx} frames, {len(all_rows)} tracking rows")

    cap.release()
    if out_video is not None:
        out_video.release()

    if all_rows:
        csv_path = output_dir / f"{video_path.stem}_tracks.csv"
        fieldnames = ["frame", "px", "py", "vx", "vy", "conf", "status", "wx", "wy"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDone: {frame_idx} frames, {len(all_rows)} tracking rows")
        print(f"CSV:   {csv_path}")
    else:
        print("\nDone: 0 frames tracked")

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
git commit -m "feat: replace ByteTrack with single-frisbee Kalman tracker + trajectory viz"
```

---

### 任务 3：端到端验证

**文件：** 无修改

- [ ] **步骤 1：在 55-56min 视频上运行（无标定，像素-only）**

```bash
python3 inference/predict_track.py \
  --video movie/25866279684-1-192_55-56min.mp4 \
  --no-visualize
```

预期：
- 输出 `runs/track/` 目录
- CSV 每帧最多 1 行，字段 `frame, px, py, vx, vy, conf, status, wx, wy`
- status 在 searching/tracking/predicting 之间切换
- 控制台打印进度

- [ ] **步骤 2：检查 CSV 质量**

```bash
python3 -c "
import csv
with open('runs/track/25866279684-1-192_55-56min/25866279684-1-192_55-56min_tracks.csv') as f:
    rows = list(csv.DictReader(f))
statuses = set(r['status'] for r in rows)
print(f'Tracking rows: {len(rows)}')
print(f'Statuses seen: {statuses}')
tracking = sum(1 for r in rows if r['status'] == 'tracking')
searching = sum(1 for r in rows if r['status'] == 'searching')
print(f'  tracking: {tracking}')
print(f'  searching: {searching}')
print(f'  predicting: {len(rows) - tracking - searching}')
# Should have no track_id field
assert 'track_id' not in rows[0], 'track_id should be removed'
print('CSV format OK')
"
```

预期：tracking > 0, track_id 不存在

- [ ] **步骤 3：在 20-23min 视频上运行**

```bash
python3 inference/predict_track.py \
  --video movie/clip_20-23min.mp4 \
  --no-visualize
```

- [ ] **步骤 4：用 GLM-4V 审视频**

```bash
# Extract a frame from the output video and send to GLM-4V
ffmpeg -y -i runs/track/clip_20-23min/clip_20-23min_tracked.mp4 \
  -vf "select=eq(n\,500)" -vsync vfr /tmp/track_eval.jpg 2>/dev/null
python3 -c "
import base64, sys
from openai import OpenAI
client = OpenAI(api_key='04b4a3613cb9da29764862eb6c1067e5.KXDv17RG8zCeshyv', base_url='https://open.bigmodel.cn/api/paas/v4')
with open('/tmp/track_eval.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
r = client.chat.completions.create(model='glm-4v-flash', messages=[{
    'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
        {'type': 'text', 'text': 'This frame shows a single-frisbee tracker. Green box=frisbee candidate, colored line=trajectory. Is the box on the frisbee or on something else? Be honest.'}
    ]
}], temperature=0.1, max_tokens=100)
print(r.choices[0].message.content)
"
```

---

### 任务 4：全部测试回归

**文件：** 无修改

- [ ] **步骤 1：运行完整测试套件**

```bash
python3 -m pytest tests/ -v
```

预期：全部 PASS（8 homography + 7 tracker_utils + 4 tracking + 14 existing = ~33）

---

### 任务 5：清理中间产物

**文件：** 无修改

- [ ] **步骤 1：清理临时文件**

```bash
rm -rf runs/track/ /tmp/track_eval.jpg
```
