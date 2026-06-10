# 质量门控 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修改评分函数奖励运动候选 + 每 5 帧检查轨迹静止则自动重置，消除白帽子/人头 FP 的"黏上"行为。

**架构：** `utils/tracker_utils.py` 的 `score_candidates` 追加 speed_score 信号 + 调整权重。`inference/predict_track.py` 追加 5 帧惰性检查逻辑。

**技术栈：** Python 3, NumPy, pytest

**规格文档：** `docs/superpowers/specs/2026-05-18-quality-gate-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `utils/tracker_utils.py` | 修改 | 评分权重调整 + speed_score |
| `inference/predict_track.py` | 修改 | 5 帧惰性检查 + 重置逻辑 |
| `tests/test_tracker_utils.py` | 修改 | 追加 speed/惰性测试 |

---

### 任务 1：修改 score_candidates — speed_score + 权重调整

**文件：**
- 修改：`utils/tracker_utils.py`
- 修改：`tests/test_tracker_utils.py`（追加测试）

- [ ] **步骤 1：编写失败测试**

追加到 `tests/test_tracker_utils.py`：

```python
def test_speed_score_favors_moving_boxes():
    traj = Trajectory()
    traj.push(100, 100, 200)
    traj.push(102, 102, 200)
    traj.push(110, 110, 200)
    static = {"box": [100, 100, 120, 120], "conf": 0.5}
    moving = {"box": [110, 110, 130, 130], "conf": 0.5}
    cands = [static, moving]
    best = score_candidates(cands, traj, traj.last_position())
    assert best == 1, "moving candidate should score higher than static"


def test_stationary_loses_to_moving():
    traj = Trajectory()
    traj.push(100, 100, 200)
    traj.push(101, 101, 200)
    traj.push(102, 102, 200)
    stationary = {"box": [100, 100, 120, 120], "conf": 0.5}
    moving = {"box": [200, 200, 220, 220], "conf": 0.35}
    cands = [stationary, moving]
    best = score_candidates(cands, traj, (105, 105))
    assert best == 1, "moving candidate should beat stationary despite lower conf"
```

- [ ] **步骤 2：运行验证失败**

运行：`python3 -m pytest tests/test_tracker_utils.py::test_speed_score_favors_moving_boxes -v`
预期：FAIL — 静止候选仍获胜

- [ ] **步骤 3：修改 score_candidates**

在 `utils/tracker_utils.py` 的 `MAX_EXPECTED_DISPLACEMENT` 下面添加：

```python
MIN_DISPLACEMENT = 5.0  # px/frame, threshold for "moving"
```

修改 `score_candidates` 的有历史评分分支（替换现有 else 块内容）：

```python
        else:
            dist = np.sqrt((cx - prediction[0]) ** 2 + (cy - prediction[1]) ** 2)
            motion_score = max(0.0, min(1.0, 1.0 - dist / MAX_EXPECTED_DISPLACEMENT))

            if trajectory.areas:
                prev_area = float(trajectory.areas[-1])
                area_consistent_score = max(0.0, 1.0 - abs(np.log2(area / max(prev_area, 1.0))))
            else:
                area_consistent_score = 1.0

            aspect_score = max(0.0, 1.0 - abs(np.log2(bw / max(bh, 1.0))))

            pts_list = list(trajectory._pts)
            if len(pts_list) >= 3:
                recent = pts_list[-3:]
                dx = recent[-1][0] - recent[0][0]
                dy = recent[-1][1] - recent[0][1]
                avg_speed = np.sqrt(dx**2 + dy**2) / max(len(recent) - 1, 1)
                speed_score = min(1.0, avg_speed / MIN_DISPLACEMENT)
            else:
                speed_score = 0.5

            score = (0.30 * motion_score + 0.20 * conf
                     + 0.20 * area_consistent_score + 0.15 * aspect_score
                     + 0.15 * speed_score)
```

- [ ] **步骤 4：运行全部测试**

运行：`python3 -m pytest tests/test_tracker_utils.py -v`
预期：11/11 PASS

- [ ] **步骤 5：Commit**

```bash
git add utils/tracker_utils.py tests/test_tracker_utils.py
git commit -m "feat: add speed_score to candidate scoring, rebalance weights"
```

---

### 任务 2：添加 5 帧惰性检查

**文件：**
- 修改：`inference/predict_track.py`

- [ ] **步骤 1：追加惰性检查和常量**

在 `inference/predict_track.py` 中：

```python
LOST_TRACK_THRESHOLD = 15
STATIONARY_CHECK_INTERVAL = 5
STATIONARY_MAX_DISPLACEMENT = 8.0
```

在 main 循环中，`current_status = status` 之后、`if not candidates` 之前插入：

```python
        current_status = status

        if status == "tracking" and frame_idx % STATIONARY_CHECK_INTERVAL == 0:
            recent = trajectory.get_window(STATIONARY_CHECK_INTERVAL)
            if len(recent) >= STATIONARY_CHECK_INTERVAL:
                max_d = max(
                    np.sqrt((recent[i][0] - recent[0][0])**2 + (recent[i][1] - recent[0][1])**2)
                    for i in range(1, STATIONARY_CHECK_INTERVAL)
                )
                if max_d < STATIONARY_MAX_DISPLACEMENT:
                    kf = init_kalman()
                    trajectory = Trajectory()
                    status = "searching"
                    lost_counter = 0
                    current_status = "searching"
                    continue
```

- [ ] **步骤 2：验证编译**

运行：`python3 -c "import py_compile; py_compile.compile('inference/predict_track.py', doraise=True)"`
预期：无输出

- [ ] **步骤 3：Commit**

```bash
git add inference/predict_track.py
git commit -m "feat: add 5-frame stationarity check with auto-reset"
```

---

### 任务 3：端到端验证

**文件：** 无修改

- [ ] **步骤 1：运行 55-56min 视频**

```bash
python3 inference/predict_track.py --video movie/25866279684-1-192_55-56min.mp4 --no-visualize
```

预期：CSV tracking 行数明显下降（之前 1048，现在应 <800）

- [ ] **步骤 2：检查 CSV 状态分布**

```bash
python3 -c "
import csv
with open('runs/track/25866279684-1-192_55-56min/25866279684-1-192_55-56min_tracks.csv') as f:
    rows = list(csv.DictReader(f))
statuses = {}
for r in rows:
    s = r['status']
    statuses[s] = statuses.get(s, 0) + 1
print(f'Total: {len(rows)}')
for s, n in sorted(statuses.items()):
    print(f'  {s}: {n}')
tracking = statuses.get('tracking', 0)
print(f'')
assert tracking < 800, f'Expected tracking <800, got {tracking}'
print('Stationarity filter working')
"
```

- [ ] **步骤 3：运行 20-23min 视频**

```bash
python3 inference/predict_track.py --video movie/clip_20-23min.mp4 --no-visualize
```

---

### 任务 4：全部测试回归

**文件：** 无修改

- [ ] **步骤 1：运行完整测试套件**

```bash
python3 -m pytest tests/ -v
```

预期：~37 PASS（8+9+4+14+2 新增）

---

### 任务 5：清理中间产物

**文件：** 无修改

- [ ] **步骤 1：清理**

```bash
rm -rf runs/track/
```
