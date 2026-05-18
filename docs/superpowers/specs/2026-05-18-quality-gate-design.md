# 质量门控设计（候选评分 + 轨迹惰性分析）

> **目标：** 消除单目标追踪器对白帽子/人头等静止 FP 的"黏上"行为。通过修改评分函数奖励运动候选 + 每 5 帧检查轨迹是否静止，静止则自动重置。
>
> **前置条件：** `utils/tracker_utils.py` 已有 score_candidates 评分函数（置信度+运动一致性+面积一致性+宽高比）。`inference/predict_track.py` 已有 Kalman 单目标追踪循环。

## 核心问题

当前评分函数的结构性缺陷：`motion_score` 奖励"离 Kalman 预测近"——静止物体（Kalman 预测=当前观测）得分反而比运动物体高。

| 信号 | 静止白帽子 | 运动飞盘 |
|------|:---:|:---:|
| motion_score (35%) | **1.0**（距预测=0px） | 0.8（距预测≠0px） |
| 总分 | **0.775** | 0.73 |

## 设计

### 1. 评分权重调整（utils/tracker_utils.py）

**有历史时评分公式：**

```
score = 0.30 * motion_score      # 预测一致性（降权）
      + 0.20 * confidence         # YOLO 置信度
      + 0.20 * area_consistency   # 面积连续性
      + 0.15 * aspect_score       # 宽高比
      + 0.15 * speed_score        # 【新增】近 3 帧平均位移
```

**speed_score 公式：**
```python
if len(list(trajectory._pts)) >= 3:
    recent = list(trajectory._pts)[-3:]
    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    avg_speed = np.sqrt(dx**2 + dy**2) / max(len(recent) - 1, 1)
    speed_score = min(1.0, avg_speed / MIN_DISPLACEMENT)
else:
    speed_score = 0.5  # neutral when not enough data
# MIN_DISPLACEMENT = 5.0 px/frame
```

静止物体 avg_speed ≈ 0 → speed_score = 0
运动物体 avg_speed ≥ 5 → speed_score = 1

### 2. 轨迹惰性分析（inference/predict_track.py）

每 5 帧检查轨迹是否静止，静止则重置 tracker。

```python
STATIONARY_CHECK_INTERVAL = 5
STATIONARY_MAX_DISPLACEMENT = 8.0  # px over 5 frames

# 在 main 循环中，每处理 5 帧后：
if status == "tracking" and frame_idx % STATIONARY_CHECK_INTERVAL == 0:
    recent = trajectory.get_window(5)
    if len(recent) >= 5:
        max_d = max(
            np.sqrt((recent[i][0] - recent[0][0])**2 + (recent[i][1] - recent[0][1])**2)
            for i in range(1, 5)
        )
        if max_d < STATIONARY_MAX_DISPLACEMENT:
            kf = init_kalman()
            trajectory = Trajectory()
            status = "searching"
            lost_counter = 0
            continue  # skip drawing this frame
```

### 3. 改动范围

| 文件 | 操作 | 职责 |
|------|------|------|
| `utils/tracker_utils.py` | 修改 | 评分权重调整 + 添加 speed_score |
| `inference/predict_track.py` | 修改 | 添加 5 帧惰性检查 + 重置逻辑 |
| `tests/test_tracker_utils.py` | 修改 | 添加 speed_score 测试 + 惰性重选后权重测试 |

## 测试

新增测试：
1. `test_speed_score_favors_moving_boxes` — 运动候选（近 3 帧位移 >5px）得分高于静止候选
2. `test_stationary_score_below_threshold` — 静止候选总分仍低于活跃候选

## 不在此设计中的范围

- 修改 YOLO 检测器（只改追踪层）
- Homography 坐标映射（保留现有逻辑）
- 多目标追踪（保持单目标）
