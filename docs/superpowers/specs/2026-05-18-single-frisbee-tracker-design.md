# 单飞盘追踪器设计

> **目标：** 替换当前 ByteTrack 多目标追踪，用单目标 Kalman 滤波 + 候选评分，每帧只追踪一个飞盘。动画画出滑动窗口轨迹线。
>
> **背景：** 场上同一时间只有一个飞盘。ByteTrack 追踪 107 条轨迹中 71 条 是静止 FP（白帽子/椅子/人头），根本原因是"多目标追踪"这个方向错了。改为单目标追踪+每帧候选评分，直接从源头解决了问题。
>
> **前置条件：** `inference/predict_track.py` 已有 CLI 骨架（argparse + YOLO 加载 + CSV 导出 + OpenCV 标注 + 标定自动查找）。

## 对比现状

| | 当前 (ByteTrack) | 新方案 |
|------|:---:|:---:|
| 追踪目标数 | 107 条 | **1 条** |
| FP 轨迹 | 71 条静止 (66%) | **0**（不选静止候选） |
| 轨迹可视化 | 无 | **红→黄淡出线** |
| 外部依赖 | ultralytics ByteTrack + lap | **无新依赖** |
| 抗丢失 | ByteTrack buffer 30帧 | Kalman 预测 15帧 |
| 评分 | IoU 匹配 | 加权打分（运动+置信度+面积） |

## 架构

```
  YOLO 检测
     ↓
  候选框列表 [(bbox, conf), ...]
     ↓
  ┌─ Kalman 预测 (px, py)
  │      ↓
  │  候选评分器 (每个候选打分)
  │    score = 0.4 × motion_likelihood
  │          + 0.3 × confidence
  │          + 0.3 × area_consistency
  │      ↓
  │  选最高分候选 → Kalman 更新
  │      ↓
  ├── 轨迹队列 push (px, py, frame)
  │      ↓
  └── 绘制滑动窗口轨迹 (红→黄淡出, 50帧)
```

## 组件设计

### 1. Kalman 滤波器（utils/tracker_utils.py）

- 状态：4 维 `(px, py, vx, vy)`
- 观测：2 维 `(px, py)`（bbox 中心点）
- 测量噪声：高（YOLO 位置有抖动）
- 过程噪声：高（飞盘可加速）
- 无新检测→最多纯预测 15 帧
- 超 15 帧或 Kalman 预测的 (px, py) 超出图像边界 w×h → 轨迹重置，进入搜索模式

> **主循环变化：** 不再调用 `model.track()`（ByteTrack），改为逐帧 `model(frame, conf=args.conf, verbose=False)` 取检测框。YOLO 只做检测，追踪完全由 Kalman + 评分器接管。

初始化（`init_kalman()`）：

```python
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
kf.transitionMatrix = np.array([
    [1,0,1,0],[0,1,0,1],
    [0,0,1,0],[0,0,0,1]
], dtype=np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.003  # 初始值，可能需要调优
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1  # 初始值，可能需要调优
```

### 2. 候选评分器（utils/tracker_utils.py）

函数 `score_candidates(candidates: list[dict], prediction: tuple|None) -> int`：

**无历史（首帧/刚丢失重搜）：**
```python
score = 0.7 * conf + 0.3 * area_size_score
```
其中 `area_size_score = max(0, min(1, 1.0 - abs(log2(area / 200))))`（假设飞盘框约 200px²，偏差越大分越低）

**有历史：**
```python
motion_score = max(0, min(1, 1.0 - dist_to_prediction / MAX_EXPECTED_DISPLACEMENT))
# MAX_EXPECTED_DISPLACEMENT = 50px (25fps下飞盘最大帧间位移)
area_consistent_score = max(0.0, 1.0 - abs(log2(float(area) / float(trajectory.areas[-1]))))
aspect_score = max(0, min(1, float(box_width) / float(box_height) / 2.0))
score = (0.35 * motion_score + 0.25 * conf + 0.25 * area_consistent_score
         + 0.15 * aspect_score)
```

返回最高分候选的索引。其中 `aspect_score` 奖励宽比高大的框（飞盘框宽高比通常 >0.8，人框 <0.6），抑制快跑的人。

返回最高分候选的索引。

### 3. 轨迹管理（utils/tracker_utils.py）

- `Trajectory` 类：维护 `(px, py, frame)` 环形缓冲，最大 2000 帧
- 同步维护 `areas` 列表：每帧选中框的面积，与坐标同步 push
- 方法：`push(px, py, frame, area)` → 追加坐标和面积
- 方法：`get_window(n=50)` → 返回最近 N 帧的坐标列表（用于绘制）
- 方法：`last_position()` → 返回最后一个坐标

### 4. 视频标注

**滑动窗口轨迹线（50帧 ~2s）：**
- 遍历 `trajectory.get_window(50)`，从旧到新
- 颜色：旧帧 (0, 127, 255) 橙色 → 新帧 (0, 0, 255) 红色 → 当前位置 (0, 255, 0) 绿色圆
- `cv2.polylines` 连接各点
- 不画点标号（避免遮挡）

**其他标注：**
- 飞盘框：绿色矩形 + `"Frisbee"` 标签
- 帧号 + 飞盘状态（追踪中/搜索中）：画面左上角
- 每帧检测数：右上角

### 5. 输出

| 格式 | 内容 | 变化 |
|------|------|------|
| `.mp4` | 框 + 轨迹线 + 状态 | **改：** ByteTrack ID 去掉，轨迹线加上 |
| `.csv` | `frame, px, py, vx, vy, conf, status` | **改：** 无 track_id，每帧最多 1 行，加 vx, vy, status |

status: `"tracking"`（正常追踪），`"predicting"`（Kalman 纯预测），`"searching"`（丢失后重搜）

### 6. 错误处理

| 场景 | 处理 |
|------|------|
| YOLO 无检测（`r.boxes is None`） | Kalman 预测 + 丢失计数器 +1，标记 predicting |
| 预测 >15 帧 | 丢弃轨迹，标记 searching，下帧重新初始化 |
| 候选框全窄高（宽高比 <0.5） | 降低其面积一致性分数（飞盘框宽比高大） |
| 丢失后发现多个候选 | 首帧评分只用 conf + 面积一致性 |

## 文件变更

| 文件 | 操作 | 职责 |
|------|------|------|
| `utils/tracker_utils.py` | 新建 | Kalman 初始化 + 候选评分 + Trajectory 管理 |
| `inference/predict_track.py` | 修改 | 替换 ByteTrack 循环为单目标 Kalman 循环 |
| `tests/test_tracker_utils.py` | 新建 | 候选评分器 + Trajectory + Kalman 预测测试 |

## 测试

`tests/test_tracker_utils.py` 包含：

1. `test_kalman_init_state` — Kalman 初始化后状态为 4 维
2. `test_kalman_predict_update` — 馈入同一观测后 Kalman 输出接近观测
3. `test_trajectory_push_window` — push 后 get_window 返回正确数量的帧
4. `test_trajectory_ring_behavior` — 环形缓冲无误
5. `test_score_prioritizes_confidence` — 无历史时高分置信度胜出
6. `test_score_uses_motion_when_predicting` — 有预测时运动一致性胜出
7. `test_aspect_score_favors_wide_boxes` — 宽高比评分偏好横长框（飞盘）而非竖高框（人）

## 不在此设计中的范围

- Homography 场地坐标映射（保留现有 utils/homography.py 用法）
- 标定 JSON 自动查找（保留现有 find_calibration 逻辑）
- 错误率 / F1 评估脚本（后续再做）
