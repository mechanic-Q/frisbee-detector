# 6态 Kalman 滤波器升级

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 将 `init_kalman()` 从 4 态 (px,py,vx,vy) 升级为 6 态 (px,py,vx,vy,ax,ay)，使 Kalman 能建模加速度，缩小预测位置误差，让 Mahalanobis d² 分离飞盘真实运动与 FP。

**架构：** 只改 `init_kalman()` 的维度常数和矩阵，`H` 矩阵从 2×4 变为 2×6，`mahalanobis_gate` 适配新维度。`score_candidates` 和主循环 0 改动。

**技术栈：** Python 3.12, cv2.KalmanFilter, numpy, pytest

---

### 任务 1：将 init_kalman 升级为 6 态 (px,py,vx,vy,ax,ay)

**文件：** 修改 `utils/tracker_utils.py:init_kalman()`

- [ ] **步骤 1：替换 init_kalman 和 H 矩阵**

```python
def init_kalman() -> cv2.KalmanFilter:
    """Create a 6-state Kalman filter (px,py,vx,vy,ax,ay) for position+velocity+acceleration tracking."""
    kf = cv2.KalmanFilter(6, 2)  # 6 states, 2 measurements (px, py)
    # Measurement matrix: observe px, py directly
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=np.float32)
    # Transition matrix: px'=px+vx*dt+0.5*ax*dt², py'=py+vy*dt+0.5*ay*dt², vx'=vx+ax*dt, vy'=vy+ay*dt, ax'=ax, ay'=ay
    dt = 1.0  # 1 frame
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0.5*dt*dt, 0],
        [0, 1, 0, dt, 0, 0.5*dt*dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1  # Higher for acceleration model
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 100.0
    return kf


# H matrix must match 6-state
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
], dtype=np.float32)
```

- [ ] **步骤 2：更新 mahalanobis_gate 的 docstring（功能不变，H 维度自动适配）**

`mahalanobis_gate` 使用 `H @ P @ H.T`，P 是 `errorCovPost`（6×6），H 是 2×6，矩阵乘法自动适配。无需改动函数体。

- [ ] **步骤 3：修复 predict_track.py 中的 prediction 索引**

当前代码：
```python
prediction = kf.predict()  # returns (6,1) now
pred_pt = (float(prediction[0, 0]), float(prediction[1, 0]))  # still correct
```
`kf.predict()` 返回 (6,1) 而非 (4,1)。但 `prediction[0,0]` 和 `prediction[1,0]` 仍是 px, py，代码不改。

验证：
```python
kf = init_kalman()
p = kf.predict()
assert p.shape == (6, 1), f"Expected (6,1), got {p.shape}"
```

- [ ] **步骤 4：修复 predict_track.py 中 kf.correct 后的 statePost 索引**

当前代码：
```python
vx = float(kf.statePost[2, 0])
vy = float(kf.statePost[3, 0])
```
6 态下 vx, vy 仍在索引 2, 3，不变。ax, ay 在 4, 5，后续可记录。代码不改。

- [ ] **步骤 5：运行全量测试**

```bash
python3 -m pytest tests/ -v --tb=short
```
预期：90/90 通过（所有 tracker_utils 测试使用 `init_kalman()`）

- [ ] **步骤 6：Commit**

```bash
git add utils/tracker_utils.py
git commit -m "feat(kalman): upgrade to 6-state model with acceleration (px,py,vx,vy,ax,ay)"
```

---

### 任务 2：回归验证——first60s d² 分布改善

**文件：** 无改动，运行对比

- [ ] **步骤 1：用 6 态 Kalman 对 first60s 生成标注视频**

```bash
python3 inference/predict_track.py \
  --video /mnt/e/frisbee-detector/movie/videoplayback_first60s.mp4 \
  --conf 0.35
```

- [ ] **步骤 2：分析 d² 分布**

```bash
python3 tools/analyze_d2.py runs/track/videoplayback_first60s/videoplayback_first60s_tracks.csv
```

- [ ] **步骤 3：与 4 态基线对比**

```python
# 对比指标：
# 1. 追踪行数（6 态应与原版持平）
# 2. d² 分布（6 态的 50th/90th/95th percentile 应显著低于 4 态）
# 3. 置信度分布（中位数不应劣化）
```

- [ ] **步骤 4：如果 d² 分布明显左移（6 态的 50th < 4 态的 50th），则可考虑设定阈值。**

- [ ] **步骤 5：Commit 分析结果和视频**

```bash
git commit --allow-empty -m "perf(analysis): 6-state Kalman d2 distribution comparison"
```
