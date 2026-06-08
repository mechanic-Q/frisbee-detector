# Mahalanobis 门控 v3 — 事后验证方案

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 在 `predict_track.py` 主循环中，对已被 `score_candidates`（原始欧氏，0 改动）选出的最佳候选做 Mahalanobis 事后验证——通过的正常 Kalman correct，不通过的降级为纯预测。

**架构：** 两层独立——`score_candidates` 保持原始 shadow 版逻辑不动；`predict_track.py` 主循环中新增一行验证，在候选被选出后、`kf.correct` 之前插入马氏门控判断。

**技术栈：** Python 3.12, cv2.KalmanFilter, numpy, pytest

**关键约束：** `utils/tracker_utils.py` 的 `score_candidates` 签名和内部逻辑恢复至与 main 一致。

---

### 任务 1：回退 tracker_utils.py 到原始状态

**文件：** 修改 `utils/tracker_utils.py`

- [ ] **步骤 1：用 git show 恢复原始文件**

```bash
cd /mnt/e/frisbee-detector/.worktrees/mahalanobis-gating
git show main:utils/tracker_utils.py > utils/tracker_utils.py
```

- [ ] **步骤 2：验证恢复结果**

```bash
diff <(git show main:utils/tracker_utils.py) utils/tracker_utils.py
```
预期：无差异输出

- [ ] **步骤 3：运行原始测试确认基线健康**

```bash
python3 -m pytest tests/test_tracker_utils.py -v --tb=short
```
预期：11/11 通过（原始测试，无 Mahalanobis 测试）

- [ ] **步骤 4：Commit**

```bash
git add utils/tracker_utils.py
git commit -m "revert: restore tracker_utils.py to main baseline"
```

---

### 任务 2：添加 mahalanobis_gate 函数（仅新增，不改现有逻辑）

**文件：** 修改 `utils/tracker_utils.py`

- [ ] **步骤 1：在 `init_kalman()` 后面新增 `H` 矩阵和 `mahalanobis_gate()` 函数**

只在文件中追加新代码，0 改动原有函数。

```bash
cd /mnt/e/frisbee-detector/.worktrees/mahalanobis-gating && python3 << 'PYEOF'
t = open('utils/tracker_utils.py').read()

# Add H matrix after init_kalman returns
insert_point = t.rfind('    return kf\n')
if insert_point < 0:
    insert_point = t.rfind('    return kf\r\n')
insert_at = t.index('\n', insert_point) + 1

new_code = '''
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

GATE_THRESHOLD = 13.8155  # chi²_{0.999}(df=2) — only reject extreme statistical outliers


def mahalanobis_gate(
    kf: cv2.KalmanFilter,
    prediction: tuple[float, float],
    measurement: tuple[float, float],
) -> float:
    innov = np.array([
        [measurement[0] - prediction[0]],
        [measurement[1] - prediction[1]],
    ], dtype=np.float32)
    P = kf.errorCovPost
    S = H @ P @ H.T + kf.measurementNoiseCov
    S_inv = np.linalg.inv(S)
    return float((innov.T @ S_inv @ innov)[0, 0])
'''

t = t[:insert_at] + new_code + t[insert_at:]
open('utils/tracker_utils.py', 'w').write(t)
print('OK')
PYEOF
```

- [ ] **步骤 2：确认原有逻辑未被触碰**

```bash
grep -n 'def score_candidates' utils/tracker_utils.py
grep -n 'def init_kalman' utils/tracker_utils.py
```
预期：`score_candidates` 签名不变，`init_kalman` 逻辑不变

- [ ] **步骤 3：导入 mahalanobis_gate 确认语法正确**

```bash
python3 -c "from utils.tracker_utils import mahalanobis_gate, GATE_THRESHOLD; print('import OK')"
```
预期：`import OK`

- [ ] **步骤 4：运行原始测试**

```bash
python3 -m pytest tests/test_tracker_utils.py -v --tb=short
```
预期：11/11 通过

- [ ] **步骤 5：Commit**

```bash
git add utils/tracker_utils.py
git commit -m "feat(tracker): add mahalanobis_gate utility, zero changes to score_candidates"
```

---

### 任务 3：在 predict_track.py 中加入事后验证

**文件：** 修改 `inference/predict_track.py`

- [ ] **步骤 1：导入 mahalanobis_gate**

修改导入行，添加 `mahalanobis_gate, GATE_THRESHOLD`:

```bash
cd /mnt/e/frisbee-detector/.worktrees/mahalanobis-gating

# 找到 import 行并修改
sed -i 's/from utils.tracker_utils import init_kalman, score_candidates, Trajectory/from utils.tracker_utils import init_kalman, score_candidates, Trajectory, mahalanobis_gate, GATE_THRESHOLD/' inference/predict_track.py
```

- [ ] **步骤 2：在 kf.correct 之前插入验证代码**

找到主循环中 `kf.correct(meas)` 被调用的位置，在前面插入马氏验证块。精确位置：`trajectory.push(cx, cy, area)` 之后、`current_status = "tracking"` 之前。

```bash
# 找到需要插入的精确位置
cd /mnt/e/frisbee-detector/.worktrees/mahalanobis-gating

python3 << 'PYEOF'
t = open('inference/predict_track.py').read()

# Find the block after trajectory.push, before current_status
old = '''                trajectory.push(cx, cy, area)
                current_status = "tracking"'''

new = '''                trajectory.push(cx, cy, area)

                # Mahalanobis post-gating: verify selected candidate
                if is_tracking:
                    d2 = mahalanobis_gate(kf, prediction, (cx, cy))
                    if d2 > GATE_THRESHOLD:
                        current_status = "predicting"
                        continue

                current_status = "tracking"'''

# Remove the second "current_status" since we now set it conditionally
# The original line sets current_status unconditionally, we changed to conditional
if old in t:
    t = t.replace(old, new)
    open('inference/predict_track.py', 'w').write(t)
    print('OK: main gating block inserted')
else:
    print('ERROR: marker not found. Manual check needed.')
PYEOF
```

- [ ] **步骤 3：验证语法**

```bash
python3 -c "import ast; ast.parse(open('inference/predict_track.py').read()); print('syntax OK')"
```

- [ ] **步骤 4：运行全量测试**

```bash
python3 -m pytest tests/ -v --tb=short
```

- [ ] **步骤 5：Commit**

```bash
git add inference/predict_track.py
git commit -m "feat(tracker): Mahalanobis post-gating in predict_track main loop"
```

---

### 任务 4：回退测试文件到原始状态

**文件：** 修改 `tests/test_tracker_utils.py`

- [ ] **步骤 1：恢复原始测试文件**

```bash
git show main:tests/test_tracker_utils.py > tests/test_tracker_utils.py
```

- [ ] **步骤 2：运行测试确认 11/11 通过**

```bash
python3 -m pytest tests/test_tracker_utils.py -v --tb=short
```
预期：11/11 通过

- [ ] **步骤 3：Commit**

```bash
git add tests/test_tracker_utils.py
git commit -m "revert: restore tests to main baseline"
```

---

### 任务 5：回归验证——对比原始 shadow 基线

**文件：** 无代码改动

- [ ] **步骤 1：用 v3 版跑 first60s**

```bash
cd /mnt/e/frisbee-detector/.worktrees/mahalanobis-gating
python3 inference/predict_track.py \
  --video /mnt/e/frisbee-detector/movie/videoplayback_first60s.mp4 \
  --conf 0.35
```

- [ ] **步骤 2：对比原始版指标**

```bash
python3 -c "
import csv, math

def analyze(path, label):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    jumps = 0
    for i in range(1, len(rows)):
        f0, x0, y0 = int(rows[i-1]['frame']), float(rows[i-1]['px']), float(rows[i-1]['py'])
        f1, x1, y1 = int(rows[i]['frame']), float(rows[i]['px']), float(rows[i]['py'])
        if 1 <= (f1-f0) <= 5 and ((x1-x0)**2+(y1-y0)**2) > 40000:
            jumps += 1
    confs = sorted([float(r['conf']) for r in rows])
    print(f'{label}: {len(rows)} rows, {jumps} jumps, med conf {confs[len(confs)//2]:.3f}')

analyze('runs/track/videoplayback_first60s/videoplayback_first60s_tracks.csv', 'v3 POST-GATE')
"
```
预期：
- 追踪行数 ≈ 1068（±5%，与原始持平）
- 大跳变 ≤ 70（不劣于原始）
- 中位置信度 ≥ 0.50（不劣于原始）

- [ ] **步骤 3：Commit**

```bash
git commit --allow-empty -m "perf: verified Mahalanobis post-gating matches shadow baseline on first60s"
```
