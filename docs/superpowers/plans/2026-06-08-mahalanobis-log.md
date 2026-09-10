# Mahalanobis 追踪质量日志 — 数据收集

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 在追踪 CSV 中新增 `mahalanobis_d2` 列，记录每帧最佳候选的马氏距离，不做任何门控/拦截。追踪结果可导入数据分析脚本（histogram/percentile 分析），为后续阈值选择提供依据。

**架构：** 0 行改动 `score_candidates`，0 行改动主循环逻辑。只在 CSV 输出中新增一列。`predict_track.py` 中新增 `d2` 计算并在 `row` 字典中写入。

**技术栈：** Python 3.12, cv2.KalmanFilter, numpy, pytest

**文件改动：**
- 修改 `inference/predict_track.py`：在 `row` 字典中添加 `mahalanobis_d2` 键
- 修改 `utils/tracker_utils.py`：`mahalanobis_gate` 和 `H` 矩阵已有，无需改动
- 创建 `tools/analyze_d2.py`：数据分析脚本

---

### 任务 1：CSV 输出新增 `mahalanobis_d2` 列

**文件：** 修改 `inference/predict_track.py`

- [ ] **步骤 1：在追踪主循环的 `row` 字典中添加 `mahalanobis_d2`**

找到 `row` 字典定义处（`vx/vy/conf/status` 附近），新增一个键。

```python
row = {
    "frame": frame_idx, "px": round(cx, 1), "py": round(cy, 1),
    "vx": round(vx, 2), "vy": round(vy, 2),
    "conf": round(float(best["conf"]), 4), "status": status,
    "wx": None, "wy": None,
    "mahalanobis_d2": None,  # ← 新增
}
```

- [ ] **步骤 2：在 kf.correct 后、trajectory.push 前计算 d²**

在 `kf.correct(meas)` 之后、`row` 字典构建之前，添加 d² 计算。

```python
kf.correct(meas)
status = "tracking"
lost_counter = 0

# ← 新增：计算 Mahalanobis d²（仅追踪时，非首次帧）
mahalanobis_d2 = None
if status == "tracking":
    try:
        mahalanobis_d2 = round(mahalanobis_gate(
            kf,
            (float(prediction[0, 0]), float(prediction[1, 0])),
            (cx, cy),
        ), 1)
    except:
        pass  # 极低概率 LinalgError，忽略
```

- [ ] **步骤 3：在 CSV 表头中添加新列**

```python
fieldnames = ["frame", "px", "py", "vx", "vy", "conf", "status", "wx", "wy", "mahalanobis_d2"]
```

- [ ] **步骤 4：验证 CSV 包含新列**

```bash
head -3 runs/track/videoplayback_first60s/videoplayback_first60s_tracks.csv
```
预期：第三行（首条数据）有 `mahalanobis_d2` 列，值不为空。

- [ ] **步骤 5：运行全量测试**

```bash
python3 -m pytest tests/ -v --tb=short
```
预期：90/90 通过

- [ ] **步骤 6：Commit**

---

### 任务 2：数据分析脚本

**文件：** 创建 `tools/analyze_d2.py`

- [ ] **步骤 1：创建分析脚本**

```python
#!/usr/bin/env python3
"""Analyze mahalanobis_d2 distribution from tracking output.

Usage:
    python3 tools/analyze_d2.py [--csv runs/track/xxx/xxx_tracks.csv]

Output:
    - Histogram percentiles (50th, 90th, 95th, 99th, max)
    - Which thresholds would reject what % of frames
    - Optional: plot histogram
"""

import csv, math, sys

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            v = r.get("mahalanobis_d2", "")
            if v and v != "None":
                rows.append(float(v))
    return rows

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        import glob
        path = glob.glob("runs/track/*/videoplayback_*_tracks.csv")[-1]
    rows = load(path)
    rows.sort()
    n = len(rows)
    print(f"Samples: {n}")
    for p in [50, 90, 95, 99]:
        print(f"  {p}th percentile: {rows[int(n*p/100)]:.1f}")
    print(f"  max: {rows[-1]:.1f}")
    print()
    for t in [10, 50, 100, 200, 500, 1000, 5000]:
        rejected = sum(1 for v in rows if v > t)
        print(f"  threshold {t:>5}: reject {rejected}/{n} = {100*rejected/max(n,1):.1f}%")

if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：在 first60s 上运行**

```bash
python3 inference/predict_track.py \
  --video /mnt/e/frisbee-detector/movie/videoplayback_first60s.mp4 \
  --conf 0.35
python3 tools/analyze_d2.py
```

- [ ] **步骤 3：在完整 89min 视频上运行（后台）**

```bash
nohup python3 inference/predict_track.py \
  --video /mnt/e/frisbee-detector/movie/videoplayback_trimmed.mp4 \
  --conf 0.35 --no-visualize \
  > /tmp/full_run.log 2>&1 &
python3 tools/analyze_d2.py
```

- [ ] **步骤 4：Commit**

---

### 任务 3：基于数据设定阈值（可选，数据收集后执行）

**文件：** 修改 `utils/tracker_utils.py` + `inference/predict_track.py`

仅在收集到完整 d² 分布后再定阈值。典型选点：99th percentile 或 99.9th。

- [ ] **步骤 1：将阈值写入配置**

```python
GATE_THRESHOLD = 500.0  # 基于 XXX frames 数据，99.9th percentile
```

- [ ] **步骤 2：重新启用门控**

在 `predict_track.py` 中，把 `mahalanobis_d2` 的日志代码扩展为拦截逻辑。

