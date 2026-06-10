# v7 Data-Driven Precision — 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 通过数据驱动方法（替换背景为bbox级hard negatives + 添加TP帧）将per-box FP率从89%降至<20%，同时保持帧检测率>=50%。

**架构：** 1个新脚本（数据预处理）+ 2个运维步骤（训练 + 评估）。先快速验证（imgsz=640, ~10-15min），通过后再精训（imgsz=1280）。

**技术栈：** Python 3.10, ultralytics (YOLOv8), OpenCV, PIL

**设计规格：** `docs/superpowers/specs/2026-05-14-v7-data-driven-precision-design.md`

---

## 文件变更

| 文件 | 变更 | 职责 |
|------|------|------|
| `tools/prep_v7_data.py` | **新增** | 数据预处理：删背景、复制FP crops、提取TP帧 |
| `models/train.py` | 无变更 | 已有 `--resume` 参数可从v3 fine-tune |
| `configs/frisbee_merged.yaml` | 自动重新生成 | `prep_v7_data.py` 调用 `write_yaml_config()` |

无测试文件要求（预处理脚本有 `--dry-run` 模式验证）。

---

### 任务 1：编写数据预处理脚本 `tools/prep_v7_data.py`

**文件：**
- 新增：`tools/prep_v7_data.py`

**分析：** 需要完成三个数据操作：(1) 删除当前训练集中的283个空标签背景图，(2) 复制178个FP crops作为新背景，(3) 从55-56min测试视频中提取19个TP帧并生成YOLO标签。

裁剪图文件名格式：`crop_NNNN_fXXXXX_cC.CONF.jpg`，其中XXXXX是帧号。TP裁剪图覆盖19个唯一帧（1,2,4,5,6,7,8,9,59,60,61,62,63,64,65,66,68,94,96）。

- [ ] **步骤 1：创建 `tools/prep_v7_data.py` 脚本**

```python
"""v7 data preparation: replace backgrounds with FP crops + add TP frames.

Usage:
    python3 tools/prep_v7_data.py                    # execute
    python3 tools/prep_v7_data.py --dry-run           # preview only
"""
```

脚本结构：

1. **argparse**: `--dry-run`（仅打印将执行的操作），`--video`（默认55-56min视频路径），`--model`（默认v3模型路径，用于TP帧标签生成）

2. **删除283个背景图**：
   - 遍历 `data/datasets/frisbee_merged/labels/train/*.txt`
   - 找到文件大小为0的标签文件（背景图）
   - 删除对应的图片文件和标签文件
   - 统计删除数量，打印

3. **复制178个FP crops**：
   - 读取 `data/perbox_crops/review_results.csv`
   - 过滤 `result == "FP"` 的行
   - 将每个FP crop复制到 `data/datasets/frisbee_merged/images/train/`，命名为 `hardneg_v7_{original_filename}`
   - 创建对应的空标签文件 `hardneg_v7_{stem}.txt` 在 `labels/train/`

4. **提取19个TP帧 + 生成YOLO标签**：
   - 从CSV中读取所有 `result == "TP"` 的行，收集唯一帧号
   - 用OpenCV打开55-56min视频，逐帧读取并保存到 `images/train/tp_v7_{frame_num:05d}.jpg`
   - 用v3模型对每个TP帧做推理（`model.predict(frame, conf=0.20, imgsz=1280)`）
   - 将预测结果转为YOLO标签格式（`class cx cy w h`），保存到 `labels/train/tp_v7_{frame_num:05d}.txt`
   - **重要**：只保存class 0（frisbee）的预测框，conf >= 0.35

5. **重新生成YAML配置**：
   - 调用 `utils.dataset.write_yaml_config()` 更新 `configs/frisbee_merged.yaml`

6. **最终统计**：
   - 打印：删除的背景数、新增的FP crop数、新增的TP帧数、最终训练集总大小、正负样本比例

- [ ] **步骤 2：Dry-run测试**

```bash
python3 tools/prep_v7_data.py --dry-run
```

验证输出：
- 确认找到283个待删除背景
- 确认178个FP crops将被复制
- 确认19个TP帧将被提取
- 确认不会修改任何文件

- [ ] **步骤 3：实际执行预处理**

```bash
python3 tools/prep_v7_data.py
```

验证：
- 打印的统计数字合理（~1698张训练图，~11.8%背景）
- `data/datasets/frisbee_merged/images/train/` 包含 `hardneg_v7_*` 和 `tp_v7_*` 文件
- `data/datasets/frisbee_merged/labels/train/` 包含对应标签文件
- `tp_v7_*` 标签文件非空（包含YOLO格式bbox）

```bash
ls data/datasets/frisbee_merged/images/train/ | wc -l
ls data/datasets/frisbee_merged/images/train/ | grep "^hardneg_v7_" | wc -l
ls data/datasets/frisbee_merged/images/train/ | grep "^tp_v7_" | wc -l
ls data/datasets/frisbee_merged/labels/train/ | grep "^tp_v7_" | xargs -I{} sh -c 'test -s data/datasets/frisbee_merged/labels/train/{} && echo "OK" || echo "EMPTY"'
```

---

### 任务 2：Phase 1 快速训练（imgsz=640, ~10-15min）

**文件：**
- 无代码变更。纯运维操作。

**重要约束：**
- YOLO训练 >10min，**必须**在tmux中运行
- 双重嵌套bug：`project="runs/detect"` 导致输出到 `runs/detect/runs/detect/`
- 使用 `--resume` 从v3 fine-tune（不是从头训练）

- [ ] **步骤 1：确认v3模型存在**

```bash
ls -lh runs/detect/frisbee_det_s_v3/weights/best.pt
```

- [ ] **步骤 2：启动tmux训练会话**

```bash
tmux new-session -d -s train -c /mnt/e/frisbee-detector
tmux send-keys -t train "python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --resume runs/detect/frisbee_det_s_v3/weights/best.pt \
  --model-size s \
  --name frisbee_det_s_v7_quick \
  --epochs 30 --imgsz 640 --batch 2 \
  --patience 10 --box 5 --close-mosaic 10 \
  --workers 2" Enter
```

确认启动：
```bash
tmux ls
```

- [ ] **步骤 3：监控训练启动**

等待30秒后检查：
```bash
sleep 30 && tmux capture-pane -t train -p | tail -20
```

- [ ] **步骤 4：等待训练完成**

定期检查（每2-3分钟）：
```bash
tmux capture-pane -t train -p | tail -5
```

训练完成标志：tmux会话不再活跃，或输出显示 "Results" / "best.pt"。

- [ ] **步骤 5：修复双重嵌套路径**

```bash
if [ -d "runs/detect/runs/detect/frisbee_det_s_v7_quick" ]; then
  mv runs/detect/runs/detect/frisbee_det_s_v7_quick runs/detect/frisbee_det_s_v7_quick
fi
```

- [ ] **步骤 6：确认模型文件**

```bash
ls -lh runs/detect/frisbee_det_s_v7_quick/weights/best.pt
```

预期：~22MB。

---

### 任务 3：Phase 1 评估

**文件：**
- 无代码变更。纯运维操作。

- [ ] **步骤 1：验证集评估**

```bash
python3 models/train.py --validate-only \
  --model-path runs/detect/frisbee_det_s_v7_quick/weights/best.pt \
  --data configs/frisbee_merged.yaml --imgsz 640
```

记录：mAP50, Precision, Recall。

- [ ] **步骤 2：55-56min测试视频评估**

```bash
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v7_quick/weights/best.pt \
  --video movie/25866279684-1-192_55-56min.mp4 --conf 0.35
```

记录：帧检测率、平均dets/帧、置信度分布。

- [ ] **步骤 3：20-23min测试视频评估**

```bash
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v7_quick/weights/best.pt \
  --video movie/clip_20-23min.mp4 --conf 0.35
```

记录：同上。

- [ ] **步骤 4：决策**

对比阈值：

| 指标 | v3基线 | v7目标 | v7实际 | 通过? |
|------|:------:|:------:|:------:|:-----:|
| 55-56min帧检测率 | 69.9% | >=50% | ? | ? |
| 20-23min帧检测率 | 61.4% | >=50% | ? | ? |
| 平均dets/帧(55-56) | 1.86 | <1.0 | ? | ? |
| Per-box FP率 | 89% | <20% | ? | ? |

**通过** → 进入任务4（Phase 2精训）
**未通过** → 停止，分析失败原因，考虑VLM扩增或调整数据策略

注意：per-box FP率需要手动抽查。从每个测试视频的检测结果中随机采样50个检测框，人工判断是否为飞盘。

---

### 任务 4：Phase 2 生产训练（imgsz=1280, ~1-2h）

**仅当任务3所有指标通过时执行。**

- [ ] **步骤 1：启动tmux精训会话**

```bash
tmux new-session -d -s train -c /mnt/e/frisbee-detector
tmux send-keys -t train "python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --resume runs/detect/frisbee_det_s_v3/weights/best.pt \
  --model-size s \
  --name frisbee_det_s_v7 \
  --epochs 100 --imgsz 1280 --batch 2 \
  --patience 20 --box 5 --close-mosaic 10 \
  --workers 2" Enter
```

- [ ] **步骤 2：修复双重嵌套路径**

```bash
if [ -d "runs/detect/runs/detect/frisbee_det_s_v7" ]; then
  mv runs/detect/runs/detect/frisbee_det_s_v7 runs/detect/frisbee_det_s_v7
fi
```

- [ ] **步骤 3：评估（同任务3流程）**

在两个测试视频上用conf=0.35评估，记录所有指标。

- [ ] **步骤 4：更新 `configs/models.py`**

如果v7优于v3：
- 更新 `DEFAULT_MODEL` 指向v7
- 更新 `DEFAULT_CONF` 如果最优阈值变化

- [ ] **步骤 5：更新 AGENTS.md**

在模型命名约定表中添加v7行：
```
frisbee_det_s_v7   → v7 (data-driven FP reduction, 178 hard neg crops + 22 TP frames)
```

---

## 回滚方案

如果v7结果不如v3：
1. 恢复原始训练数据：重新运行 `python3 tools/merge_datasets.py`（从源数据集重建）
2. v3模型文件未受影响（训练不修改原始模型）
3. v7 quick和v7模型可以删除：`rm -rf runs/detect/frisbee_det_s_v7*`

---

## 预估时间

| 任务 | 预估时间 | 依赖 |
|------|:--------:|------|
| 任务1：编写+测试预处理脚本 | 10-15min | 无 |
| 任务1：执行预处理 | 2-3min | 脚本就绪 |
| 任务2：快速训练 | 10-15min | 数据就绪 |
| 任务3：评估 | 5-10min | 模型就绪 |
| 任务4：精训（如果通过） | 1-2h | Phase 1通过 |
| **总计（Phase 1）** | **~30-45min** | |
| **总计（含Phase 2）** | **~2.5-3h** | |
