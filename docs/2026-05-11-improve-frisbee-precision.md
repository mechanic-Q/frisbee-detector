# 飞盘检测精度提升 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 合并所有已下载飞盘数据集 + 自动标注视频帧，重新训练 YOLOv8s 模型，提高飞盘检测精度（减少误检）。

**Architecture:** 数据层——2个正样本数据集（UltimateML + kaggle_frisbee）+ 2个负样本源（frisbee_dataset全转负样本 + pseudo-labeling零检测帧）+ pseudo-labeling正样本帧；训练层——用合并数据集从 COCO 预训练权重重新训练 YOLOv8s（imgsz=1280, 精度优先参数，训练名 v2）；推理层——验证集上调优 conf 阈值，对比新旧模型。

**Tech Stack:** Python 3.10, ultralytics (YOLOv8), OpenCV, PyYAML, PyTorch

---

## 项目目的

构建一个高精度的极限飞盘（frisbee）目标检测模型，用于从视频/图片中准确识别飞盘位置，减少误检和漏检。

当前模型在55-56分钟测试片段上的问题：
- 32% 的帧零检测（漏检）
- 检测到的帧中平均每帧2.25个框（视频只有1个飞盘，多余的均为误检）
- 61.5% 的检测置信度低于0.50

本计划通过 **扩充训练数据** 和 **优化训练策略** 来解决上述问题，优先提升精度（减少误检）。

---

## Git 管理策略

- **开发分支：** `improve-precision`（从 `dev` 分支创建）
- **每完成一个 Task 即 commit**，commit message 格式：`feat/fix/data/chore: 描述`
- 所有训练产出（模型权重、训练日志）不提交 git（已被 .gitignore 排除）
- **合并时机：** 全部 Task 完成并验证通过后，合并 `improve-precision` → `dev` → `main`

### 需要提交的文件
| 文件 | 类型 | 说明 |
|------|------|------|
| `tools/convert_kaggle_frisbee.py` | 新建 | kaggle 数据集转换脚本 |
| `tools/convert_frisbee_to_negatives.py` | 新建 | frisbee_dataset 转负样本脚本 |
| `tools/extract_frames.py` | 新建 | 视频抽帧脚本 |
| `tools/auto_label.py` | 新建 | 自动标注脚本 |
| `tools/merge_datasets.py` | 新建 | 数据集合并脚本 |
| `configs/frisbee_merged.yaml` | 新建 | 合并数据集配置 |
| `models/train.py` | 修改 | 增加训练参数 |

### 不提交的文件
| 文件 | 原因 |
|------|------|
| `data/datasets/frisbee_kaggle/` | 数据量大，可从脚本重新生成 |
| `data/datasets/frisbee_negatives/` | 同上 |
| `data/datasets/frisbee_pseudo/` | 同上 |
| `data/datasets/frisbee_merged/` | 同上 |
| `data/frames/` | 抽帧图片，可重新生成 |
| `runs/` | 训练产出，可复现 |
| `movie/*.mp4` | 视频文件 |
| `*.pt` 权重文件 | 太大 |

---

## Task 1: 转换 kaggle_frisbee 数据集

**目的：** 将 kaggle_frisbee 的2类标注统一为单类飞盘标注，转换为标准 YOLO 格式（images/ + labels/ 分目录）。

**改进目的：** 引入 kaggle 数据增加场景多样性（851张图，995个飞盘实例，0张负样本）。也为合并数据集增加更多标注数据。

**Files:**
- Create: `tools/convert_kaggle_frisbee.py`

- [ ] **Step 1: 编写转换脚本**

脚本功能：
- 读取 `/mnt/e/firsbee/03_datasets/kaggle_frisbee/train/` 下的 jpg+txt 混合目录
- 将所有 class ID 统一设为 0（原 class 0 有475个实例，class 1 有520个实例，均为飞盘）
- 按 80/10/10 随机划分 train/val/test（SEED=42）
- 所有851张图都有标注（0张负样本）
- 输出到 `data/datasets/frisbee_kaggle/` 标准 YOLO 目录结构
- 生成 `frisbee.yaml`

- [ ] **Step 2: 运行转换脚本，验证输出**

Run: `python3 tools/convert_kaggle_frisbee.py`
Expected: `data/datasets/frisbee_kaggle/` 包含 images/labels 各三分片子目录及 `frisbee.yaml`

验证：检查图片和标签数量一致（851张），标签内容均为 class 0，无空标签文件。

- [ ] **Step 3: Commit**

```bash
git checkout dev && git checkout -b improve-precision
git add tools/convert_kaggle_frisbee.py
git commit -m "data: add kaggle_frisbee dataset conversion script"
```

---

## Task 2: frisbee_dataset 全部转负样本

**目的：** frisbee_dataset 的381个标注中149个（39%）面积>50%，属于全图标注或超大框错误。由于无法可靠区分哪些小框是真正有效的飞盘标注，将全部80张图片转为负样本（删除所有标注，只保留空标签文件）。

**改进目的：** 80张负样本帮助模型学习"什么不是飞盘"，对抑制误检有益。避免低质量标注污染训练数据。

**Files:**
- Create: `tools/convert_frisbee_to_negatives.py`

- [ ] **Step 1: 编写转负样本脚本**

脚本功能：
- 读取 `/mnt/e/firsbee/03_datasets/frisbee_dataset/` 的 images 和 labels
- **删除所有标注**：每张图片的标签文件替换为空文件
- 保留所有图片作为负样本（空标签 = 背景图，无飞盘）
- 输出到 `data/datasets/frisbee_negatives/` 标准 YOLO 目录结构
- 生成 `frisbee.yaml`（nc=1, names=['frisbee']）
- 报告：80张图片全部转为负样本

- [ ] **Step 2: 运行脚本，验证输出**

Run: `python3 tools/convert_frisbee_to_negatives.py`
Expected: 80张图片，80个空标签文件，0个非空标签。输出目录 `data/datasets/frisbee_negatives/`。

- [ ] **Step 3: Commit**

```bash
git add tools/convert_frisbee_to_negatives.py
git commit -m "data: add frisbee_dataset to negatives conversion script"
```

---

## Task 3: 从视频抽取帧

**目的：** 从多个视频中均匀间隔抽帧，最大化场景多样性（远景/近景/不同光线/不同角度）。

**改进目的：** 仅用标注数据训练的模型泛化能力有限。从实际应用视频中抽取帧可增加最贴近真实使用场景的训练数据。pseudo-labeling 会自动区分正负样本。

**Files:**
- Create: `tools/extract_frames.py`

- [ ] **Step 1: 编写抽帧脚本**

抽帧配置：
| 视频 | 间隔 | 最多帧数 | 场景特点 |
|------|------|----------|----------|
| `25866279684-1-192.mp4` | 每5秒1帧 | 1200 | 比赛远景 |
| `clip_20-23min.mp4` | 每1秒1帧 | 200 | 比赛片段 |
| `clip-5.mp4` | 每0.2秒1帧 | 100 | 近景跟踪 |
| `backhand_2.mp4` | 每0.2秒1帧 | 100 | 投掷近景 |

输出：`data/frames/<video_name>/` 按帧序号命名

- [ ] **Step 2: 运行抽帧，验证输出**

Run: `python3 tools/extract_frames.py`
Expected: `data/frames/` 下按视频名分目录，总计约 1600 帧

- [ ] **Step 3: Commit**

```bash
git add tools/extract_frames.py
git commit -m "data: add video frame extraction script"
```

---

## Task 4: 自动标注抽帧

**目的：** 用当前 best.pt 模型对抽取的帧进行推理，将 conf >= 0.60 的检测保存为 YOLO 格式标注，零检测帧保存为空标签文件（负样本）。

**改进目的：** Pseudo-labeling 策略——用高置信度阈值筛选"模型确信"的检测作为正样本标注，0.25-0.60 之间的低质量检测丢弃，零检测帧作为负样本。这是本项目**最主要的负样本来源**（预计400-800帧），对抑制误检至关重要。

**Files:**
- Create: `tools/auto_label.py`

- [ ] **Step 1: 编写自动标注脚本**

核心逻辑：
- 用 `runs/detect/runs/detect/frisbee_det_s/weights/best.pt` 推理
- conf >= 0.60 → 保存为 YOLO 标签文件（使用 `Boxes.xywhn` 获取归一化坐标，class 0）
- 无检测 → 创建空标签文件（负样本）
- 0.25 <= conf < 0.60 → 不保存（低质量标注）
- 输出到 `data/datasets/frisbee_pseudo/`
- **重要**：使用 ultralytics 的 `Boxes.xywhn` 属性获取归一化坐标（0-1），不要用 `Boxes.xywh`（返回像素坐标）再手动除以 orig_shape

- [ ] **Step 2: 运行自动标注**

Run: `python3 tools/auto_label.py`
Expected: `data/datasets/frisbee_pseudo/` 包含自动标注的图片和标签

- [ ] **Step 3: 统计标注质量**

检查：有标注帧数（正样本）vs 空标签帧数（负样本），bbox 大小分布，确认无异常。预期正样本约800-1200帧，负样本约400-800帧。

- [ ] **Step 4: Commit**

```bash
git add tools/auto_label.py
git commit -m "data: add auto-labeling script for pseudo-labeling"
```

---

## Task 5: 合并所有数据集

**目的：** 将4个数据源合并为统一的 YOLO 格式数据集，按 80/10/10 划分 train/val/test。

**改进目的：** 当前仅用 UltimateML 单一数据源（1001张），合并后预计 ~3000+ 张，数据量翻3倍且场景多样性大幅增加——这是提升模型泛化和精度的关键。负样本占比预计15-25%（来自 frisbee_negatives 80张 + pseudo 零检测帧 ~400-800张），对抑制误检有重要作用。

**Files:**
- Create: `tools/merge_datasets.py`
- Create: `configs/frisbee_merged.yaml`

- [ ] **Step 1: 编写合并脚本**

合并4个源：
1. `frisbee_ultimateml` (~1001张，全部正样本)
2. `frisbee_kaggle` (~851张，全部正样本)
3. `frisbee_negatives` (80张，**全部负样本**——空标签)
4. `frisbee_pseudo` (~1600张，正负混合)

为避免文件名冲突，每张图片加上源前缀。按 SEED=42 随机 80/10/10 划分。

- [ ] **Step 2: 运行合并，验证输出**

Run: `python3 tools/merge_datasets.py`
Expected: 约3000+张图片，`configs/frisbee_merged.yaml` 生成

- [ ] **Step 3: 用 verify_dataset.py 验证数据集完整性**

Run: `python3 tools/verify_dataset.py configs/frisbee_merged.yaml`
Expected: 无孤立图片、无孤立标签、无无效框

- [ ] **Step 4: Commit**

```bash
git add tools/merge_datasets.py configs/frisbee_merged.yaml
git commit -m "data: add dataset merging script and merged config"
```

---

## Task 6: YOLOv8s 精度优先训练

**目的：** 用合并后的数据集从 COCO 预训练权重重新训练 YOLOv8s。

**改进目的：** 当前模型仅用 UltimateML 单一数据集训练。用3倍以上的多样化数据重训是提升精度的核心手段。调整 box loss 权重和 close_mosaic 参数提升小目标定位。

**Files:**
- Modify: `models/train.py`

**训练参数变更：**

| 参数 | 当前值 | 新值 | 改进目的 |
|------|--------|------|----------|
| `--data` | `frisbee.yaml` | `configs/frisbee_merged.yaml` | 合并数据集 |
| `--resume` | UltimateML best.pt | 不使用 | 从COCO预训练开始 |
| `name` | `frisbee_det_s` | `frisbee_det_s_v2` | **保留旧模型，不覆盖** |
| `--epochs` | 50 | 100 | 训练更充分 |
| `--batch` | 2 | 8（OOM时降至4） | RTX 5080 16GB 可支持 |
| `box` | 7.5 | 10 | 增加定位损失权重 |
| `close_mosaic` | 10 | 15 | 更长关闭mosaic |
| `patience` | 20 | 30 | 更长早停耐心 |

- [ ] **Step 1: 更新 train.py 添加可配置参数**

在 argparse 中添加 `--box`（默认7.5）、`--close-mosaic`（默认10）、`--patience`（默认20）参数。
同时修改 `train_frisbee_detector` 函数，将 `name` 参数从硬编码 `frisbee_det_{model_size}` 改为可通过 `--name` 参数指定（默认 `frisbee_det_{model_size}`）。

- [ ] **Step 2: 创建 improve-precision 分支并启动训练**

```bash
python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --model-size s \
  --name frisbee_det_s_v2 \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --patience 30 \
  --device 0 \
  --box 10 \
  --close-mosaic 15
```

如果出现 OOM (Out of Memory)，降低 batch：
```bash
python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --model-size s \
  --name frisbee_det_s_v2 \
  --epochs 100 \
  --imgsz 1280 \
  --batch 4 \
  --patience 30 \
  --device 0 \
  --box 10 \
  --close-mosaic 15
```

预计训练时间：5-8小时

- [ ] **Step 3: 训练期间监控**

定期检查 `runs/detect/frisbee_det_s_v2/results.csv`，关注 mAP50 和 mAP50-95。

- [ ] **Step 4: Commit train.py changes**

```bash
git add models/train.py
git commit -m "feat: add configurable box loss, close_mosaic, patience, name params to train.py"
```

---

## Task 7: 评估与调优

**目的：** 在验证集和测试视频上评估新模型，找到精度优先的最佳 conf 阈值。

**改进目的：** 当前 conf=0.25 太低导致大量误检。找到最佳阈值，平衡误检和漏检。

- [ ] **Step 1: 在验证集上评估新模型**

Run: `python3 models/train.py --validate-only --model-path runs/detect/frisbee_det_s_v2/weights/best.pt --data configs/frisbee_merged.yaml`

- [ ] **Step 2: 在测试视频上对比不同 conf 阈值**

对55-56分钟片段用 conf=0.25, 0.35, 0.45, 0.55, 0.65 分别推理。目标：每帧平均检测数接近1.0（因只有1个飞盘）。

- [ ] **Step 3: 对比新旧模型**

在相同 conf 阈值下，比较新旧模型的：
1. 总检测帧数（帧级检测率）
2. 平均每帧检测数（目标：接近1.0）
3. 置信度分布（高置信度占比）
4. mAP50 和 mAP50-95

- [ ] **Step 4: 更新 predict_video.py 添加统计输出**

在 `predict_video.py` 中添加以下统计输出：
- 每帧检测数分布（0检、1检、2检、3+检的帧数和占比）
- 置信度统计（min、max、mean、median、<0.5占比、>=0.7占比）
- 帧级检测率（至少1个检测的帧占比）

- [ ] **Step 5: Commit**

```bash
git add inference/predict_video.py
git commit -m "feat: add detection statistics output to predict_video"
```

---

## Task 8: Git 合并与最终提交

**目的：** 将开发成果从 `improve-precision` 合并到 `dev`，再到 `main`。

- [ ] **Step 1: 确认所有代码已提交**

```bash
git status
git log --oneline improve-precision
```

- [ ] **Step 2: 合并 improve-precision → dev**

```bash
git checkout dev
git merge improve-precision --no-ff -m "feat: improve frisbee detection precision - merged datasets, retrained model"
git branch -d improve-precision
```

- [ ] **Step 3: 合并 dev → main**

```bash
git checkout main
git merge dev --no-ff -m "release: improved frisbee detection model with merged datasets"
```

- [ ] **Step 4: 推送到远程（如果需要）**

```bash
git push origin main
git push origin dev
```

---

## 执行顺序

| Task | 名称 | 预计耗时 | 依赖 |
|------|------|----------|------|
| 1 | 转换 kaggle_frisbee | 1分钟 | 无 |
| 2 | frisbee_dataset 转负样本 | 1分钟 | 无 |
| 3 | 视频抽帧 | 5分钟 | 无 |
| 4 | 自动标注 | 15-30分钟 | Task 3 |
| 5 | 合并数据集 | 2分钟 | Task 1, 2, 4 |
| 6 | 训练模型 | 5-8小时 | Task 5 |
| 7 | 评估调优 | 10分钟 | Task 6 |
| 8 | Git合并 | 2分钟 | Task 7 |

**Task 1、2、3 可并行执行。总预计时间约6-9小时（训练占5-8小时）。**