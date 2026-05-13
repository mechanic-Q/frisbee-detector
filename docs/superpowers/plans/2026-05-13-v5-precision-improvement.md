# v5 Precision Improvement — 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `models/train.py` 中添加 `--cls` 参数支持，用 `cls=1.3` 重新训练 v5 模型，将误检率从 ~60% 降至 <15%。

**架构：** 单文件代码变更 + 4 个运维步骤。先在 `train.py` 中添加 `--cls` argparse + 函数参数，然后在 tmux 中训练 v5，最后在验证集和两个测试视频上评估。

**技术栈：** Python 3.10, ultralytics (YOLOv8), PyTorch, SAHI, OpenCV

---

## 文件变更

| 文件 | 变更 | 职责 |
|------|------|------|
| `models/train.py` | 修改: 19-76 | 添加 `--cls` argparse 参数，传入 `train_frisbee_detector()` |

无新增文件。无测试文件（代码变更极简，属于训练脚本的参数传递）。

---

### 任务 1：添加 `--cls` 参数到 train.py

**文件：**
- 修改：`models/train.py:19-76`（`train_frisbee_detector` 函数签名 + argparse）

**分析：** 当前 `train_frisbee_detector()` 函数签名已有 `box` 参数但无 `cls`。argparse 也无 `--cls` 参数。需要同时添加两者并用 kwargs 方式传入 `model.train()`。

当前 `model.train()` 调用没有显式传 `cls`，因此 YOLO 使用默认值 0.5。v4 的 args.yaml 证实了这一点。

添加 `cls` 参数到函数签名，默认同 Ultralytics 默认值 0.5，以便 `--validate-only` 等非训练路径不受影响。

- [ ] **步骤 1：读取当前 train.py**

```bash
cat models/train.py
```

确认当前代码布局。

- [ ] **步骤 2：修改函数签名**

在 `train_frisbee_detector()` 的 `box: float = 7.5` 参数后添加 `cls: float = 0.5`：

```
box: float = 7.5,
cls: float = 0.5,
```

- [ ] **步骤 3：在 argparse 中添加 `--cls`**

在 `--box` 参数块后添加：

```python
parser.add_argument("--cls", type=float, default=0.5, help="Classification loss weight")
```

- [ ] **步骤 4：将 cls 传入函数调用**

在 `train_frisbee_detector()` 调用位置添加 `cls=args.cls`

- [ ] **步骤 5：将 cls 传入 model.train()**

`cls` 参数通过 kwargs 自动传入 `model.train()`，确保已有的调用链将其传入：

```python
results = model.train(
    ...
    box=box,
    cls=cls,
    ...
)
```

- [ ] **步骤 6：测试 `--help` 确认参数暴露**

```bash
python3 models/train.py --help
```

确认输出包含 `--cls`。

- [ ] **步骤 7：Commit**

```bash
git add models/train.py
git commit -m "feat: add --cls argument to train.py for classification loss weight"
```

---

### 任务 2：训练 v5 模型

**文件：**
- 无代码变更。纯运维操作。

**重要约束：** YOLO 训练 >10min，必须在 tmux 中运行。Bash 工具有 10 分钟超时，训练需 1.5-2.5h。

**注意双嵌套 bug：** `project="runs/detect"` 导致输出到 `runs/detect/runs/detect/frisbee_det_s_v5/`。训练结束后必须移出。

- [ ] **步骤 1：启动 tmux 训练会话**

```bash
tmux new-session -d -s train -c /mnt/e/frisbee-detector
tmux send-keys -t train "python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --model-size s \
  --name frisbee_det_s_v5 \
  --epochs 100 --imgsz 1280 --batch 2 \
  --patience 20 --box 5 --cls 1.3 --close-mosaic 10" Enter
```

确认已启动：

```bash
tmux ls
```

- [ ] **步骤 2：确认训练启动状态**

```bash
sleep 30 && tail -5 runs/detect/runs/detect/frisbee_det_s_v5/results.csv 2>/dev/null || echo "仍在初始化..."
```

- [ ] **步骤 3：修复双嵌套路径**

训练结束后（从 results.csv 确认 epoch 数不再增加或 tmux 会话退出）：

```bash
tmux capture-pane -t train -p | tail -20
```

修复路径：

```bash
mv runs/detect/runs/detect/frisbee_det_s_v5 runs/detect/frisbee_det_s_v5
```

- [ ] **步骤 4：确认模型文件存在**

```bash
ls -lh runs/detect/frisbee_det_s_v5/weights/best.pt
```

预期：~22MB 文件存在。

---

### 任务 3：验证集评估

**文件：**
- 无代码变更。纯运维操作。

- [ ] **步骤 1：运行验证**

```bash
python3 models/train.py --validate-only \
  --model-path runs/detect/frisbee_det_s_v5/weights/best.pt \
  --data configs/frisbee_merged.yaml
```

预期输出 mAP50, mAP50-95, Precision, Recall。

- [ ] **步骤 2：记录并与 v4 对比**

| 指标 | v4 | v5 | 变化 |
|------|:--:|:--:|:----:|
| mAP50 | 0.815 | ? | ? |
| Precision | 0.847 | ? | ? |
| Recall | 0.706 | ? | ? |

---

### 任务 4：测试视频评估

**文件：**
- 无代码变更。纯运维操作。

- [ ] **步骤 1：评估 55-56min 测试片段**

```bash
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v5/weights/best.pt \
  --video movie/25866279684-1-192_55-56min.mp4 \
  --conf 0.20
```

关注：帧检测率（目标 60-70%）、平均检测/帧（目标 1.0-1.3）、置信度分布。

- [ ] **步骤 2：评估 20-23min 测试片段**

```bash
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v5/weights/best.pt \
  --video movie/clip_20-23min.mp4 \
  --conf 0.20
```

同样关注三个指标。

- [ ] **步骤 3：FP 目测抽查**

各抽 50 帧（共 100 帧），目测每帧的检测框准确性。

使用以下方法输出中间帧的标注结果：

```bash
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_s_v5/weights/best.pt \
  --video movie/25866279684-1-192_55-56min.mp4 \
  --conf 0.20
# 从 runs/detect/eval/ 目录看保存的标注结果
```

---

### 任务 5：OpenImages 下载（如果 v5 未达标，启动此备份计划）

**文件：**
- 仅当 v5 FP 率 >15% 或 dets/frame >1.3 时才执行

- [ ] **步骤 1：收集 Flying disc ID**

```bash
cat /mnt/e/firsbee/03_datasets/openimages_frisbee/flying_disc_all_ids.txt \
    /mnt/e/firsbee/03_datasets/openimages_frisbee/disc_golf_all_ids.txt \
    /mnt/e/firsbee/03_datasets/openimages_frisbee/frisbee_games_all_ids.txt \
    | sort -u > /tmp/all_frisbee_ids.txt
wc -l /tmp/all_frisbee_ids.txt
```

预期：约 865 行（去重后）。

- [ ] **步骤 2：尝试 CVDF 镜像下载**

从 OpenImages v6 中 Flying disc 类的 ID 下载图片。使用 20 线程并发。

安装下载工具：

```bash
pip install --break-system-packages oidv6
```

或使用 wget/curl 按 ID 构造 URL：

```
URL_TEMPLATE="https://storage.googleapis.com/openimages/2018_04/train/train_%s_%02d.jpg"
```

如果 Google Storage 返回 403，尝试 CVDF 镜像：

```
URL_TEMPLATE="https://storage.cvdfoundation.org/openimages/2018_04/train/train_%s_%02d.jpg"
```

- [ ] **步骤 3：验证下载量并记录到 wiki**

```bash
ls /mnt/e/firsbee/03_datasets/openimages_frisbee/images/*.jpg | wc -l
```

抽检 20 张确认标注质量。将下载结果记录到 wiki：

```
新条目：raw/articles/openimages-download-result.md
更新：concepts/frisbee-recognition-project.md 中 OpenImages 小节
```

- [ ] **步骤 4：添加至合并数据集**

修改 `tools/merge_datasets.py` 的 SOURCES 列表：

```python
{"name": "openimages", "splits": ["train", "val"]},
```

重新运行合并：

```bash
python3 tools/merge_datasets.py
```

- [ ] **步骤 5：训练 v6（与 v5 相同参数）**

```bash
tmux new-session -d -s train -c /mnt/e/frisbee-detector
tmux send-keys -t train "python3 models/train.py \
  --data configs/frisbee_merged.yaml \
  --model-size s \
  --name frisbee_det_s_v6 \
  --epochs 100 --imgsz 1280 --batch 2 \
  --patience 20 --box 5 --cls 1.3 --close-mosaic 10" Enter
```

---
