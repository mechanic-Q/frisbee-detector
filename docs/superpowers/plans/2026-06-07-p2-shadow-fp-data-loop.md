# P2 阴影漏检 + 场边误检数据闭环执行计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 用已建好的 annotation 任务层，对 `videoplayback_trimmed.mp4` 生成阴影候选和误检候选，人工复核后导出训练样本，合并入 `frisbee_merged` 数据集，重训 P2 模型并复测。

**架构：** 复用 `tools/generate_annotation_tasks.py` → `tools/review_tasks.py` → `tools/export_annotation_tasks.py` 三段式流程，新增轻量合并脚本将导出样本并入现有训练集，最后 tmux 训练 + 评估。

**技术栈：** Python、Ultralytics YOLO、OpenCV、Streamlit、tmux。

---

## 参考

- 标注任务层设计：`docs/superpowers/specs/2026-06-06-p2-annotation-task-layer-design.md`
- 标注任务层实现计划：`docs/superpowers/plans/2026-06-06-p2-annotation-task-layer.md`
- 项目配置：`configs/annotation/p2_shadow_fp_round1.yaml`
- 训练配置：`configs/frisbee_merged.yaml`
- AGENTS.md 训练约束（RTX 5080 16GB、tmux、double-nesting bug）

## 文件结构

### 新增文件

| 文件 | 职责 |
|------|------|
| `tools/merge_annotation_exports.py` | 将 annotation 导出目录合并到 `frisbee_merged/images/train` 和 `labels/train` |

### 已有文件（复用，不修改）

| 文件 | 角色 |
|------|------|
| `configs/annotation/p2_shadow_fp_round1.yaml` | 项目配置：模型、视频、eval_segments、输出路径 |
| `tools/generate_annotation_tasks.py` | 候选生成（bbox_review + frame_label） |
| `tools/review_tasks.py` | 通用 Streamlit reviewer |
| `tools/export_annotation_tasks.py` | 导出 YOLO 标签 + hard negative |
| `configs/frisbee_merged.yaml` | 训练数据集 YAML |
| `models/train.py` | YOLO 训练入口 |

### 不改文件

- 不修改 `tools/annotation_core.py`、`tools/review_web.py`、`tools/extract_detection_crops.py`
- 不修改 P2 模型结构
- 不提交 `data/`、`movie/`、`runs/`、`*.pt`

## 提交规则

- 每个任务完成后 commit
- 禁止 `git add .`
- 禁止提交 `data/`、`movie/`、`runs/`、`*.pt`、`.env`、`node_modules/`

---

### 任务 1：生成 bbox_review 候选（场边误检）

**文件：** 无新建，产出写入 `data/annotation/p2_shadow_fp_round1/`

- [ ] **步骤 1：dry-run 验证配置**

```bash
python3 tools/generate_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml --task-type bbox_review --dry-run
```

预期输出：
```
Project: p2_shadow_fp_round1
Task store: data/annotation/p2_shadow_fp_round1/tasks.jsonl
Exclude ranges: 1
```

- [ ] **步骤 2：生成 bbox_review 候选**

```bash
python3 tools/generate_annotation_tasks.py \
  --project configs/annotation/p2_shadow_fp_round1.yaml \
  --task-type bbox_review \
  --max-tasks 150 \
  --frame-stride 5
```

预期：输出 `Generated tasks: N`（N ≤ 150），任务写入 `tasks.jsonl`。

- [ ] **步骤 3：验证任务池内容**

```bash
wc -l data/annotation/p2_shadow_fp_round1/tasks.jsonl
head -1 data/annotation/p2_shadow_fp_round1/tasks.jsonl | python3 -m json.tool | head -15
```

预期：行数 > 0，每条 JSON 含 `task_type: "bbox_review"`、`sample_role: "hard_negative_candidate"`、`review_status: "pending"`、`crop_path` 非空。

- [ ] **步骤 4：确认无泄露**

```bash
python3 -c "
from tools.annotation_core import read_tasks, load_project_config
config = load_project_config('configs/annotation/p2_shadow_fp_round1.yaml')
tasks = read_tasks(config['outputs']['task_store'])
from tools.annotation_core import is_excluded_timestamp
leaking = [t.task_id for t in tasks if is_excluded_timestamp(t.source_video, t.timestamp_sec, config['exclude_ranges'])]
print(f'Leaking: {len(leaking)}')
assert len(leaking) == 0, f'Leaking tasks: {leaking}'
print('OK')
"
```

预期：`Leaking: 0` → `OK`

- [ ] **步骤 5：Commit**

```bash
# 只提交代码变更（如有），不提交 data/
git status --short
# 确认无 data/ 文件后
git add -A
git commit -m "feat(annotation): generate bbox_review candidates for p2_shadow_fp_round1"
```

---

### 任务 2：生成 frame_label 候选（阴影漏检）

**文件：** 无新建，产出写入 `data/annotation/p2_shadow_fp_round1/`

- [ ] **步骤 1：dry-run**

```bash
python3 tools/generate_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml --task-type frame_label --dry-run
```

预期：同任务 1 步骤 1。

- [ ] **步骤 2：生成 frame_label 候选**

```bash
python3 tools/generate_annotation_tasks.py \
  --project configs/annotation/p2_shadow_fp_round1.yaml \
  --task-type frame_label \
  --max-tasks 150 \
  --frame-stride 5 \
  --shadow-threshold 0.55
```

预期：`Generated tasks: N`（N ≤ 150），任务追加到 `tasks.jsonl`（不覆盖已有 bbox_review 任务）。

- [ ] **步骤 3：验证两类任务共存**

```bash
python3 -c "
from tools.annotation_core import read_tasks
tasks = read_tasks('data/annotation/p2_shadow_fp_round1/tasks.jsonl')
types = {}
for t in tasks:
    types[t.task_type] = types.get(t.task_type, 0) + 1
print(types)
"
```

预期：`{'bbox_review': N1, 'frame_label': N2}`，两者均 > 0。

- [ ] **步骤 4：确认无泄露**

运行任务 1 步骤 4 相同命令。

- [ ] **步骤 5：Commit**

```bash
git status --short
git add -A
git commit -m "feat(annotation): generate frame_label shadow candidates for p2_shadow_fp_round1"
```

---

### 任务 3：人工复核

**文件：** 只写 `data/annotation/p2_shadow_fp_round1/tasks.jsonl`（通过 Streamlit）

- [ ] **步骤 1：启动 reviewer**

```bash
streamlit run tools/review_tasks.py -- --project configs/annotation/p2_shadow_fp_round1.yaml
```

- [ ] **步骤 2：逐条复核**

对每个 pending 任务：
- `bbox_review` 任务：看 crop 和 frame context，判断是否飞盘
  - 飞盘 → `frisbee`
  - 不是飞盘 → `not_frisbee`
  - 不确定 → `uncertain`
- `frame_label` 任务：看 frame，判断是否有飞盘，如有则在 bbox 字段填坐标
  - 有飞盘 → `frisbee`（需额外填 bbox — 当前 reviewer 暂不支持 bbox 标注，先用 `frisbee` 标记帧，bbox 后续补）
  - 无飞盘 → `not_frisbee`
  - 不确定 → `uncertain`

- [ ] **步骤 3：验证复核进度**

```bash
python3 -c "
from tools.annotation_core import read_tasks
tasks = read_tasks('data/annotation/p2_shadow_fp_round1/tasks.jsonl')
statuses = {}
for t in tasks:
    s = t.review_status
    statuses[s] = statuses.get(s, 0) + 1
decisions = {}
for t in tasks:
    if t.reviewer_decision:
        decisions[t.reviewer_decision] = decisions.get(t.reviewer_decision, 0) + 1
print('Statuses:', statuses)
print('Decisions:', decisions)
"
```

预期：pending 减少，accepted/skipped/rejected 有值。

- [ ] **步骤 6：Commit**（复核结果在 data/ 中，不提交）

```bash
git status --short
# 确认无需要提交的代码变更
```

---

### 任务 4：导出已复核样本

**文件：** 无新建，产出写入 `data/annotation/p2_shadow_fp_round1/export/`

- [ ] **步骤 1：运行导出**

```bash
python3 tools/export_annotation_tasks.py --project configs/annotation/p2_shadow_fp_round1.yaml
```

预期：输出 JSON 报告，含 `positive_count` 和 `hard_negative_count`。

- [ ] **步骤 2：验证导出目录结构**

```bash
find data/annotation/p2_shadow_fp_round1/export/ -type f | head -20
cat data/annotation/p2_shadow_fp_round1/export/export_report.json
```

预期：
```
export/
  images/positive/*.jpg   ← frame_label + frisbee 的帧
  labels/positive/*.txt   ← YOLO 标签
  images/hard_negative/*.jpg  ← bbox_review + not_frisbee 的 crop
  labels/hard_negative/*.txt  ← 空文件
  export_report.json
```

- [ ] **步骤 3：抽查导出样本质量**

```bash
ls data/annotation/p2_shadow_fp_round1/export/images/positive/ | head -5
ls data/annotation/p2_shadow_fp_round1/export/images/hard_negative/ | head -5
# 抽一条正样本标签确认格式
cat data/annotation/p2_shadow_fp_round1/export/labels/positive/*.txt | head -3
```

预期：正样本标签格式 `0 cx cy w h`（归一化），hard negative 标签为空。

- [ ] **步骤 4：Commit**

```bash
git status --short
# 导出结果在 data/ 中，不提交
```

---

### 任务 5：合并导出样本到训练集

**文件：**
- 创建：`tools/merge_annotation_exports.py`

- [ ] **步骤 1：编写合并脚本**

```python
"""Merge annotation export samples into frisbee_merged training dataset.

Usage:
    python3 tools/merge_annotation_exports.py --export-dir data/annotation/p2_shadow_fp_round1/export
"""

import argparse
import shutil
import sys
from pathlib import Path
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
del _Path

from configs.paths import PROJECT_ROOT

DST_IMAGES = PROJECT_ROOT / "data" / "datasets" / "frisbee_merged" / "images" / "train"
DST_LABELS = PROJECT_ROOT / "data" / "datasets" / "frisbee_merged" / "labels" / "train"


def merge_export(export_dir: Path) -> dict:
    if not export_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    report = {"positive_copied": 0, "hard_negative_copied": 0}

    # Positive samples: frame_label + frisbee → YOLO labels
    pos_images = sorted((export_dir / "images" / "positive").glob("*.jpg"))
    pos_labels = sorted((export_dir / "labels" / "positive").glob("*.txt"))
    for img in pos_images:
        shutil.copy2(str(img), str(DST_IMAGES / img.name))
    for lbl in pos_labels:
        shutil.copy2(str(lbl), str(DST_LABELS / lbl.name))
    report["positive_copied"] = len(pos_images)

    # Hard negatives: bbox_review + not_frisbee → empty labels
    hn_images = sorted((export_dir / "images" / "hard_negative").glob("*.jpg"))
    hn_labels = sorted((export_dir / "labels" / "hard_negative").glob("*.txt"))
    for img in hn_images:
        shutil.copy2(str(img), str(DST_IMAGES / img.name))
    for lbl in hn_labels:
        shutil.copy2(str(lbl), str(DST_LABELS / lbl.name))
    report["hard_negative_copied"] = len(hn_images)

    total = report["positive_copied"] + report["hard_negative_copied"]
    print(f"Copied {report['positive_copied']} positive + {report['hard_negative_copied']} hard negative = {total} total")
    print(f"Destination: {DST_IMAGES}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge annotation exports into training dataset")
    parser.add_argument("--export-dir", required=True, help="Annotation export directory")
    args = parser.parse_args()

    DST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    merge_export(Path(args.export_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 2：运行合并**

```bash
python3 tools/merge_annotation_exports.py --export-dir data/annotation/p2_shadow_fp_round1/export
```

预期：输出 copied 计数，正样本 + hard negative 总数。

- [ ] **步骤 3：验证合并后数据集**

```bash
echo "Train images:" $(ls data/datasets/frisbee_merged/images/train/ | wc -l)
echo "Train labels:" $(ls data/datasets/frisbee_merged/labels/train/ | wc -l)
```

预期：images 和 labels 数量一致，比合并前增加。

- [ ] **步骤 4：Commit**

```bash
git add tools/merge_annotation_exports.py
git commit -m "feat(annotation): add merge_annotation_exports tool"
```

---

### 任务 6：重训模型

**文件：** 不修改代码，训练产物写入 `runs/detect/`

- [ ] **步骤 1：确认数据集就绪**

```bash
python3 -c "
from pathlib import Path
import yaml
config = yaml.safe_load(Path('configs/frisbee_merged.yaml').read_text())
train_dir = Path(config['path']) / config['train']
labels_dir = Path(config['path']) / 'labels' / 'train'
imgs = list(train_dir.glob('*.jpg'))
lbls = list(labels_dir.glob('*.txt'))
print(f'Images: {len(imgs)}, Labels: {len(lbls)}')
# Check for orphan labels (labels without images) and orphan images (images without labels)
img_stems = {p.stem for p in imgs}
lbl_stems = {p.stem for p in lbls}
orphan_labels = lbl_stems - img_stems
orphan_images = img_stems - lbl_stems
print(f'Orphan labels: {len(orphan_labels)}, Orphan images: {len(orphan_images)}')
if orphan_labels:
    print(f'  First 5 orphan label stems: {sorted(list(orphan_labels))[:5]}')
if orphan_images:
    print(f'  First 5 orphan image stems: {sorted(list(orphan_images))[:5]}')
"
```

预期：images 和 labels 数量一致，orphan 为 0。

- [ ] **步骤 2：在 tmux 中启动训练**

```bash
tmux new-session -d -s train-shadow -c /mnt/e/frisbee-detector
tmux send-keys -t train-shadow "python3 models/train.py --data configs/frisbee_merged.yaml --box 5 --cls 0.5 --name frisbee_det_p2_shadow_v1 --epochs 100 --batch 2 --workers 2" Enter
```

注意：`--batch 2 --workers 2`（RTX 5080 16GB 上限），`--cls 0.5`（保持 v3 的 cls 权重）。

- [ ] **步骤 3：监控训练**

```bash
# 查看训练进程
tmux capture-pane -t train-shadow -p | tail -20
```

训练完成后确认模型路径并修复 double-nesting bug：

```bash
# 如果出现 double-nesting
if [ -d runs/detect/runs/detect/frisbee_det_p2_shadow_v1 ]; then
    mv runs/detect/runs/detect/frisbee_det_p2_shadow_v1 runs/detect/frisbee_det_p2_shadow_v1
fi
ls -la runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt
```

- [ ] **步骤 4：Commit**（如有代码变更）

```bash
git status --short
# 训练产物不提交
```

---

### 任务 7：复测评估

**文件：** 不修改

- [ ] **步骤 1：first60s 评估片段检测**

```bash
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt \
  --video movie/videoplayback_first60s.mp4 \
  --conf 0.35
```

预期：输出检测统计（帧检测率、平均检测数/帧），与 P2 v3 基线对比。

- [ ] **步骤 2：独立片段 sanity check**

如有独立测试视频（非 first60s 的其他片段），同样跑一遍：

```bash
python3 inference/predict_video.py \
  --model runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt \
  --video <test_video_path> \
  --conf 0.35
```

- [ ] **步骤 3：与基线对比**

对比指标：
- 帧检测率（detection rate）：不应明显塌缩
- 平均检测数/帧：应下降（FP 减少）
- first60s 和独立片段趋势一致

- [ ] **步骤 4：更新 DEFAULT_MODEL（如效果优于当前）**

编辑 `configs/models.py`，更新 `DEFAULT_MODEL` 指向新模型。

- [ ] **步骤 5：Commit**

```bash
# 如有更新 configs/models.py
git add configs/models.py
git commit -m "feat(model): promote frisbee_det_p2_shadow_v1 as default"
```

---

## 最终验收

```bash
# 数据集完整性
python3 -c "
from pathlib import Path
import yaml
config = yaml.safe_load(Path('configs/frisbee_merged.yaml').read_text())
base = Path(config['path'])
for split in ['train', 'val', 'test']:
    imgs = set(p.stem for p in (base / 'images' / split).glob('*.jpg'))
    lbls = set(p.stem for p in (base / 'labels' / split).glob('*.txt'))
    orphan_labels = lbls - imgs
    orphan_images = imgs - lbls
    print(f'{split}: {len(imgs)} images, {len(lbls)} labels, orphan_labels={len(orphan_labels)}, orphan_images={len(orphan_images)}')
"

# 模型可用性
ls -la runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt

# 评估
python3 inference/predict_video.py --model runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt --video movie/videoplayback_first60s.mp4 --conf 0.35
```

验收标准：
- 训练集 images 和 labels 数量一致，无 orphan
- 模型文件存在
- first60s 帧检测率不塌缩，平均检测数/帧下降
- 未提交 `data/`、`movie/`、`runs/`、`*.pt`

## 实现后暂停点

完成评估后停下，向用户报告：
- 候选生成数量（bbox_review / frame_label 各多少）
- 人工复核结果分布（frisbee / not_frisbee / uncertain / skipped / rejected）
- 导出样本数量（正样本 / hard negative）
- 训练后评估指标对比基线
- 是否建议继续迭代（增加候选、调整阈值等）

不要自动启动下一轮数据闭环。


## 流程修正：2026-06-07

阴影漏检正样本不再使用“随机暗帧 + VLM 整帧判断”。该流程会产生大量连续帧，而且无 bbox 的 `frame_label` 无法导出 YOLO 正样本。

固定流程：

1. 低阈值 YOLO (`--candidate-conf 0.03`) 在暗帧中生成候选 bbox。
2. `--temporal-dedupe-frames 250` 去掉连续帧，只保留局部最高 `model_conf` 候选。
3. VLM 只判断红框内对象是否是飞盘，而不是判断整帧是否有飞盘。
4. `frame_label` 必须带 `bbox_xyxy` 和 `crop_path`，reviewer 页面用红框展示候选。
5. `bbox_review` 和 `frame_label` 在 UI 文案上分开：前者问“红框是不是模型误检/真飞盘”，后者问“红框是不是阴影飞盘正样本”。
