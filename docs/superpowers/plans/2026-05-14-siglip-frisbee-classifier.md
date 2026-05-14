# SigLIP 零样本飞盘分类 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 用 SigLIP（Google）零样本分类器过滤 YOLO 检测裁剪图，区分飞盘和白帽子/人头/反光物体等假阳性，产出一个可重用的命令行工具。

**架构：** `tools/classify_frisbee.py` — 单文件脚本，加载 `google/siglip-so400m-patch14-384`，用 "a photo of a frisbee" prompt 做零样本分类。支持单图推理和目录批量处理，输出 CSV。验证使用已标注的 200 张裁剪图。

**技术栈：** PyTorch 2.11 + Transformers（HuggingFace）+ SigLIP (Google SO400M-14-384, ~3.5GB)

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `tools/classify_frisbee.py` | SigLIP 零样本分类器：模型加载、单图分类、目录批量处理、argparse 入口 |
| `tests/test_classify_frisbee.py` | 测试：模型前向传播、输出格式、批量 CSV 正确性 |

---

### 任务 1：创建 SigLIP 分类器脚本

**文件：**
- 创建：`tools/classify_frisbee.py`
- 创建：`tests/test_classify_frisbee.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_classify_frisbee.py`：

```python
"""Tests for SigLIP zero-shot frisbee classifier."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pytest
import numpy as np


@pytest.mark.slow
def test_model_loading_output_shape():
    """SigLIP 模型加载后前向传播输出形状正确."""
    from tools.classify_frisbee import load_model_and_processor
    from PIL import Image

    model, processor = load_model_and_processor(device="cpu")

    dummy_img = Image.new("RGB", (384, 384), color=(128, 128, 128))
    inputs = processor(text=["a photo of a frisbee"], images=dummy_img,
                       padding="max_length", return_tensors="pt")

    import torch
    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits_per_image.shape == (1, 1)


def test_classify_image_returns_tuple():
    """单图分类返回 (is_frisbee: bool, probability: float)."""
    from tools.classify_frisbee import load_model_and_processor, classify_image

    model, processor = load_model_and_processor(device="cpu")
    from PIL import Image
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        dummy = Image.new("RGB", (100, 100), color=(0, 0, 0))
        dummy.save(f.name, format="JPEG")
        is_frisbee, prob = classify_image(f.name, model, processor, device="cpu")

    assert isinstance(is_frisbee, bool)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_classify_directory_writes_csv(tmp_path):
    """批量分类目录输出 CSV，包含所有文件."""
    from tools.classify_frisbee import load_model_and_processor, classify_directory
    from PIL import Image
    import csv

    model, processor = load_model_and_processor(device="cpu")

    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(str(tmp_path / f"crop_{i}_c0.50_vid.jpg"), format="JPEG")

    output_csv = str(tmp_path / "results.csv")
    classify_directory(model, processor, str(tmp_path), output_csv, device="cpu")

    with open(output_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        for row in rows:
            assert "filename" in row
            assert "label" in row
            assert "confidence" in row
            assert row["label"] in ("frisbee", "not_frisbee")
            conf = float(row["confidence"])
            assert 0.0 <= conf <= 1.0
```

- [ ] **步骤 2：运行测试验证失败**

```bash
python3 -m pytest tests/test_classify_frisbee.py::test_classify_image_returns_tuple -v
```
预期：`ModuleNotFoundError: No module named 'tools.classify_frisbee'`

- [ ] **步骤 3：编写最少实现**

创建 `tools/classify_frisbee.py`：

```python
"""SigLIP zero-shot frisbee classifier for YOLO detection crops."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import torch
from PIL import Image


def load_model_and_processor(device="cuda"):
    """Load SigLIP model and processor. First call downloads ~3.5GB weights."""
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    model.eval()
    return model, processor


def classify_image(image_path, model, processor, device="cuda"):
    """Classify a single crop. Returns (is_frisbee, probability)."""
    prompt = "a photo of a frisbee or flying disc"
    image = Image.open(str(image_path)).convert("RGB")

    inputs = processor(text=[prompt], images=image,
                       padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        prob = torch.sigmoid(logits)[0][0].item()

    return prob >= 0.5, prob


def classify_directory(model, processor, img_dir, output_csv, device="cuda"):
    """Classify all images in a directory, write results to CSV."""
    img_dir = Path(img_dir)
    results = []

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )

    for img_path in image_files:
        is_frisbee, prob = classify_image(str(img_path), model, processor, device=device)
        label = "frisbee" if is_frisbee else "not_frisbee"
        results.append((img_path.name, label, prob))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])
        for filename, label, prob in results:
            writer.writerow([filename, label, f"{prob:.4f}"])

    return len(results)


def main():
    parser = argparse.ArgumentParser(description="SigLIP zero-shot frisbee classifier")
    parser.add_argument("--crop-dir", required=True, help="Directory with crop images")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading SigLIP model (first time downloads ~3.5GB)...")
    model, processor = load_model_and_processor(device=device)

    crop_dir = Path(args.crop_dir)
    output = Path(args.output) if args.output else crop_dir / "siglip_results.csv"

    n = classify_directory(model, processor, str(crop_dir), str(output), device=device)
    print(f"Classified {n} images")
    print(f"Results saved to {output}")

    if output.exists():
        frisbee = 0
        not_frisbee = 0
        with open(output) as f:
            for row in csv.DictReader(f):
                if row["label"] == "frisbee":
                    frisbee += 1
                else:
                    not_frisbee += 1
        total = frisbee + not_frisbee
        if total > 0:
            print(f"frisbee={frisbee} not_frisbee={not_frisbee} "
                  f"detection_rate={frisbee/total*100:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：运行测试验证通过**

```bash
python3 -m pytest tests/test_classify_frisbee.py::test_classify_image_returns_tuple tests/test_classify_frisbee.py::test_classify_directory_writes_csv -v
```
预期：2 PASSED

- [ ] **步骤 5：Run all fast tests**

```bash
python3 -m pytest tests/test_classify_frisbee.py -v -m "not slow"
```
预期：2 PASSED, 1 deselected (`test_model_loading_output_shape` 被 `@pytest.mark.slow` 标记，需要 GPU 下载权重)

- [ ] **步骤 6：Commit**

```bash
git add tests/test_classify_frisbee.py tools/classify_frisbee.py
git commit -m "feat: add SigLIP zero-shot frisbee classifier"
```

---

### 任务 2：在 200 张标注裁剪图上验证

- [ ] **步骤 1：运行 SigLIP 分类器对 200 张已知裁剪图分类**

```bash
python3 tools/classify_frisbee.py --crop-dir data/perbox_crops --device cuda
```
预期：输出 `data/perbox_crops/siglip_results.csv`

- [ ] **步骤 2：对比 SigLIP 结果与人工标注，计算准确率**

运行以下 Python 脚本：

```python
"""Compare SigLIP results against human labels."""
import csv
from pathlib import Path

human_csv = Path("data/perbox_crops/review_results.csv")
siglip_csv = Path("data/perbox_crops/siglip_results.csv")

# Load human labels
human = {}
with open(human_csv) as f:
    for row in csv.DictReader(f):
        human[row["filename"]] = row["result"].strip().upper() == "TP"

# Load SigLIP labels
siglip = {}
with open(siglip_csv) as f:
    for row in csv.DictReader(f):
        siglip[row["filename"]] = row["label"] == "frisbee"

# Compare
correct = 0
total = 0
tp_correct = 0
fp_correct = 0
tp_total = 0
fp_total = 0

for fname in human:
    if fname not in siglip:
        continue
    total += 1
    if human[fname] == siglip[fname]:
        correct += 1
        if human[fname]:
            tp_correct += 1
    if human[fname]:
        tp_total += 1
    else:
        fp_total += 1

# SigLIP FP = human says not frisbee, SigLIP says frisbee
siglip_fp = sum(1 for f in human if not human[f] and siglip.get(f, False))
siglip_fn = sum(1 for f in human if human[f] and not siglip.get(f, True))

print(f"Total: {total}")
print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
print(f"TP accuracy: {tp_correct}/{tp_total} = {tp_correct/max(tp_total,1)*100:.1f}%")
print(f"SigLIP false positives: {siglip_fp}/{fp_total} = {siglip_fp/max(fp_total,1)*100:.1f}%")
print(f"SigLIP false negatives: {siglip_fn}/{tp_total} = {siglip_fn/max(tp_total,1)*100:.1f}%")
```

- [ ] **步骤 3：记录验证结果并决策**

| SigLIP 准确率 | 结论 |
|:------------:|------|
| > 90% | ✅ 零样本方案有效，直接用于过滤 YOLO FP |
| 75-90% | ⚠️ 效果不错，但需调 threshold 优化 F1 |
| < 75% | ❌ SigLIP 也不能区分飞盘 vs 特定 FP 类型 |

- [ ] **步骤 4：如果效果良好，运行 SigLIP 过滤 v7_quick 全部 4228 个检测裁剪图**

```bash
python3 tools/classify_frisbee.py --crop-dir data/fp_spotcheck_v7_55min --device cuda
python3 tools/classify_frisbee.py --crop-dir data/fp_spotcheck_v7_20min --device cuda
```

- [ ] **步骤 5：Commit 验证脚本**

```bash
git add -A
git commit -m "feat: add SigLIP validation script against 200 labeled crops"
```

---

## 自检

**1. 规格覆盖度：**
- [x] SigLIP 模型加载 + 前向传播测试 → 任务 1
- [x] 单图分类 (`classify_image`) 返回 `(bool, float)` → 任务 1
- [x] 目录批量分类 (`classify_directory`) 输出 CSV → 任务 1
- [x] argparse 入口 (`--crop-dir`, `--output`, `--device`) → 任务 1
- [x] 在 200 张已知数据上验证准确率 → 任务 2
- [x] 阈值调优（可选）→ 任务 2

**2. 占位符扫描：** 无 TODO、无 "待定"、无缺少的代码块。所有步骤都有完整代码。

**3. 类型一致性：**
- `classify_image` 返回 `(bool, float)` → `classify_directory` 使用 `is_frisbee` (bool) 写标签
- CSV 列名为 `filename`, `label`, `confidence` → 与 `classify_crops.py` 的 ResNet18 版本一致
- `load_model_and_processor` 返回 `(model, processor)` → 所有函数都接收 (model, processor)
