# SigLIP 零样本飞盘分类 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 用 SigLIP（Google）零样本分类器过滤 YOLO 检测裁剪图，区分飞盘和白帽子/人头/反光物体等假阳性，产出一个可重用的命令行工具和验证脚本。

**架构：** `tools/classify_frisbee.py`（分类器）+ `tools/validate_classifier.py`（验证对比脚本）。分类器加载 `google/siglip-so400m-patch14-384`，用 "a photo of a frisbee" prompt 做零样本分类，支持单图推理和目录批量处理，输出 CSV。验证脚本对比 SigLIP 结果与人工标注，输出混淆矩阵和准确率。

**技术栈：** PyTorch 2.11 + Transformers (HuggingFace) + SigLIP (google/siglip-so400m-patch14-384, ~3.5GB, 首次运行自动下载)

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `tools/classify_frisbee.py` | SigLIP 零样本分类器：模型加载、单图分类、目录批量处理、argparse 入口 |
| `tools/validate_classifier.py` | 验证脚本：对比分类器结果与人工标注 CSV，输出准确率/FP/FN 统计 |
| `tests/test_classify_frisbee.py` | 测试：模型前向传播形状、单图分类输出格式、目录批量 CSV 正确性 |

---

### 任务 1：注册 pytest 标记 + 创建测试文件

**文件：**
- 修改：`pytest.ini`
- 创建：`tests/test_classify_frisbee.py`

- [ ] **步骤 1：注册 `slow` 标记**

在 `pytest.ini` 的 `markers` 下追加 `slow`：

```ini
[pytest]
markers =
    integration: marks tests that require real data files
    slow: marks tests that require GPU and model download
```

- [ ] **步骤 2：编写失败的测试**

创建 `tests/test_classify_frisbee.py`：

```python
"""Tests for SigLIP zero-shot frisbee classifier."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pytest
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_model_forward_pass_output_shape():
    """SigLIP 模型前向传播输出形状为 (1, 1)."""
    from tools.classify_frisbee import load_model_and_processor
    from PIL import Image
    import torch

    model, processor = load_model_and_processor(device="cpu")
    dummy_img = Image.new("RGB", (384, 384), color=(128, 128, 128))
    inputs = processor(text=["a photo of a frisbee"], images=dummy_img,
                       padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits_per_image.shape == (1, 1)


def test_classify_image_returns_tuple():
    """单图分类返回 (is_frisbee: bool, probability: float)."""
    from tools.classify_frisbee import load_model_and_processor, classify_image
    from PIL import Image
    import tempfile

    model, processor = load_model_and_processor(device="cpu")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        dummy = Image.new("RGB", (100, 100), color=(0, 0, 0))
        dummy.save(f.name, format="JPEG")
        is_frisbee, prob = classify_image(f.name, model, processor, device="cpu")
    os.unlink(f.name)

    assert isinstance(is_frisbee, bool)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_classify_directory_writes_csv(tmp_path):
    """批量分类目录输出 CSV，包含所有文件."""
    from tools.classify_frisbee import load_model_and_processor, classify_directory
    from PIL import Image

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

- [ ] **步骤 3：运行测试验证失败**

```bash
python3 -m pytest tests/test_classify_frisbee.py::test_classify_image_returns_tuple -v
```
预期：`ModuleNotFoundError: No module named 'tools.classify_frisbee'`

- [ ] **步骤 4：编写最少实现**

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
    """Load SigLIP model and processor. First call downloads ~3.5GB."""
    from transformers import AutoModel, AutoProcessor
    model_name = "google/siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def classify_image(image_path, model, processor, device="cuda", threshold=0.5):
    """Classify a single crop. Returns (is_frisbee, probability)."""
    prompt = "a photo of a frisbee or flying disc"
    image = Image.open(str(image_path)).convert("RGB")

    inputs = processor(text=[prompt], images=image,
                       padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        prob = torch.sigmoid(logits)[0][0].item()

    return prob >= threshold, prob


def classify_directory(model, processor, img_dir, output_csv, device="cuda", threshold=0.5):
    """Classify all images in a directory, write results to CSV."""
    img_dir = Path(img_dir)
    results = []

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )

    for img_path in image_files:
        is_frisbee, prob = classify_image(str(img_path), model, processor, device=device, threshold=threshold)
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
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading SigLIP model (first time downloads ~3.5GB)...")
    model, processor = load_model_and_processor(device=device)

    crop_dir = Path(args.crop_dir)
    output = Path(args.output) if args.output else crop_dir / "siglip_results.csv"

    n = classify_directory(model, processor, str(crop_dir), str(output), device=device, threshold=args.threshold)
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

- [ ] **步骤 5：运行测试验证通过**

```bash
python3 -m pytest tests/test_classify_frisbee.py -v -m "not slow"
```
预期：2 PASSED, 1 deselected

- [ ] **步骤 6：运行慢速测试（首次需下载 ~3.5GB）**

```bash
python3 -m pytest tests/test_classify_frisbee.py::test_model_forward_pass_output_shape -v
```
预期：1 PASSED

- [ ] **步骤 7：Commit**

```bash
git add pytest.ini tests/test_classify_frisbee.py tools/classify_frisbee.py
git commit -m "feat: add SigLIP zero-shot frisbee classifier"
```

---

### 任务 2：验证对比脚本

**文件：**
- 创建：`tools/validate_classifier.py`

- [ ] **步骤 1：创建验证脚本**

创建 `tools/validate_classifier.py`：

```python
"""Compare classifier results against human labels, output confusion matrix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-csv", required=True, help="Human labels CSV (review_results.csv)")
    parser.add_argument("--classifier-csv", required=True, help="Classifier output CSV")
    parser.add_argument("--human-label-col", default="result", help="Column name for human label")
    parser.add_argument("--human-positive", default="TP", help="Positive label value in human CSV")
    parser.add_argument("--classifier-label-col", default="label", help="Column name for classifier label")
    parser.add_argument("--classifier-positive", default="frisbee", help="Positive label value in classifier CSV")
    args = parser.parse_args()

    human = {}
    with open(args.human_csv) as f:
        for row in csv.DictReader(f):
            human[row["filename"]] = row[args.human_label_col].strip().upper() == args.human_positive.upper()

    classifier = {}
    with open(args.classifier_csv) as f:
        for row in csv.DictReader(f):
            classifier[row["filename"]] = row[args.classifier_label_col] == args.classifier_positive

    tp = fp = fn = tn = 0
    for fname, is_positive in human.items():
        if fname not in classifier:
            continue
        pred = classifier[fname]
        if is_positive and pred:
            tp += 1
        elif not is_positive and pred:
            fp += 1
        elif is_positive and not pred:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1)

    print(f"=== Classification Report ===")
    print(f"Total:       {total}")
    print(f"TP (correct frisbee):     {tp}")
    print(f"FP (false frisbee):       {fp}")
    print(f"FN (missed frisbee):      {fn}")
    print(f"TN (correct non-frisbee): {tn}")
    print(f"Accuracy:    {accuracy:.1f}%")
    print(f"Precision:   {precision:.1f}%")
    print(f"Recall:      {recall:.1f}%")
    print(f"F1:          {f1:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：Commit**

```bash
git add tools/validate_classifier.py
git commit -m "feat: add classifier validation script"
```

---

### 任务 3：在 200 张标注裁剪图上验证

- [ ] **步骤 1：运行 SigLIP 分类器**

```bash
python3 tools/classify_frisbee.py --crop-dir data/perbox_crops --device cuda
```
预期：输出 `data/perbox_crops/siglip_results.csv`，显示 TP/FP 统计

- [ ] **步骤 2：对比 SigLIP 结果与人工标注**

```bash
python3 tools/validate_classifier.py \
  --human-csv data/perbox_crops/review_results.csv \
  --classifier-csv data/perbox_crops/siglip_results.csv
```
预期：输出混淆矩阵和 Precision/Recall/F1

- [ ] **步骤 3：决策门**

根据输出决定：

| SigLIP F1 | 结论 |
|:---------:|------|
| > 85% | ✅ 零样本方案有效，进入任务 4 |
| 70-85% | ⚠️ 尝试调 `--threshold`（0.3, 0.4, 0.6, 0.7）重新跑，选最优 F1 |
| < 70% | ❌ SigLIP 效果不足，报告失败 |

如果需调阈值，重新运行：

```bash
python3 tools/classify_frisbee.py --crop-dir data/perbox_crops --threshold 0.3 --device cuda
python3 tools/validate_classifier.py --human-csv data/perbox_crops/review_results.csv --classifier-csv data/perbox_crops/siglip_results.csv
```

- [ ] **步骤 4：记录结果**

```bash
git add -A
git commit -m "results: SigLIP validation on 200 labeled crops"
```

---

### 任务 4：过滤 v7_quick 全部检测裁剪图（仅任务 3 通过后执行）

- [ ] **步骤 1：分类 55-56min 裁剪图**

```bash
python3 tools/classify_frisbee.py --crop-dir data/fp_spotcheck_v7_55min --threshold <best_threshold> --device cuda
```

- [ ] **步骤 2：分类 20-23min 裁剪图**

```bash
python3 tools/classify_frisbee.py --crop-dir data/fp_spotcheck_v7_20min --threshold <best_threshold> --device cuda
```

- [ ] **步骤 3：汇总两个视频的过滤结果**

查看两个 `siglip_results.csv` 的 frisbee/not_frisbee 统计，估算 v7_quick 的真实 FP 率。

- [ ] **步骤 4：记录结果并决策**

```bash
git add -A
git commit -m "results: SigLIP filtering on v7_quick 4228 crops"
```

---

## 自检

**1. 规格覆盖度：**
- [x] SigLIP 模型加载 + 前向传播测试 → 任务 1 (`test_model_forward_pass_output_shape`)
- [x] 单图分类返回 `(bool, float)` → 任务 1 (`test_classify_image_returns_tuple`)
- [x] 目录批量分类输出 CSV → 任务 1 (`test_classify_directory_writes_csv`)
- [x] argparse 入口（`--crop-dir`, `--output`, `--threshold`, `--device`）→ 任务 1
- [x] 阈值可调 → 任务 1 `--threshold` 参数
- [x] 验证对比脚本（混淆矩阵 + Precision/Recall/F1）→ 任务 2
- [x] 在 200 张已知数据上验证准确率 → 任务 3
- [x] 阈值调优流程 → 任务 3 步骤 3
- [x] 过滤 v7_quick 全量裁剪图 → 任务 4

**2. 占位符扫描：** 无 TODO、无 "待定"。任务 4 中的 `<best_threshold>` 由任务 3 确定后填入，是唯一的动态值，已在步骤中注明。

**3. 类型一致性：**
- `load_model_and_processor` → 返回 `(model, processor)` → 所有函数签名接收 `(model, processor, ...)`
- `classify_image` → 返回 `(bool, float)` → `classify_directory` 解构为 `is_frisbee, prob`
- CSV 列名：`filename`, `label`, `confidence` → `validate_classifier.py` 使用 `--classifier-label-col` 和 `--classifier-positive` 参数匹配
- 标签值：`"frisbee"` / `"not_frisbee"` → 测试和实现中一致
