# v7 二分类器 + 硬负样本扩充 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 用200张已标注裁剪图（22 TP + 178 FP）训练ResNet18二分类器，自动分类v7_quick的4228个检测，获取真实FP率并收集新硬负样本用于v8训练。

**架构：** `tools/train_frisbee_classifier.py`（训练脚本）→ 产出模型 → `tools/classify_crops.py`（推理脚本）→ 产出各裁剪图的 TP/FP 标签。两者共享 `classifier_utils.py` 中的通用函数（数据加载、模型构建、图像变换）。

**技术栈：** PyTorch 2.11 + torchvision (ResNet18)、OpenCV（图像读取）、pytest

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `tests/test_classifier.py` | 测试：数据加载、模型构造、训练冒烟测试、单图分类、批量分类 |
| `tools/classifier_utils.py` | 共享模块：`load_labeled_dataset()`、`create_model()`、`get_transforms()` |
| `tools/train_frisbee_classifier.py` | 训练入口：加载数据 → 训练 → 验证 → 保存模型 |
| `tools/classify_crops.py` | 推理入口：加载模型 → 批量分类裁剪图目录 → 输出CSV |

---

### 任务 1：创建测试文件 + 数据加载测试

**文件：**
- 创建：`tests/test_classifier.py`
- 创建：`tools/classifier_utils.py`

- [ ] **步骤 1：编写失败的测试（数据加载）**

```python
"""Tests for frisbee binary classifier."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import pytest
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_CSV = PROJECT_ROOT / "data" / "perbox_crops" / "review_results.csv"
CROP_DIR = PROJECT_ROOT / "data" / "perbox_crops"


def test_load_labeled_dataset():
    """加载已标注数据集，返回 (路径, 标签) 列表，0=FP, 1=TP."""
    from tools.classifier_utils import load_labeled_dataset

    samples = load_labeled_dataset(REVIEW_CSV, CROP_DIR)

    assert len(samples) == 200
    paths, labels = zip(*samples)
    assert all(p.is_file() for p in paths)
    assert set(labels) == {0, 1}

    tp_count = sum(labels)
    fp_count = len(labels) - tp_count
    assert tp_count == 22
    assert fp_count == 178
```

```python
def test_train_val_split():
    """分层划分训练/验证集，保留类别比例."""
    from tools.classifier_utils import split_train_val

    import tempfile, csv

    # Create a minimal temporary dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Create 5 fake images and a CSV
        for i in range(5):
            import cv2
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp / f"crop_{i}.jpg"), img)

        csv_path = tmp / "review.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "result", "frame", "conf"])
            writer.writerow(["crop_0.jpg", "TP", "1", "0.5"])
            writer.writerow(["crop_1.jpg", "FP", "2", "0.4"])
            writer.writerow(["crop_2.jpg", "TP", "3", "0.6"])
            writer.writerow(["crop_3.jpg", "FP", "4", "0.3"])
            writer.writerow(["crop_4.jpg", "TP", "5", "0.7"])

        samples = [(tmp / row[0], 1 if row[1] == "TP" else 0) for row in [
            ("crop_0.jpg", "TP"), ("crop_1.jpg", "FP"), ("crop_2.jpg", "TP"),
            ("crop_3.jpg", "FP"), ("crop_4.jpg", "TP")
        ]]

        train, val = split_train_val(samples, val_ratio=0.4, seed=42)

        assert len(train) + len(val) == 5
        train_tp = sum(1 for _, l in train if l == 1)
        train_fp = sum(1 for _, l in train if l == 0)
        val_tp = sum(1 for _, l in val if l == 1)
        val_fp = sum(1 for _, l in val if l == 0)

        assert train_tp + val_tp == 3
        assert train_fp + val_fp == 2
        assert train_tp >= 1 and val_tp >= 1
        assert train_fp >= 1 and val_fp >= 1
```

- [ ] **步骤 2：运行测试验证失败**

```bash
python3 -m pytest tests/test_classifier.py::test_load_labeled_dataset -v
```
预期：`ModuleNotFoundError: No module named 'tools.classifier_utils'`

- [ ] **步骤 3：编写最小实现**

创建 `tools/classifier_utils.py`：

```python
"""Shared utilities for frisbee classifier."""
import csv
from pathlib import Path


def load_labeled_dataset(csv_path, img_dir):
    """Load labeled dataset from CSV and image directory.

    Returns:
        list of (Path, int): (image_path, label) where label is 0=FP, 1=TP.
    """
    samples = []
    img_dir = Path(img_dir)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            label = 1 if row["result"].strip().upper() == "TP" else 0
            img_path = img_dir / row["filename"]
            if img_path.is_file():
                samples.append((img_path, label))
    return samples


def split_train_val(samples, val_ratio=0.2, seed=42):
    """Stratified split into train and validation sets."""
    import random
    random.seed(seed)

    tp_samples = [(p, l) for p, l in samples if l == 1]
    fp_samples = [(p, l) for p, l in samples if l == 0]

    random.shuffle(tp_samples)
    random.shuffle(fp_samples)

    tp_split = int(len(tp_samples) * (1 - val_ratio))
    fp_split = int(len(fp_samples) * (1 - val_ratio))

    train = tp_samples[:tp_split] + fp_samples[:fp_split]
    val = tp_samples[tp_split:] + fp_samples[fp_split:]

    random.shuffle(train)
    random.shuffle(val)
    return train, val
```

- [ ] **步骤 4：运行测试验证通过**

```bash
python3 -m pytest tests/test_classifier.py::test_load_labeled_dataset tests/test_classifier.py::test_train_val_split -v
```
预期：2 PASSED

- [ ] **步骤 5：Commit**

```bash
git add tests/test_classifier.py tools/classifier_utils.py
git commit -m "feat: add labeled dataset loader for binary classifier"
```

---

### 任务 2：模型构造 + 前向传播测试

**文件：**
- 修改：`tests/test_classifier.py`（追加测试）
- 修改：`tools/classifier_utils.py`（追加函数）

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_classifier.py` 末尾追加：

```python
def test_create_model_output_shape():
    """模型输出形状正确：batch_size=4, num_classes=2."""
    from tools.classifier_utils import create_model
    import torch

    model = create_model(num_classes=2, pretrained=True)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 2)
```

```python
def test_get_transforms_train_inference():
    """训练transforms有数据增强，推理transforms没有."""
    from tools.classifier_utils import get_transforms

    train_transform = get_transforms(is_train=True)
    eval_transform = get_transforms(is_train=False)

    import torchvision.transforms as T
    # Verify train has RandomHorizontalFlip
    has_random = any(isinstance(t, T.RandomHorizontalFlip) for t in train_transform.transforms)
    assert has_random, "train transform should have augmentation"

    # Verify eval does not
    has_random_eval = any(isinstance(t, T.RandomHorizontalFlip) for t in eval_transform.transforms)
    assert not has_random_eval, "eval transform should not have augmentation"
```

- [ ] **步骤 2：运行测试验证失败**

```bash
python3 -m pytest tests/test_classifier.py::test_create_model_output_shape -v
```
预期：`AttributeError: module 'tools.classifier_utils' has no attribute 'create_model'`

- [ ] **步骤 3：编写最小实现**

在 `tools/classifier_utils.py` 末尾追加：

```python
def create_model(num_classes=2, pretrained=True):
    """Create a ResNet18 binary classifier."""
    import torch
    import torch.nn as nn
    import torchvision.models as models

    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_transforms(is_train=True):
    """Get image transforms for training or inference."""
    import torchvision.transforms as T

    if is_train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
```

- [ ] **步骤 4：运行测试验证通过**

```bash
python3 -m pytest tests/test_classifier.py::test_create_model_output_shape tests/test_classifier.py::test_get_transforms_train_inference -v
```
预期：2 PASSED

- [ ] **步骤 5：Commit**

```bash
git add tests/test_classifier.py tools/classifier_utils.py
git commit -m "feat: add ResNet18 model creation and image transforms"
```

---

### 任务 3：分类器推理函数测试

**文件：**
- 修改：`tests/test_classifier.py`（追加测试）
- 修改：`tools/classifier_utils.py`（追加函数）

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_classifier.py` 末尾追加：

```python
def test_classify_image_returns_label_and_confidence():
    """单图分类返回 (label, confidence) 元组."""
    from tools.classifier_utils import create_model, get_transforms, classify_image
    import torch

    model = create_model(num_classes=2, pretrained=True)
    model.eval()
    transform = get_transforms(is_train=False)

    # Create a dummy image
    import cv2
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(f.name, dummy)
        label, conf = classify_image(model, f.name, transform, device="cpu")

    assert label in (0, 1)
    assert 0.0 <= conf <= 1.0
    assert isinstance(conf, float)
```

```python
def test_classify_directory_writes_csv(tmp_path):
    """批量分类目录输出CSV，包含所有裁剪图."""
    from tools.classifier_utils import create_model, get_transforms, classify_directory
    import cv2

    model = create_model(num_classes=2, pretrained=True)
    model.eval()
    transform = get_transforms(is_train=False)

    # Create 3 fake crops
    for i in range(3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"crop_{i}_c0.50_vid.jpg"), img)

    output_csv = str(tmp_path / "results.csv")
    classify_directory(model, str(tmp_path), output_csv, transform, device="cpu")

    with open(output_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        assert "filename" in rows[0]
        assert "label" in rows[0]
        assert "confidence" in rows[0]
        assert rows[0]["label"] in ("0", "1")
```

- [ ] **步骤 2：运行测试验证失败**

```bash
python3 -m pytest tests/test_classifier.py::test_classify_image_returns_label_and_confidence -v
```
预期：`AttributeError: module 'tools.classifier_utils' has no attribute 'classify_image'`

- [ ] **步骤 3：编写最小实现**

在 `tools/classifier_utils.py` 末尾追加：

```python
def classify_image(model, image_path, transform, device="cuda"):
    """Classify a single image. Returns (label, confidence)."""
    import torch
    import cv2

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = probs.max(dim=1)

    return pred.item(), conf.item()


def classify_directory(model, img_dir, output_csv, transform, device="cuda"):
    """Classify all images in a directory, write results to CSV."""
    import csv
    from pathlib import Path

    img_dir = Path(img_dir)
    results = []

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )

    for img_path in image_files:
        label, conf = classify_image(model, str(img_path), transform, device=device)
        results.append((img_path.name, label, conf))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])
        for filename, label, conf in results:
            writer.writerow([filename, label, f"{conf:.4f}"])

    return len(results)
```

- [ ] **步骤 4：运行测试验证通过**

```bash
python3 -m pytest tests/test_classifier.py::test_classify_image_returns_label_and_confidence tests/test_classifier.py::test_classify_directory_writes_csv -v
```
预期：2 PASSED

- [ ] **步骤 5：Commit**

```bash
git add tests/test_classifier.py tools/classifier_utils.py
git commit -m "feat: add single-image and directory classifier inference"
```

---

### 任务 4：训练脚本（训练 + 验证循环）

**文件：**
- 创建：`tools/train_frisbee_classifier.py`
- 修改：`tests/test_classifier.py`（追加冒烟测试）

- [ ] **步骤 1：编写失败的冒烟测试**

在 `tests/test_classifier.py` 末尾追加：

```python
def test_training_can_overfit_small_batch():
    """训练能在10张图上过拟合（loss接近0，acc接近1.0）."""
    from tools.classifier_utils import create_model, get_transforms
    import torch
    import tempfile
    import cv2
    import csv

    model = create_model(num_classes=2, pretrained=True)

    # Create 10 labeled images in temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp / f"tp_{i}.jpg"), img)
        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp / f"fp_{i}.jpg"), img)

        csv_path = tmp / "review.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "result", "frame", "conf"])
            for i in range(5):
                writer.writerow([f"tp_{i}.jpg", "TP", str(i), "0.5"])
            for i in range(5):
                writer.writerow([f"fp_{i}.jpg", "FP", str(i+5), "0.5"])

        # Use the training function directly
        from tools.classifier_utils import load_labeled_dataset, split_train_val
        samples = load_labeled_dataset(csv_path, tmp)
        train_samples, val_samples = split_train_val(samples, val_ratio=0.2)

        # Train for 50 epochs on this tiny set
        from tools.train_frisbee_classifier import train_one_epoch, validate
        import torch.optim as optim
        import torch.nn as nn

        model = create_model(num_classes=2, pretrained=False)
        device = "cpu"
        model.to(device)

        transform = get_transforms(is_train=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(100):
            train_loss, train_acc = train_one_epoch(
                model, train_samples, transform, optimizer, criterion, device, batch_size=4
            )
            val_loss, val_acc = validate(
                model, val_samples, get_transforms(is_train=False), criterion, device, batch_size=4
            )

        # Should overfit
        assert train_acc > 0.9, f"train acc {train_acc:.2f} should be > 0.9"
```

- [ ] **步骤 2：运行测试验证失败**

```bash
python3 -m pytest tests/test_classifier.py::test_training_can_overfit_small_batch -v
```
预期：`ModuleNotFoundError: No module named 'tools.train_frisbee_classifier'`

- [ ] **步骤 3：编写实现**

创建 `tools/train_frisbee_classifier.py`：

```python
"""Train a frisbee binary classifier on manually-labeled detection crops."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from classifier_utils import (
    load_labeled_dataset, split_train_val, create_model, get_transforms
)


def train_one_epoch(model, samples, transform, optimizer, criterion, device, batch_size=16):
    """Train for one epoch. Uses oversampling: ensures batch has both classes."""
    model.train()
    tp_samples = [(p, l) for p, l in samples if l == 1]
    fp_samples = [(p, l) for p, l in samples if l == 0]

    total_loss = 0.0
    correct = 0
    total = 0

    # Oversample TP class to balance batches
    num_batches = max(len(tp_samples), len(fp_samples)) // (batch_size // 2)
    if num_batches == 0:
        num_batches = 1

    for _ in range(num_batches):
        # Pick half batch from each class
        batch_tp = random.choices(tp_samples, k=batch_size // 2)
        batch_fp = random.choices(fp_samples, k=batch_size // 2)
        batch = batch_tp + batch_fp
        random.shuffle(batch)

        images = []
        labels = []
        for img_path, label in batch:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img)
            images.append(tensor)
            labels.append(label)

        x = torch.stack(images).to(device)
        y = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = output.max(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / num_batches, correct / total


def validate(model, samples, transform, criterion, device, batch_size=16):
    """Validate on a set of samples."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        images = []
        labels = []
        for img_path, label in batch:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img)
            images.append(tensor)
            labels.append(label)

        if not images:
            continue

        x = torch.stack(images).to(device)
        y = torch.tensor(labels, dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(x)
            loss = criterion(output, y)
            _, preds = output.max(dim=1)

        total_loss += loss.item()
        correct += (preds == y).sum().item()
        total += y.size(0)
        num_batches += 1

    return total_loss / max(num_batches, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/perbox_crops/review_results.csv")
    parser.add_argument("--img-dir", default="data/perbox_crops")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--name", default="frisbee_classifier_v1")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    samples = load_labeled_dataset(csv_path, img_dir)
    print(f"Loaded {len(samples)} samples ({sum(1 for _,l in samples if l==1)} TP, {sum(1 for _,l in samples if l==0)} FP)")

    train_samples, val_samples = split_train_val(samples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train: {len(train_samples)} ({sum(1 for _,l in train_samples if l==1)} TP, {sum(1 for _,l in train_samples if l==0)} FP)")
    print(f"Val:   {len(val_samples)} ({sum(1 for _,l in val_samples if l==1)} TP, {sum(1 for _,l in val_samples if l==0)} FP)")

    # Create model
    model = create_model(num_classes=2, pretrained=True)
    model.to(device)

    # Class weights for imbalance
    tp_count = sum(1 for _, l in train_samples if l == 1)
    fp_count = sum(1 for _, l in train_samples if l == 0)
    tp_weight = fp_count / max(tp_count, 1)
    class_weights = torch.tensor([1.0, tp_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Separate learning rates
    optimizer = optim.Adam([
        {"params": model.fc.parameters(), "lr": args.lr * 10},
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": args.lr},
    ])

    transform_train = get_transforms(is_train=True)
    transform_val = get_transforms(is_train=False)

    best_val_acc = 0.0
    save_path = Path("runs/classify") / args.name
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_samples, transform_train, optimizer, criterion, device, args.batch_size
        )
        val_loss, val_acc = validate(
            model, val_samples, transform_val, criterion, device, args.batch_size
        )

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, save_path / f"{args.name}.pt")
            print(f"  => saved (val_acc={val_acc:.4f})")

    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Model saved to {save_path / args.name}.pt")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：运行冒烟测试验证**

```bash
python3 -m pytest tests/test_classifier.py::test_training_can_overfit_small_batch -v
```
预期：PASS（可能有一定波动，acc需 > 0.9）

- [ ] **步骤 5：Run all tests to confirm nothing broken**

```bash
python3 -m pytest tests/test_classifier.py -v
```
预期：全部 PASS

- [ ] **步骤 6：Commit**

```bash
git add tools/train_frisbee_classifier.py tests/test_classifier.py
git commit -m "feat: add classifier training script with oversampling"
```

---

### 任务 5：推理脚本（批量分类 v7_quick 裁剪图）

**文件：**
- 创建：`tools/classify_crops.py`

- [ ] **步骤 1：创建推理入口脚本**

`tools/classify_crops.py` 无需单独测试（`classify_directory` 已在任务3中测试）。直接编写：

```python
"""Classify detection crops as TP/FP using trained binary classifier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
from classifier_utils import create_model, get_transforms, classify_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt model file")
    parser.add_argument("--crop-dir", required=True, help="Directory with crop images")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <crop-dir>/classify_results.csv)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", "unknown")
    print(f"Model loaded (val_acc={val_acc})")

    transform = get_transforms(is_train=False)
    crop_dir = Path(args.crop_dir)
    output = Path(args.output) if args.output else crop_dir / "classify_results.csv"

    n = classify_directory(model, str(crop_dir), str(output), transform, device=device)
    print(f"Classified {n} images")
    print(f"Results saved to {output}")

    # Print summary
    if output.exists():
        import csv
        tp = fp = 0
        with open(output) as f:
            for row in csv.DictReader(f):
                if row["label"] == "1":
                    tp += 1
                else:
                    fp += 1
        total = tp + fp
        print(f"TP={tp} FP={fp} FP_rate={fp/total*100:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：Commit**

```bash
git add tools/classify_crops.py
git commit -m "feat: add batch crop classifier script"
```

---

### 任务 6：运行实际训练 + 分类

- [ ] **步骤 1：训练分类器**

```bash
python3 tools/train_frisbee_classifier.py \
  --csv data/perbox_crops/review_results.csv \
  --img-dir data/perbox_crops \
  --epochs 50 --batch-size 16 --lr 1e-4 \
  --name frisbee_classifier_v1
```
预期：训练完成，val_acc > 0.75

- [ ] **步骤 2：分类 55-56min 裁剪图**

```bash
python3 tools/classify_crops.py \
  --model runs/classify/frisbee_classifier_v1/frisbee_classifier_v1.pt \
  --crop-dir data/fp_spotcheck_v7_55min
```
预期：输出 `data/fp_spotcheck_v7_55min/classify_results.csv`，显示 TP/FP 统计

- [ ] **步骤 3：分类 20-23min 裁剪图**

```bash
python3 tools/classify_crops.py \
  --model runs/classify/frisbee_classifier_v1/frisbee_classifier_v1.pt \
  --crop-dir data/fp_spotcheck_v7_20min
```
预期：输出 `data/fp_spotcheck_v7_20min/classify_results.csv`

- [ ] **步骤 4：汇总两个视频的结果**

```bash
python3 -c "
import csv
from pathlib import Path

for name in ['55min', '20min']:
    csv_path = Path(f'data/fp_spotcheck_v7_{name}/classify_results.csv')
    if csv_path.exists():
        tp = fp = 0
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row['label'] == '1': tp += 1
                else: fp += 1
        total = tp + fp
        print(f'v7_{name}: TP={tp} FP={fp} FP_rate={fp/total*100:.1f}% (n={total})')
"
```
预期：显示每个视频的FP率

- [ ] **步骤 5：Commit 训练结果（仅代码和元数据，不含模型）**

```bash
git add runs/classify/frisbee_classifier_v1/  # 如果有的话
git commit -m "results: classifier v1 training and crop classification" --allow-empty
```

---

### 任务 7：分析结果 + 决策

- [ ] **步骤 1：获取准确的FP率**

运行任务6的汇总命令，得到每个视频的TP/FP统计。

- [ ] **步骤 2：决策门**

根据FP率决定：
- **FP率 < 20%** → 直接进入Phase 2生产训练（imgsz=1280, 100 epochs）
- **FP率 ≥ 20%** → 将分类器标为FP的裁剪图复制为硬负样本，编写 `tools/expand_hard_negatives.py`，重新训练v8

---

## 自检

**1. 规格覆盖度：**
- [x] 数据加载：`load_labeled_dataset`, `split_train_val` → 任务1
- [x] 模型：`create_model` (ResNet18) → 任务2
- [x] 图像变换：`get_transforms` → 任务2
- [x] 单图分类：`classify_image` → 任务3
- [x] 批量分类：`classify_directory` → 任务3
- [x] 训练循环：`train_one_epoch`, `validate`, `main` → 任务4
- [x] 推理脚本：`classify_crops.py` → 任务5
- [x] 实际运行 + 决策 → 任务6, 7

**2. 占位符扫描：** 无 TODO/待定/补充细节。所有代码步骤都有完整实现。

**3. 类型一致性：**
- `samples` 类型：`list of (Path, int)` 贯穿全部函数
- `transform` 类型：`torchvision.transforms.Compose` 贯穿全部函数
- `model` 类型：`nn.Module` 贯穿全部函数
- 标签约定：`0=FP, 1=TP` 一致
