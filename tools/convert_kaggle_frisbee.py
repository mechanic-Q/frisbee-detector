"""将 kaggle_frisbee 数据集从 YOLO v3 Darknet 格式（图片和标签在同目录）
转换为标准 YOLO 格式（images/ + labels/ 分目录），同时将2类标注合并为单类飞盘。

源: /mnt/e/firsbee/03_datasets/kaggle_frisbee/train/
  - jpg+txt 混合在同一目录
  - class 0 (475实例)和 class 1 (520实例)均为飞盘标注
  - 所有851张图都有标注（0张负样本）

输出: /mnt/e/frisbee-detector/data/datasets/frisbee_kaggle/
  - images/train/, images/val/, images/test/
  - labels/train/, labels/val/, labels/test/
  - frisbee.yaml
"""
import shutil
import random
from pathlib import Path

SRC_DIR = Path("/mnt/e/firsbee/03_datasets/kaggle_frisbee/train")
DST_DIR = Path("/mnt/e/frisbee-detector/data/datasets/frisbee_kaggle")
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SEED = 42


def convert_labels(src_path, dst_path):
    with open(src_path, "r") as f:
        lines = f.readlines()
    converted = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        parts[0] = "0"  # 无论原来0还是1，统一映射为 class 0
        converted.append(" ".join(parts))
    if converted:
        with open(dst_path, "w") as f:
            f.write("\n".join(converted) + "\n")


def main():
    random.seed(SEED)
    jpg_files = sorted(SRC_DIR.glob("*.jpg"))
    random.shuffle(jpg_files)
    n = len(jpg_files)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": jpg_files[:n_train],
        "val": jpg_files[n_train:n_train + n_val],
        "test": jpg_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        img_dir = DST_DIR / "images" / split_name
        lbl_dir = DST_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for jpg_path in files:
            txt_path = jpg_path.with_suffix(".txt")
            shutil.copy2(jpg_path, img_dir / jpg_path.name)
            if txt_path.exists():
                convert_labels(txt_path, lbl_dir / txt_path.name)
            else:
                (lbl_dir / txt_path.name).touch()

        print(f"  {split_name}: {len(files)} images")

    total_instances = 0
    for split_name in splits:
        lbl_dir = DST_DIR / "labels" / split_name
        for txt in lbl_dir.glob("*.txt"):
            with open(txt) as f:
                total_instances += sum(1 for line in f if line.strip())

    yaml_content = f"path: {DST_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: ['frisbee']\n"
    yaml_path = DST_DIR / "frisbee.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Total images: {n}")
    print(f"Total frisbee instances: {total_instances}")
    print(f"YAML: {yaml_path}")
    print(f"Output: {DST_DIR}")


if __name__ == "__main__":
    main()
