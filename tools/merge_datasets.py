"""合并4个数据源为统一的 YOLO 格式数据集，按 80/10/10 划分 train/val/test。

源数据集:
1. frisbee_ultimateml  (~1001张, 全部正样本)
2. frisbee_kaggle      (~851张,  全部正样本)
3. frisbee_negatives   (80张,    全部负样本)
4. frisbee_pseudo      (~1264张, 正负混合)

输出: /mnt/e/frisbee-detector/data/datasets/frisbee_merged/
"""
import shutil
import random
from pathlib import Path

SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)

SOURCES = [
    {
        "name": "ultimateml",
        "path": "/mnt/e/frisbee-detector/data/datasets/frisbee_ultimateml",
        "splits": ["train", "valid", "test"],
    },
    {
        "name": "kaggle",
        "path": "/mnt/e/frisbee-detector/data/datasets/frisbee_kaggle",
        "splits": ["train", "val", "test"],
    },
    {
        "name": "negatives",
        "path": "/mnt/e/frisbee-detector/data/datasets/frisbee_negatives",
        "splits": ["train"],
    },
    {
        "name": "pseudo",
        "path": "/mnt/e/frisbee-detector/data/datasets/frisbee_pseudo",
        "splits": ["train"],
    },
]

DST_DIR = Path("/mnt/e/frisbee-detector/data/datasets/frisbee_merged")


def collect_all_files():
    all_files = []
    for source in SOURCES:
        src_path = Path(source["path"])
        if not src_path.exists():
            print(f"  SKIP: {src_path} not found")
            continue
        for split in source["splits"]:
            img_dir = src_path / "images" / split
            lbl_dir = src_path / "labels" / split
            if not img_dir.exists():
                continue

            for img_file in img_dir.rglob("*"):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                rel_to_img_dir = img_file.relative_to(img_dir)
                lbl_file = lbl_dir / rel_to_img_dir.with_suffix(".txt")
                all_files.append((img_file, lbl_file, source["name"]))
    return all_files


def main():
    all_files = collect_all_files()
    print(f"Total files collected: {len(all_files)}")

    if not all_files:
        print("No files found. Run previous steps first.")
        return

    random.seed(SEED)
    random.shuffle(all_files)
    n = len(all_files)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": all_files[:n_train],
        "val": all_files[n_train:n_train + n_val],
        "test": all_files[n_train + n_val:],
    }

    total_with_labels = 0
    total_negatives = 0

    for split_name, files in splits.items():
        img_dir = DST_DIR / "images" / split_name
        lbl_dir = DST_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        with_labels = 0
        negatives = 0

        for img_file, lbl_file, source_name in files:
            new_name = f"{source_name}_{img_file.stem}"
            new_img = img_dir / f"{new_name}{img_file.suffix}"
            new_lbl = lbl_dir / f"{new_name}.txt"

            shutil.copy2(str(img_file), str(new_img))
            if lbl_file.exists():
                shutil.copy2(str(lbl_file), str(new_lbl))
                # 判断是否为空标签（负样本）
                if new_lbl.stat().st_size == 0:
                    negatives += 1
                else:
                    with_labels += 1
            else:
                new_lbl.touch()
                negatives += 1

        total_with_labels += with_labels
        total_negatives += negatives
        print(f"  {split_name}: {len(files)} images ({with_labels} positive, {negatives} negative)")

    total = total_with_labels + total_negatives
    print(f"\nTotal merged: {total} images")
    print(f"  Positive: {total_with_labels} ({total_with_labels/total*100:.1f}%)")
    print(f"  Negative: {total_negatives} ({total_negatives/total*100:.1f}%)")

    yaml_content = (
        f"path: {DST_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['frisbee']\n"
    )
    config_path = Path("/mnt/e/frisbee-detector/configs/frisbee_merged.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(yaml_content)

    print(f"\nConfig: {config_path}")
    print(f"Output: {DST_DIR}")


if __name__ == "__main__":
    main()
