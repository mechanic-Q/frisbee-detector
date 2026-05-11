"""将 frisbee_dataset 全部转为负样本（删除所有标注，只保留空标签文件）。

原因: frisbee_dataset 的381个标注中149个（39%）面积>50%，属于全图标注错误。
无法可靠区分哪些小框是真正的飞盘标注，因此全部转为负样本最安全。

源: /mnt/e/firsbee/03_datasets/frisbee_dataset/
输出: /mnt/e/frisbee-detector/data/datasets/frisbee_negatives/
  - 80张图片 + 80个空标签文件（全部负样本）
  - frisbee.yaml
"""
import shutil
from pathlib import Path

SRC_IMG_DIR = Path("/mnt/e/firsbee/03_datasets/frisbee_dataset/images/train")
DST_DIR = Path("/mnt/e/frisbee-detector/data/datasets/frisbee_negatives")


def main():
    # 创建 train 目录（负样本全放 train，merge 时会重新划分）
    dst_img = DST_DIR / "images" / "train"
    dst_lbl = DST_DIR / "labels" / "train"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in sorted(SRC_IMG_DIR.glob("*.jpg")):
        shutil.copy2(img, dst_img / img.name)
        (dst_lbl / img.with_suffix(".txt").name).touch()  # 空标签 = 负样本
        count += 1

    yaml_content = f"path: {DST_DIR}\ntrain: images/train\nval: images/train\ntest: images/train\nnc: 1\nnames: ['frisbee']\n"
    with open(DST_DIR / "frisbee.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Converted {count} images to negative samples (empty labels)")
    print(f"Output: {DST_DIR}")


if __name__ == "__main__":
    main()
