"""YOLO(YOLO格式) → COCO 格式转换（供 RF-DETR 训练使用）。
运行: python tools/yolo_to_coco.py
输出: data/datasets/frisbee_coco/{train,valid,test}/[images + _annotations.coco.json]
（图片用硬链接失败则复制）
"""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/datasets/frisbee_merged_v2"
DST = ROOT / "data/datasets/frisbee_coco"
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
CATEGORIES = [{"id": 0, "name": "frisbee", "supercategory": "none"}]


def convert_split(yolo_split: str, coco_split: str):
    img_dir = SRC / "images" / yolo_split
    lbl_dir = SRC / "labels" / yolo_split
    out_dir = DST / coco_split
    (out_dir).mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    for i, img_path in enumerate(sorted(img_dir.glob("*.*"))):
        lbl = lbl_dir / (img_path.stem + ".txt")
        # 用 PIL 读尺寸（比 cv2 快且不挑格式）
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
        file_name = img_path.name
        try:
            os.link(img_path, out_dir / file_name)
        except OSError:
            shutil.copy2(img_path, out_dir / file_name)
        images.append({"id": i, "file_name": file_name, "width": w, "height": h})
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                cx, cy, bw, bh = (float(x) for x in p[1:])
                bw *= w
                bh *= h
                x1 = cx * w - bw / 2
                y1 = cy * h - bh / 2
                annotations.append({
                    "id": ann_id, "image_id": i, "category_id": 0,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2), "iscrowd": 0,
                })
                ann_id += 1
    json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES},
              open(out_dir / "_annotations.coco.json", "w"))
    print(f"{yolo_split}->{coco_split}: {len(images)} imgs, {len(annotations)} anns")


if __name__ == "__main__":
    if DST.exists():
        shutil.rmtree(DST)
    for y, c in SPLIT_MAP.items():
        convert_split(y, c)
    print("done ->", DST)
