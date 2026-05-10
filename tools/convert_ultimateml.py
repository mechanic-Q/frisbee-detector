import os
import shutil
import yaml
import sys
from pathlib import Path

def find_split_images_dir(src_dir, split):
    candidates = [
        os.path.join(src_dir, split, "images"),
        os.path.join(src_dir, split, split, "images"),
    ]
    if os.path.exists(src_dir):
        for sd in os.listdir(src_dir):
            candidate = os.path.join(src_dir, sd, split, "images")
            if os.path.isdir(candidate):
                candidates.append(candidate)
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def find_split_labels_dir(src_dir, split):
    candidates = [
        os.path.join(src_dir, split, "labels"),
        os.path.join(src_dir, split, split, "labels"),
    ]
    if os.path.exists(src_dir):
        for sd in os.listdir(src_dir):
            candidate = os.path.join(src_dir, sd, split, "labels")
            if os.path.isdir(candidate):
                candidates.append(candidate)
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def convert_to_1class(src_dir, dst_dir, target_class_id=2):
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(dst_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, "labels", split), exist_ok=True)
    kept = {"train": 0, "valid": 0, "test": 0}
    total = {"train": 0, "valid": 0, "test": 0}
    for split in ["train", "valid", "test"]:
        src_img_dir = find_split_images_dir(src_dir, split)
        src_lbl_dir = find_split_labels_dir(src_dir, split)
        if src_img_dir is None or src_lbl_dir is None:
            print(f"  Skipping {split}: directory not found after checking all nesting patterns")
            continue
        dst_img_dir = os.path.join(dst_dir, "images", split)
        dst_lbl_dir = os.path.join(dst_dir, "labels", split)
        img_files = list(Path(src_img_dir).glob("*.jpg")) + list(Path(src_img_dir).glob("*.png"))
        total[split] = len(img_files)
        for img_file in img_files:
            lbl_file = Path(src_lbl_dir) / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue
            lines = lbl_file.read_text().strip().split("\n")
            frisbee_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id == target_class_id:
                    frisbee_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            if frisbee_lines:
                shutil.copy2(img_file, dst_img_dir)
                Path(os.path.join(dst_lbl_dir, img_file.stem + ".txt")).write_text("\n".join(frisbee_lines) + "\n")
                kept[split] += 1
        print(f"  {split}: {kept[split]}/{total[split]} images contain frisbee labels")
    dataset_yaml = {
        "path": os.path.abspath(dst_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": 1,
        "names": ["frisbee"],
    }
    yaml_path = os.path.join(dst_dir, "frisbee.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    print(f"\nDataset config saved to {yaml_path}")
    print(f"Total: {sum(kept.values())}/{sum(total.values())} images with frisbee labels")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/firsbee/03_datasets/UltimateML/ultimate_train/Ultimate-Frisbee-Game-7"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/mnt/e/frisbee-detector/data/datasets/frisbee_ultimateml"
    convert_to_1class(src, dst, target_class_id=2)
