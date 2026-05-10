import os
import yaml
from pathlib import Path
from collections import Counter

def analyze_yolo_dataset(labels_dir, class_names):
    stats = {"class_counts": Counter(), "total_labels": 0, "total_images": 0,
             "bbox_areas": [], "images_per_class": Counter(), "empty_labels": 0}
    label_files = list(Path(labels_dir).glob("*.txt"))
    stats["total_images"] = len(label_files)
    for lf in label_files:
        lines = lf.read_text().strip().split("\n")
        has_content = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            cls = int(cls)
            area = w * h
            stats["class_counts"][cls] += 1
            stats["bbox_areas"].append((cls, area))
            stats["images_per_class"][cls] += 1
            has_content = True
            stats["total_labels"] += 1
        if not has_content:
            stats["empty_labels"] += 1
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total labels: {stats['total_labels']}")
    print(f"  Empty labels: {stats['empty_labels']}")
    print("\n  Class distribution:")
    for cls, count in sorted(stats["class_counts"].items()):
        name = class_names.get(cls, f"class_{cls}")
        print(f"    {name} (id={cls}): {count} instances in {stats['images_per_class'][cls]} images")
    print(f"\n  BBox size distribution (relative to image):")
    sizes = {"tiny (<0.1%)": 0, "small (0.1-1%)": 0, "medium (1-5%)": 0, "large (>5%)": 0}
    for _, a in stats["bbox_areas"]:
        if a < 0.001: sizes["tiny (<0.1%)"] += 1
        elif a < 0.01: sizes["small (0.1-1%)"] += 1
        elif a < 0.05: sizes["medium (1-5%)"] += 1
        else: sizes["large (>5%)"] += 1
    for label, count in sizes.items():
        print(f"    {label}: {count}")
    return stats

if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/firsbee/03_datasets/UltimateML/ultimate_train/Ultimate-Frisbee-Game-7"
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        print(f"ERROR: data.yaml not found at {data_yaml_path}")
        sys.exit(1)
    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    class_names = {i: n for i, n in enumerate(data_cfg["names"])}
    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {class_names}\n")
    for split in ["train", "valid", "test"]:
        labels_dir = None
        candidates = [
            os.path.join(dataset_dir, split, "labels"),
            os.path.join(dataset_dir, split, split, "labels"),
        ]
        if os.path.exists(dataset_dir):
            for sd in os.listdir(dataset_dir):
                candidate = os.path.join(dataset_dir, sd, split, "labels")
                if os.path.isdir(candidate):
                    candidates.append(candidate)
        for c in candidates:
            if os.path.exists(c):
                labels_dir = c
                break
        if labels_dir is None:
            print(f"\n=== {split} === SKIPPED (directory not found)")
            continue
        print(f"\n=== {split} ===")
        analyze_yolo_dataset(labels_dir, class_names)
