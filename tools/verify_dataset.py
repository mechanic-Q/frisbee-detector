"""Verify YOLO dataset integrity: orphan files, invalid annotations, class IDs."""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def verify_yolo_dataset(yaml_path: str) -> int:
    """Verify a YOLO dataset. Returns total number of issues found."""
    print(f"Verifying dataset: {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    base = cfg.get("path", os.path.dirname(os.path.abspath(yaml_path)))
    nc = cfg.get("nc", 999)
    names = cfg.get("names", [])
    print(f"  Base path: {base}")
    print(f"  Classes: {nc} ({names})")

    total_issues = 0
    for split in ["train", "val", "test"]:
        img_rel = cfg.get(split, "")
        if not img_rel:
            print(f"\n  {split}: not configured, skipping")
            continue

        lbl_rel = img_rel.replace("images", "labels", 1)
        img_dir = os.path.join(base, img_rel)
        lbl_dir = os.path.join(base, lbl_rel)

        if not os.path.exists(img_dir):
            print(f"\n  {split}: directory not found: {img_dir}")
            total_issues += 1
            continue

        img_files: set[Path] = set()
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_files.update(Path(img_dir).glob(ext))
        lbl_files: set[Path] = set(Path(lbl_dir).glob("*.txt")) if os.path.exists(lbl_dir) else set()

        # Check for orphan images (no matching label file)
        orphan_imgs = [f for f in img_files if (Path(lbl_dir) / (f.stem + ".txt")) not in lbl_files]

        # Check for orphan labels (no matching image file across all extensions)
        orphan_lbls = []
        for f in lbl_files:
            found = False
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                if (Path(img_dir) / (f.stem + ext)) in img_files:
                    found = True
                    break
            if not found:
                orphan_lbls.append(f)

        invalid = 0
        for lf in lbl_files:
            for line in lf.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid += 1
                    continue
                try:
                    cls_id, cx, cy, w, h = map(float, parts)
                except ValueError:
                    invalid += 1
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid += 1
                if int(cls_id) >= nc:
                    invalid += 1

        issues = len(orphan_imgs) + len(orphan_lbls) + invalid
        total_issues += issues
        print(f"\n  {split}: {len(img_files)} images, {len(lbl_files)} labels")
        print(f"    Orphan images: {len(orphan_imgs)}")
        print(f"    Orphan labels: {len(orphan_lbls)}")
        print(f"    Invalid boxes/class IDs: {invalid}")
        print(f"    Status: {'OK' if issues == 0 else 'ISSUES FOUND'}")

    print(f"\nTotal issues: {total_issues}")
    return total_issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify YOLO dataset integrity")
    parser.add_argument("yaml", nargs="?", default="data/datasets/frisbee_ultimateml/frisbee.yaml", help="Dataset YAML path")
    args = parser.parse_args()
    verify_yolo_dataset(args.yaml)
