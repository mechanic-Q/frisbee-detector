import os
import yaml
from pathlib import Path

def verify_yolo_dataset(yaml_path):
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
        img_files = set()
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            img_files.update(Path(img_dir).glob(ext))
        lbl_files = set(Path(lbl_dir).glob("*.txt")) if os.path.exists(lbl_dir) else set()
        orphan_imgs = [f for f in img_files if (Path(lbl_dir) / (f.stem + ".txt")) not in lbl_files]
        orphan_lbls = [f for f in lbl_files
                       if (Path(img_dir) / (f.stem + ".jpg")) not in img_files
                       and (Path(img_dir) / (f.stem + ".png")) not in img_files]
        invalid = 0
        for lf in lbl_files:
            for line in lf.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid += 1
                    continue
                try:
                    cls, cx, cy, w, h = map(float, parts)
                except ValueError:
                    invalid += 1
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid += 1
                if int(cls) >= nc:
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
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/datasets/frisbee_ultimateml/frisbee.yaml"
    verify_yolo_dataset(path)
