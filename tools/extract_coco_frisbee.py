"""Download COCO 2017 frisbee subset via zip extraction (fast)."""

import json
import os
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import PROJECT_ROOT

COCO_CLASS_ID = 34
OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets" / "frisbee_coco"
TMP_DIR = Path("/tmp/coco_frisbee")
TRAIN_ZIP_URL = "http://images.cocodataset.org/zips/train2017.zip"
VAL_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"


def download_zips() -> tuple[Path, Path]:
    """Download COCO zip files if not cached. Returns (train_zip, val_zip)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    train_zip = TMP_DIR / "train2017.zip"
    val_zip = TMP_DIR / "val2017.zip"

    for url, path, name in [(TRAIN_ZIP_URL, train_zip, "train"), (VAL_ZIP_URL, val_zip, "val")]:
        if path.exists() and path.stat().st_size > 1000000:
            print(f"  {name} zip already downloaded ({path.stat().st_size // 1048576}MB)")
            continue
        print(f"  Downloading {name}2017.zip (this may take a few minutes)...")
        urllib.request.urlretrieve(url, str(path))

    return train_zip, val_zip


def get_frisbee_filenames(ann_file: Path) -> set[str]:
    """Get filenames of all images containing frisbee."""
    with open(ann_file) as f:
        data = json.load(f)

    frisbee_img_ids: set[int] = set()
    for ann in data["annotations"]:
        if ann["category_id"] == COCO_CLASS_ID:
            frisbee_img_ids.add(ann["image_id"])

    filenames: set[str] = set()
    for img in data["images"]:
        if img["id"] in frisbee_img_ids:
            filenames.add(img["file_name"])

    return filenames


def coco_labels(ann_file: Path) -> dict[str, str]:
    """Build YOLO label content per image. Returns {filename: label_text}."""
    with open(ann_file) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}

    labels: dict[str, list[str]] = {}
    for ann in data["annotations"]:
        if ann["category_id"] != COCO_CLASS_ID:
            continue
        img = img_map.get(ann["image_id"])
        if img is None:
            continue

        x, y, w, h = ann["bbox"]
        iw, ih = img["width"], img["height"]
        cx = (x + w / 2) / iw
        cy = (y + h / 2) / ih
        nw = w / iw
        nh = h / ih
        labels.setdefault(img["file_name"], []).append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return {fn: "\n".join(ls) + "\n" for fn, ls in labels.items()}


def extract_images(zip_path: Path, filenames: set[str], split: str, labels: dict[str, str]) -> int:
    """Extract only frisbee images from zip. Returns count."""
    img_dir = OUTPUT_DIR / "images" / split
    lbl_dir = OUTPUT_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Zip contains: train2017/000000000139.jpg
        prefix = "train2017/" if "train" in split else "val2017/"
        for fname in filenames:
            zip_name = f"{prefix}{fname}"
            try:
                zf.extract(zip_name, str(TMP_DIR))
                src = TMP_DIR / zip_name
                shutil.move(str(src), str(img_dir / fname))
                # Write label
                if fname in labels:
                    lbl_path = lbl_dir / fname.replace(".jpg", ".txt").replace(".png", ".txt")
                    lbl_path.write_text(labels[fname])
                count += 1
                if count % 200 == 0:
                    print(f"    Extracted {count}/{len(filenames)}...")
            except KeyError:
                pass  # file not in zip
    return count


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Annotation files (already extracted by convert_coco_frisbee.py)
    ann_dir = TMP_DIR / "annotations" / "annotations"
    train_ann = ann_dir / "instances_train2017.json"
    val_ann = ann_dir / "instances_val2017.json"

    if not train_ann.exists():
        print("ERROR: Run convert_coco_frisbee.py first to download annotations.")
        return

    # Find frisbee filenames
    print("Parsing annotations...")
    train_files = get_frisbee_filenames(train_ann)
    val_files = get_frisbee_filenames(val_ann)
    print(f"  Train frisbee images: {len(train_files)}")
    print(f"  Val frisbee images:   {len(val_files)}")

    # Download zips
    print("\nDownloading COCO zips...")
    train_zip, val_zip = download_zips()

    # Build labels
    print("Building YOLO labels...")
    train_labels = coco_labels(train_ann)
    val_labels = coco_labels(val_ann)

    # Extract only frisbee images
    print(f"\nExtracting {len(train_files)} train images...")
    train_count = extract_images(train_zip, train_files, "train", train_labels)
    print(f"  Extracted {train_count} train images with labels")

    print(f"\nExtracting {len(val_files)} val images...")
    val_count = extract_images(val_zip, val_files, "val", val_labels)
    print(f"  Extracted {val_count} val images with labels")

    # Write config (using existing utility)
    from utils.dataset import write_yaml_config
    write_yaml_config(
        output_path=OUTPUT_DIR / "frisbee.yaml",
        dataset_dir=str(OUTPUT_DIR),
    )

    total_bboxes = sum(len(l.split("\n")) for l in train_labels.values() if l.strip()) + \
                   sum(len(l.split("\n")) for l in val_labels.values() if l.strip())
    print(f"\nDone! {train_count + val_count} images, ~{total_bboxes} bboxes")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
