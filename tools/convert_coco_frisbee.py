"""Download COCO 2017 frisbee images and convert to YOLO format.

Only downloads images containing frisbee (class 29), not the full dataset.
"""

import json
import os
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

# Workaround for SSL verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Bootstrap: project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import PROJECT_ROOT

COCO_CLASS_ID = 34  # frisbee (category_id in COCO JSON)
ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"  # also used for val2017
OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets" / "frisbee_coco"
TMP_DIR = Path("/tmp/coco_frisbee")


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress. Returns True on success."""
    print(f"  Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def extract_frisbee_ids(annotation_path: Path) -> set[int]:
    """Parse COCO JSON and return set of image_ids containing frisbee."""
    with open(annotation_path) as f:
        data = json.load(f)

    # Build a mapping: image_id -> list of frisbee annotations
    frisbee_image_ids: set[int] = set()
    for ann in data["annotations"]:
        if ann["category_id"] == COCO_CLASS_ID:
            frisbee_image_ids.add(ann["image_id"])

    # Get filenames for those image_ids
    img_id_to_file: dict[int, str] = {}
    for img in data["images"]:
        if img["id"] in frisbee_image_ids:
            img_id_to_file[img["id"]] = img["file_name"]

    print(f"  Found {len(frisbee_image_ids)} images with frisbee annotations")
    return img_id_to_file


def coco_to_yolo(annotation_path: Path, img_id_to_file: dict[int, str]) -> dict[str, str]:
    """Build label file content for each image. Returns {filename: label_content}."""
    with open(annotation_path) as f:
        data = json.load(f)

    labels: dict[str, list[str]] = {fname: [] for fname in img_id_to_file.values()}

    for ann in data["annotations"]:
        if ann["category_id"] != COCO_CLASS_ID:
            continue
        img_info = None
        for img in data["images"]:
            if img["id"] == ann["image_id"]:
                img_info = img
                break
        if img_info is None or img_info["file_name"] not in labels:
            continue

        # COCO bbox: [x, y, width, height] (absolute pixels)
        x, y, w, h = ann["bbox"]
        img_w, img_h = img_info["width"], img_info["height"]

        # Convert to YOLO: [class, cx_center, cy_center, w_norm, h_norm]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        labels[img_info["file_name"]].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    result: dict[str, str] = {}
    for fname, lines in labels.items():
        result[fname] = "\n".join(lines) + "\n" if lines else ""
    return result


def download_images(img_files: list[str], split: str) -> list[str]:
    """Download specific COCO images. Returns list of successful downloads."""
    img_dir = OUTPUT_DIR / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []

    # COCO images are in train2017/ folder on the server for both splits
    base = "http://images.cocodataset.org/train2017"

    for i, fname in enumerate(img_files):
        dest = img_dir / fname
        if dest.exists() and dest.stat().st_size > 1000:
            downloaded.append(fname)
            continue

        url = f"{base}/{fname}"
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Downloading image {i+1}/{len(img_files)}: {fname}")
        try:
            urllib.request.urlretrieve(url, str(dest))
            downloaded.append(fname)
        except Exception as e:
            print(f"  ERROR downloading {fname}: {e}")

    return downloaded


def main() -> None:
    """Download and convert COCO frisbee subset."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download annotations
    ann_zip = TMP_DIR / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        if not download_file(ANNOTATIONS_URL, ann_zip):
            print("Failed to download annotations. Exiting.")
            return
    else:
        print("  Annotations already downloaded.")

    # Step 2: Extract annotations
    ann_dir = TMP_DIR / "annotations"
    if not ann_dir.exists():
        print("  Extracting annotations...")
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extractall(ann_dir)
    else:
        print("  Annotations already extracted.")

    # Step 3: Parse and filter for frisbee
    print("\nParsing COCO annotations for frisbee images...")
    train_ann = ann_dir / "annotations" / "instances_train2017.json"
    val_ann = ann_dir / "annotations" / "instances_val2017.json"

    train_ids = extract_frisbee_ids(train_ann)
    val_ids = extract_frisbee_ids(val_ann)

    # Step 4: Download only frisbee images
    print(f"\nDownloading {len(train_ids)} training images...")
    train_downloaded = download_images(list(train_ids.values()), "train")
    print(f"  Downloaded {len(train_downloaded)} train images")

    print(f"\nDownloading {len(val_ids)} validation images...")
    val_downloaded = download_images(list(val_ids.values()), "val")
    print(f"  Downloaded {len(val_downloaded)} val images")

    # Step 5: Convert labels to YOLO format
    print("\nConverting labels to YOLO format...")
    train_labels = coco_to_yolo(train_ann, train_ids)
    val_labels = coco_to_yolo(val_ann, val_ids)

    train_lbl_dir = OUTPUT_DIR / "labels" / "train"
    val_lbl_dir = OUTPUT_DIR / "labels" / "val"
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    for fname, content in train_labels.items():
        if fname in train_downloaded:
            lbl_path = train_lbl_dir / fname.replace(".jpg", ".txt")
            lbl_path.write_text(content)
            if content.strip():
                train_count += 1

    val_count = 0
    for fname, content in val_labels.items():
        if fname in val_downloaded:
            lbl_path = val_lbl_dir / fname.replace(".jpg", ".txt")
            lbl_path.write_text(content)
            if content.strip():
                val_count += 1

    print(f"  Train: {train_count} labeled images")
    print(f"  Val: {val_count} labeled images")

    # Step 6: Create dataset config (using image counts from metadata for test split)
    # For simplicity, put val images in both val and test
    import shutil
    test_img_dir = OUTPUT_DIR / "images" / "test"
    test_lbl_dir = OUTPUT_DIR / "labels" / "test"
    if not test_img_dir.exists():
        # Use val as test since we don't have a separate test set
        # Just symlink or copy val to test
        test_img_dir.mkdir(parents=True, exist_ok=True)
        test_lbl_dir.mkdir(parents=True, exist_ok=True)
        for fname in val_downloaded[: max(1, len(val_downloaded) // 2)]:
            src_img = OUTPUT_DIR / "images" / "val" / fname
            src_lbl = OUTPUT_DIR / "labels" / "val" / fname.replace(".jpg", ".txt")
            if src_img.exists():
                shutil.copy2(str(src_img), str(test_img_dir / fname))
            if src_lbl.exists():
                shutil.copy2(str(src_lbl), str(test_lbl_dir / fname.replace(".jpg", ".txt")))

    from utils.dataset import write_yaml_config
    write_yaml_config(
        output_path=OUTPUT_DIR / "frisbee.yaml",
        dataset_dir=str(OUTPUT_DIR),
    )

    print(f"\nDone! Dataset: {OUTPUT_DIR}")
    print(f"  Train images with labels: {train_count}")
    print(f"  Val images with labels: {val_count}")


if __name__ == "__main__":
    main()
