"""Download COCO frisbee images with parallel threads + convert to YOLO."""

import json
import os
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import PROJECT_ROOT

COCO_CLASS_ID = 34
ANN_DIR = Path("/tmp/coco_frisbee/annotations/annotations")
OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets" / "frisbee_coco"
BASE_URL = "http://images.cocodataset.org"
MAX_WORKERS = 20


def get_frisbee_images(ann_file: Path) -> tuple[dict, dict]:
    """Return ({filename: img_info}, {filename: label_lines})."""
    with open(ann_file) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}
    frisbee_img_ids = set()
    for ann in data["annotations"]:
        if ann["category_id"] == COCO_CLASS_ID:
            frisbee_img_ids.add(ann["image_id"])

    images: dict[str, dict] = {}
    labels: dict[str, list[str]] = {}

    for img in data["images"]:
        if img["id"] not in frisbee_img_ids:
            continue
        fn = img["file_name"]
        images[fn] = img
        labels[fn] = []

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
        labels[img["file_name"]].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return images, labels


def download_one(fn: str, url: str, dest: Path) -> bool:
    """Download single image. Returns True on success."""
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception:
        return False


def download_all(images: dict[str, dict], split: str) -> int:
    """Parallel download. Returns count downloaded."""
    img_dir = OUTPUT_DIR / "images" / split
    lbl_dir = OUTPUT_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.stem for f in img_dir.glob("*"))

    tasks = []
    existing_count = 0
    for fn in sorted(images.keys()):
        stem = Path(fn).stem
        if stem in existing:
            existing_count += 1
            continue
        url = f"{BASE_URL}/train2017/{fn}"
        dest = img_dir / fn
        tasks.append((fn, url, dest))

    print(f"  {split}: {existing_count} already exist, {len(tasks)} to download")

    if tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            fut_map = {pool.submit(download_one, fn, url, dest): fn for fn, url, dest in tasks}
            done = 0
            for fut in as_completed(fut_map):
                fn = fut_map[fut]
                if fut.result():
                    done += 1
                else:
                    print(f"  FAILED: {fn}")
                    dest.unlink(missing_ok=True)
                if done % 200 == 0:
                    print(f"    Progress: {done}/{len(tasks)}")

        print(f"  Downloaded: {done}/{len(tasks)}")

    # Write labels
    count = 0
    for fn, lines in images.items():
        if lines:
            lbl_path = lbl_dir / fn.replace(".jpg", ".txt")
            lbl_path.write_text("\n".join(lines) + "\n")
            count += 1

    total_imgs = existing_count + len(tasks)
    print(f"  Total {split} images with labels: {count} labeled, {total_imgs} images")
    return total_imgs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ANN_DIR.exists():
        print("ERROR: annotations not found. Run convert_coco_frisbee.py first.")
        return

    print("Parsing COCO annotations...")
    train_images, train_labels = get_frisbee_images(ANN_DIR / "instances_train2017.json")
    val_images, val_labels = get_frisbee_images(ANN_DIR / "instances_val2017.json")

    print(f"  Train frisbee images: {len(train_images)}")
    print(f"  Val frisbee images:   {len(val_images)}")

    # Fix val URLs - val images are in val2017/
    val_urls = {}
    # We need to remap URLs for val images
    print("\nNote: val images use val2017/ prefix. Fixed.")

    print("\nDownloading training images (parallel)...")
    train_count = download_all(train_images, "train")

    print("\nDownloading val images (parallel)...")
    # For val images, we need to modify the URL
    img_dir = OUTPUT_DIR / "images" / "val"
    lbl_dir = OUTPUT_DIR / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.stem for f in img_dir.glob("*"))
    tasks = []
    existing_count = 0
    for fn in sorted(val_images.keys()):
        stem = Path(fn).stem
        if stem in existing:
            existing_count += 1
            continue
        url = f"{BASE_URL}/val2017/{fn}"
        dest = img_dir / fn
        tasks.append((fn, url, dest))

    print(f"  val: {existing_count} already exist, {len(tasks)} to download")
    if tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            fut_map = {pool.submit(download_one, fn, url, dest): fn for fn, url, dest in tasks}
            done = 0
            for fut in as_completed(fut_map):
                fn = fut_map[fut]
                if fut.result():
                    done += 1
                else:
                    print(f"  FAILED: {fn}")
                if done % 50 == 0:
                    print(f"    Progress: {done}/{len(tasks)}")
        print(f"  Downloaded: {done}/{len(tasks)}")

    val_count = 0
    for fn, lines in val_images.items():
        if lines:
            lbl_path = lbl_dir / fn.replace(".jpg", ".txt")
            lbl_path.write_text("\n".join(lines) + "\n")
            val_count += 1

    print(f"  Total val images with labels: {val_count}")

    from utils.dataset import write_yaml_config
    write_yaml_config(
        output_path=OUTPUT_DIR / "frisbee.yaml",
        dataset_dir=str(OUTPUT_DIR),
    )

    print(f"\nDone! Output: {OUTPUT_DIR}")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")


if __name__ == "__main__":
    main()
