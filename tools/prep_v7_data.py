"""v7 data preparation: replace backgrounds with FP crops + add TP frames.

Replaces 483 generic/backgound training images with 178 bbox-level hard negatives
(FP crops from review) and adds 19 TP frames from test video with YOLO labels.

Usage:
    python3 tools/prep_v7_data.py [-h] [--dry-run]
                                  [--video VIDEO]
                                  [--model MODEL]
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm import tqdm

from configs.paths import PROJECT_ROOT
from utils.dataset import write_yaml_config

CROPS_DIR = PROJECT_ROOT / "data" / "perbox_crops"
REVIEW_CSV = CROPS_DIR / "review_results.csv"
TRAIN_IMG_DIR = PROJECT_ROOT / "data" / "datasets" / "frisbee_merged" / "images" / "train"
TRAIN_LBL_DIR = PROJECT_ROOT / "data" / "datasets" / "frisbee_merged" / "labels" / "train"
DST_DIR = PROJECT_ROOT / "data" / "datasets" / "frisbee_merged"

def find_all_backgrounds() -> list[tuple[Path, Path]]:
    """Find all background images (empty label files). Returns (image_path, label_path) pairs."""
    backgrounds = []
    for lbl_path in sorted(TRAIN_LBL_DIR.glob("*.txt")):
        if lbl_path.stat().st_size == 0:
            img_path = _find_image_for_label(lbl_path)
            if img_path:
                backgrounds.append((img_path, lbl_path))
            else:
                print(f"  WARNING: No image found for label {lbl_path.name}")
    return backgrounds


def _find_image_for_label(lbl_path: Path) -> Path | None:
    """Find the image file corresponding to a label file."""
    stem = lbl_path.stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        img_path = TRAIN_IMG_DIR / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def load_review_results() -> list[dict]:
    """Load the per-box review CSV."""
    rows = []
    with open(REVIEW_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_fp_crops(review_rows: list[dict]) -> list[dict]:
    """Filter rows marked as FP."""
    return [r for r in review_rows if r["result"] == "FP"]


def get_tp_entries(review_rows: list[dict]) -> list[dict]:
    """Filter rows marked as TP."""
    return [r for r in review_rows if r["result"] == "TP"]


def extract_tp_frame_and_labels(
    video_path: Path,
    model: YOLO,
    frame_num: int,
    n_tp: int,
    output_stem: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Extract one frame from video, generate YOLO labels for top-N detections.
    
    Labels the top N v3 detections (by confidence) as TP, where N is the number
    of TP entries from the review for this frame. This is correct because the
    original crop extraction sorted by confidence, so the top-N now matches
    the original TP crops.
    
    Returns (detections_labeled, detections_total).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video {video_path}")
        return 0, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 1-indexed → 0-indexed
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  ERROR: Cannot read frame {frame_num} from video")
        return 0, 0

    # Run v3 inference
    results = model.predict(frame, conf=0.10, verbose=False, imgsz=1280)

    # Collect all detections sorted by confidence descending
    all_dets: list[tuple[float, float, float, float, float]] = []
    for r in results:
        if r.boxes is None:
            continue
        h_img, w_img = frame.shape[:2]
        for box in r.boxes:
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2 / w_img
            cy = (y1 + y2) / 2 / h_img
            bw = (x2 - x1) / w_img
            bh = (y2 - y1) / h_img
            all_dets.append((conf, cx, cy, bw, bh))

    all_dets.sort(key=lambda x: x[0], reverse=True)
    top_n = all_dets[:n_tp]

    if not dry_run:
        img_out = TRAIN_IMG_DIR / f"{output_stem}.jpg"
        lbl_out = TRAIN_LBL_DIR / f"{output_stem}.txt"
        cv2.imwrite(str(img_out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if top_n:
            lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for _, cx, cy, bw, bh in top_n]
            lbl_out.write_text("\n".join(lines) + "\n")
        else:
            lbl_out.touch()

    return len(top_n), len(all_dets)


def main():
    parser = argparse.ArgumentParser(description="v7 data preparation")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    parser.add_argument("--video", default=str(PROJECT_ROOT / "movie" / "25866279684-1-192_55-56min.mp4"),
                        help="Source video for TP frame extraction")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "runs" / "detect" / "frisbee_det_s_v3" / "weights" / "best.pt"),
                        help="v3 model for TP label generation")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    dry_run = args.dry_run

    # Validate inputs
    if not REVIEW_CSV.exists():
        print(f"ERROR: Review CSV not found: {REVIEW_CSV}")
        sys.exit(1)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    if not TRAIN_IMG_DIR.exists() or not TRAIN_LBL_DIR.exists():
        print(f"ERROR: Training directories not found in {DST_DIR}")
        sys.exit(1)

    # Load review data
    review_rows = load_review_results()
    print(f"\nReview data: {len(review_rows)} total")

    # Phase A: Remove backgrounds
    print(f"\n{'='*60}")
    print("Phase A: Remove background images (empty labels)")
    print(f"{'='*60}")

    backgrounds = find_all_backgrounds()
    print(f"  Found {len(backgrounds)} background images to remove")

    if dry_run:
        print(f"  [DRY-RUN] Would remove {len(backgrounds)} images + labels")
    else:
        for img_path, lbl_path in tqdm(backgrounds, desc="Removing backgrounds"):
            img_path.unlink(missing_ok=True)
            lbl_path.unlink(missing_ok=True)
        print(f"  Removed {len(backgrounds)} background images")

    # Phase B: Add FP crops as hard negatives
    print(f"\n{'='*60}")
    print("Phase B: Add FP crops as hard negative backgrounds")
    print(f"{'='*60}")

    fp_crops = get_fp_crops(review_rows)
    print(f"  Found {len(fp_crops)} FP crops to add")

    if dry_run:
        for row in fp_crops[:3]:
            print(f"  [DRY-RUN] Would copy {row['filename']} → hardneg_v7_{row['filename']}")
        if len(fp_crops) > 3:
            print(f"  [DRY-RUN] ... and {len(fp_crops) - 3} more")
    else:
        copied = 0
        for row in tqdm(fp_crops, desc="Copying FP crops"):
            src = CROPS_DIR / row["filename"]
            dst_name = f"hardneg_v7_{row['filename']}"
            dst_img = TRAIN_IMG_DIR / dst_name
            dst_lbl = TRAIN_LBL_DIR / dst_name.replace(".jpg", ".txt")
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping")
                continue
            shutil.copy2(str(src), str(dst_img))
            dst_lbl.touch()
            copied += 1
        print(f"  Copied {copied} FP crops")

    # Phase C: Extract TP frames + generate labels
    print(f"\n{'='*60}")
    print("Phase C: Extract TP frames and generate YOLO labels")
    print(f"{'='*60}")

    tp_entries = get_tp_entries(review_rows)
    unique_frames: dict[int, int] = {}
    for entry in tp_entries:
        frame = int(entry["frame"])
        unique_frames[frame] = unique_frames.get(frame, 0) + 1

    print(f"  TP entries: {len(tp_entries)} across {len(unique_frames)} unique frames")
    print(f"  Frame numbers: {sorted(unique_frames.keys())}")

    if not dry_run:
        print(f"  Loading v3 model...")
        model = YOLO(str(model_path))
    else:
        model = None

    total_labeled = 0
    total_dets_found = 0
    for frame_num in tqdm(sorted(unique_frames.keys()), desc="Extracting TP frames"):
        output_stem = f"tp_v7_{frame_num:05d}"
        n_tp = unique_frames[frame_num]

        if dry_run:
            print(f"  [DRY-RUN] Would extract frame {frame_num} → {output_stem}.jpg (top-{n_tp} detections)")
            continue

        labeled, total = extract_tp_frame_and_labels(
            video_path, model, frame_num, n_tp, output_stem, dry_run=False
        )
        total_labeled += labeled
        total_dets_found += total

    if not dry_run:
        print(f"  TP frames extracted: {len(unique_frames)}")
        print(f"  TP detections labeled: {total_labeled}/{total_dets_found} (top-N out of all v3 detections)")
        if total_labeled != len(tp_entries):
            print(f"  NOTE: Labeled {total_labeled} but expected {len(tp_entries)} (some frames may have fewer detections than expected)")

    # Phase D: Regenerate YAML config
    print(f"\n{'='*60}")
    print("Phase D: Regenerate YAML config")
    print(f"{'='*60}")

    if not dry_run:
        write_yaml_config(
            output_path=PROJECT_ROOT / "configs" / "frisbee_merged.yaml",
            dataset_dir=str(DST_DIR),
        )
        print(f"  Config written to: {PROJECT_ROOT / 'configs' / 'frisbee_merged.yaml'}")
    else:
        print(f"  [DRY-RUN] Would regenerate: {PROJECT_ROOT / 'configs' / 'frisbee_merged.yaml'}")

    # Final summary
    if not dry_run:
        remaining_imgs = len(list(TRAIN_IMG_DIR.glob("*")))
        remaining_empty = sum(1 for f in TRAIN_LBL_DIR.glob("*.txt") if f.stat().st_size == 0)
        remaining_nonempty = sum(1 for f in TRAIN_LBL_DIR.glob("*.txt") if f.stat().st_size > 0)
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"  Training images:     {remaining_imgs}")
        print(f"  Positive (labels):   {remaining_nonempty}")
        print(f"  Background (empty):  {remaining_empty}")
        pct = remaining_empty / max(remaining_imgs, 1) * 100
        print(f"  Background ratio:    {pct:.1f}%")


if __name__ == "__main__":
    main()
