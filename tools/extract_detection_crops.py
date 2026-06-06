"""Extract detection crops from a model + video for FP spot-check.

Saves each detection as a crop image to a directory, then the user can
review with: streamlit run tools/review_web.py -- --crop-dir <dir>

Usage:
    python3 tools/extract_detection_crops.py \
        --model runs/detect/frisbee_det_s_v7_quick/weights/best.pt \
        --video movie/25866279684-1-192_55-56min.mp4 \
        --output data/fp_spotcheck_v7_55min \
        --conf 0.35
    python3 tools/extract_detection_crops.py \
        --model runs/detect/frisbee_det_s_v7_quick/weights/best.pt \
        --video movie/clip_20-23min.mp4 \
        --output data/fp_spotcheck_v7_20min \
        --conf 0.35
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm import tqdm

from tools.generate_annotation_tasks import append_unique_tasks, build_bbox_review_task

def main():
    parser = argparse.ArgumentParser(description="Extract detection crops for FP spot-check")
    parser.add_argument("--model", required=True, help="Model weights path")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--output", required=True, help="Output directory for crops")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument('--task-store', default=None, help='Optional JSONL task store for bbox_review AnnotationTasks')
    parser.add_argument('--frame-output', default=None, help='Optional directory for source frame context images')
    args = parser.parse_args()

    model_path = Path(args.model)
    video_path = Path(args.video)
    output_dir = Path(args.output)
    conf_thresh = args.conf

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem.replace("25866279684-1-192_55-56min", "55-56min")

    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Conf:  {conf_thresh}")
    print(f"Output: {output_dir}")

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    generated_tasks = []
    frame_output_dir = Path(args.frame_output) if args.frame_output else output_dir / "frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)


    crop_idx = 0
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(frame, conf=conf_thresh, verbose=False, imgsz=640)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                fname = f"crop_{crop_idx:04d}_f{frame_idx:05d}_c{conf:.2f}_{video_name}.jpg"
                cv2.imwrite(str(output_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                crop_idx += 1
                # Save frame context (dedup per frame)
                frame_context_path = frame_output_dir / f"frame_f{frame_idx:05d}_{video_name}.jpg"
                if not frame_context_path.exists():
                    cv2.imwrite(str(frame_context_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                if args.task_store:
                    timestamp_sec = frame_idx / fps
                    generated_tasks.append(
                        build_bbox_review_task(
                            source_video=str(video_path),
                            timestamp_sec=timestamp_sec,
                            frame_index=frame_idx,
                            bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                            crop_path=str(output_dir / fname),
                            frame_path=str(frame_context_path),
                            model_name=model_path.stem,
                            model_conf=conf,
                            tags=["detection_crop"],
                        )
                    )

        pbar.update(1)
        if frame_idx % 500 == 0:
            pbar.set_postfix(crops=crop_idx)

    cap.release()
    pbar.close()

    print(f"\nExtracted {crop_idx} crops to {output_dir}")

    # Create review CSV with all crops unclassified
    csv_path = output_dir / "review_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "result", "frame", "conf"])
        paths = sorted(output_dir.glob("*.jpg"))
        for p in paths:
            parts = p.stem.split("_")
            # crop_NNNN_fXXXXX_cC.CONF_VIDEO
            frame_part = parts[2]  # fXXXXX
            conf_part = parts[3]   # cC.CONF
            frame = frame_part[1:]
            conf = conf_part[1:]
            writer.writerow([p.name, "", frame, conf])

    print(f"Review CSV: {csv_path}")
    print(f"Total crops: {crop_idx}")
    if args.task_store:
        merged = append_unique_tasks(args.task_store, generated_tasks)
        print(f"Task store: {args.task_store}")
        print(f"Generated tasks: {len(generated_tasks)}")
        print(f"Total tasks: {len(merged)}")

    print(f"\nTo review: streamlit run tools/review_web.py -- --crop-dir {output_dir}")


if __name__ == "__main__":
    main()
