"""Video inference with per-frame detection statistics.

Supports standard YOLO inference and SAHI sliced inference for small objects.
"""

import argparse
import time
from pathlib import Path

import cv2

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
del _Path

from ultralytics import YOLO

from configs.paths import EXTERNAL
from configs.models import DEFAULT_MODEL, DEFAULT_CONF


def _print_stats(total_frames: int, frames_with_det: int, total_detections: int,
                 per_frame_counts: dict[int, int], all_confs: list[float], elapsed: float) -> float:
    """Print unified detection statistics. Returns frame detection rate."""
    rate = frames_with_det / max(total_frames, 1)
    avg_per_frame = total_detections / max(total_frames, 1)
    fps = total_frames / max(elapsed, 0.001)

    print(f"\n{'='*50}")
    print("Video Inference Evaluation")
    print(f"Total frames:            {total_frames}")
    print(f"Frames with detections:  {frames_with_det} ({rate:.1%})")
    print(f"Total detections:        {total_detections}")
    print(f"Avg detections/frame:    {avg_per_frame:.2f}")
    print(f"Elapsed:                 {elapsed:.1f}s ({fps:.1f} fps)")
    print()
    print("Per-frame distribution:")
    for n in sorted(per_frame_counts.keys()):
        count = per_frame_counts[n]
        pct = count / total_frames * 100
        bar = "#" * int(pct / 2)
        print(f"  {n:>3} dets: {count:>5} ({pct:5.1f}%) {bar}")
    if all_confs:
        all_confs.sort()
        n = len(all_confs)
        print()
        print(f"Confidence stats ({n} detections):")
        print(f"  Min:    {all_confs[0]:.4f}")
        print(f"  Max:    {all_confs[-1]:.4f}")
        print(f"  Mean:   {sum(all_confs)/n:.4f}")
        print(f"  Median: {all_confs[n//2]:.4f}")
        below_50 = sum(1 for c in all_confs if c < 0.50)
        above_70 = sum(1 for c in all_confs if c >= 0.70)
        print(f"  <0.50:  {below_50} ({below_50/n*100:.1f}%)")
        print(f"  >=0.70: {above_70} ({above_70/n*100:.1f}%)")
    print(f"{'='*50}")
    return rate


def evaluate_trained_model(
    model_path: str | Path,
    video_path: str | Path,
    conf_threshold: float = DEFAULT_CONF,
) -> float:
    """Run YOLO inference on a video, print detection statistics.

    Returns the fraction of frames with at least one detection.
    """
    model = YOLO(str(model_path))
    print(f"\nModel: {model_path}")
    print(f"Inference on: {video_path}")
    print(f"Conf threshold: {conf_threshold}")

    results = model.predict(
        source=str(video_path),
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/eval",
        name=Path(model_path).parent.parent.name if "weights" in str(model_path) else "eval",
    )

    total_frames = 0
    frames_with_det = 0
    total_detections = 0
    per_frame_counts: dict[int, int] = {}
    all_confs: list[float] = []

    for r in results:
        total_frames += 1
        n = len(r.boxes) if r.boxes is not None else 0
        total_detections += n
        per_frame_counts[n] = per_frame_counts.get(n, 0) + 1
        if n > 0:
            frames_with_det += 1
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    all_confs.append(float(box.conf[0]))

    return _print_stats(total_frames, frames_with_det, total_detections,
                       per_frame_counts, all_confs, elapsed=0)


def evaluate_sahi(
    model_path: str | Path,
    video_path: str | Path,
    conf_threshold: float = DEFAULT_CONF,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_ratio: float = 0.2,
    frame_skip: int = 1,
    output_csv: str | None = None,
) -> float:
    """Run SAHI sliced inference on a video frame-by-frame.

    SAHI slices each frame into overlapping tiles, enabling the model to detect
    very small or distant objects that are invisible at full resolution.

    Returns the fraction of frames with at least one detection.
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    video_path = Path(video_path)
    model_path_str = str(model_path)

    print(f"\n[SAHI] Model: {model_path_str}")
    print(f"[SAHI] Video: {video_path}")
    print(f"[SAHI] Conf: {conf_threshold}, Slice: {slice_height}x{slice_width}, Overlap: {overlap_ratio}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path_str,
        confidence_threshold=conf_threshold,
        device="cuda:0",
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[SAHI] Total frames: {total_frames}, processing every {frame_skip} frame(s)")

    frames_with_det = 0
    total_detections = 0
    per_frame_counts: dict[int, int] = {}
    all_confs: list[float] = []

    csv_rows: list[str] = []
    if output_csv:
        csv_rows.append("frame,det_id,cx,cy,w,h,conf")

    t0 = time.time()
    idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip != 0:
            idx += 1
            continue

        processed += 1
        if processed % 100 == 0:
            elapsed = time.time() - t0
            fps = processed / max(elapsed, 0.001)
            print(f"  Progress: {processed}/{total_frames//frame_skip + 1} frames ({fps:.1f} fps)")

        result = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )

        preds = result.object_prediction_list
        n = len(preds)
        total_detections += n
        per_frame_counts[n] = per_frame_counts.get(n, 0) + 1
        if n > 0:
            frames_with_det += 1
            for pred in preds:
                conf = pred.score.value
                all_confs.append(conf)
                if output_csv:
                    b = pred.bbox
                    csv_rows.append(f"{processed},{len(csv_rows)},{b.center_x:.1f},{b.center_y:.1f},{b.width:.1f},{b.height:.1f},{conf:.4f}")

        idx += 1

    cap.release()
    elapsed = time.time() - t0

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(output_csv).write_text("\n".join(csv_rows))
        print(f"[SAHI] CSV: {output_csv}")

    return _print_stats(processed, frames_with_det, total_detections,
                       per_frame_counts, all_confs, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference on a video")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to model weights")
    parser.add_argument("--video", type=str, default=str(EXTERNAL.frisbee_tracking_dir / "clip-5.mp4"), help="Path to video")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--sahi", action="store_true", help="Use SAHI sliced inference for small objects")
    parser.add_argument("--slice-height", type=int, default=640, help="SAHI slice height")
    parser.add_argument("--slice-width", type=int, default=640, help="SAHI slice width")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI slice overlap ratio")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (SAHI only)")
    parser.add_argument("--output-csv", type=str, default=None, help="Save per-detection CSV to path")
    args = parser.parse_args()

    if args.sahi:
        evaluate_sahi(
            args.model, args.video, args.conf,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_ratio=args.overlap,
            frame_skip=args.frame_skip,
            output_csv=args.output_csv,
        )
    else:
        evaluate_trained_model(args.model, args.video, args.conf)
