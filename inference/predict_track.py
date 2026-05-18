"""ByteTrack tracking + field coordinate mapping.

Runs YOLO inference with ByteTrack on a video, maps detections to field
coordinates via homography, outputs annotated video + CSV trajectory data.

Usage:
    python3 inference/predict_track.py --video movie/test.mp4
    python3 inference/predict_track.py --video movie/test.mp4 --no-visualize
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from configs.models import V3_MODEL, DEFAULT_CONF
from utils.homography import load_calibration, pixel_to_world


def find_calibration(video_path: Path) -> dict | None:
    """Auto-detect calibration JSON for a video. Returns None if not found."""
    calib_dir = Path("configs/homography")
    if not calib_dir.exists():
        return None

    stem = video_path.stem
    candidate = calib_dir / f"{stem}.json"
    if candidate.exists():
        return load_calibration(candidate)

    if "_" in stem:
        base = stem.split("_", 1)[0]
        candidate = calib_dir / f"{base}.json"
        if candidate.exists():
            return load_calibration(candidate)

    return None


def export_tracks_csv(rows: list[dict], output_path: Path) -> None:
    """Write tracking data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "px", "py", "wx", "wy", "conf"])
        writer.writeheader()
        writer.writerows(rows)


def get_args():
    parser = argparse.ArgumentParser(description="ByteTrack tracking with field coordinate mapping")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default=str(V3_MODEL))
    parser.add_argument("--calibration", type=str, default=None, help="Explicit calibration JSON path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-visualize", action="store_true", default=False, help="Skip video output")
    return parser.parse_args()


def main():
    args = get_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print(f"Video:  {video_path.name}")
    print(f"Model:  {args.model}")
    print(f"Conf:   {args.conf}")

    calib = None
    matrix = None
    if args.calibration:
        calib = load_calibration(Path(args.calibration))
        matrix = calib["matrix"]
        print(f"Calib:  {args.calibration}")
    else:
        calib = find_calibration(video_path)
        if calib:
            matrix = calib["matrix"]
            print("Calib:  auto-detected")
        else:
            print("Calib:  NONE — output will be pixel-only")

    output_dir = Path(args.output_dir or f"runs/track/{video_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    model = YOLO(args.model)
    print("\nTracking...")

    all_rows: list[dict] = []
    out_video = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not args.no_visualize:
        out_video = cv2.VideoWriter(
            str(output_dir / f"{video_path.stem}_tracked.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
        )

    results = model.track(
        source=str(video_path),
        conf=args.conf,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for frame_idx, r in enumerate(results, 1):
        orig_frame = r.orig_img if hasattr(r, "orig_img") else None
        if orig_frame is None:
            continue

        if r.boxes is not None and r.boxes.id is not None:
            for box, tid, conf_val in zip(r.boxes.xyxy, r.boxes.id, r.boxes.conf):
                px = float((box[0] + box[2]) / 2.0)
                py = float(box[3])
                tid_int = int(tid)

                row = {"frame": frame_idx, "track_id": tid_int, "px": round(px, 1),
                       "py": round(py, 1), "wx": None, "wy": None, "conf": round(float(conf_val), 4)}

                if matrix is not None:
                    try:
                        wx, wy = pixel_to_world(matrix, px, py)
                        row["wx"] = round(wx, 2)
                        row["wy"] = round(wy, 2)
                    except Exception:
                        print(f"  WARNING: pixel_to_world failed frame {frame_idx}, track {tid_int}")

                all_rows.append(row)

                if out_video is not None:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    tid_label = f"#{tid_int}"
                    cv2.putText(orig_frame, tid_label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if row["wx"] is not None:
                        coord_text = f"({row['wx']},{row['wy']})m"
                        cv2.putText(orig_frame, coord_text, (x2 - 120, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if out_video is not None:
            info = f"Frame: {frame_idx}  Dets: {len(r.boxes) if r.boxes else 0}"
            cv2.putText(orig_frame, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            out_video.write(orig_frame)

        if frame_idx % 100 == 0:
            print(f"  {frame_idx} frames, {len(all_rows)} detections")

    if out_video is not None:
        out_video.release()

    total_frames = frame_idx
    csv_path = output_dir / f"{video_path.stem}_tracks.csv"
    export_tracks_csv(all_rows, csv_path)
    dets_per_sec = len(all_rows) / max(total_frames / fps, 0.001) if fps > 0 and total_frames > 0 else 0
    print(f"\nDone: {total_frames} frames, {len(all_rows)} detections ({dets_per_sec:.1f} dets/s)")
    print(f"CSV:   {csv_path}")
    if out_video is not None:
        print(f"Video: {output_dir / f'{video_path.stem}_tracked.mp4'}")


if __name__ == "__main__":
    main()
