"""Single-frisbee tracker with trajectory visualization.

Runs YOLO per-frame, picks the most-likely frisbee candidate via Kalman
filter + weighted scoring, draws a sliding-window trajectory line.

Usage:
    python3 inference/predict_track.py --video movie/test.mp4
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from configs.models import V3_MODEL, DEFAULT_CONF
from utils.homography import load_calibration, pixel_to_world
from utils.tracker_utils import init_kalman, score_candidates, Trajectory

LOST_TRACK_THRESHOLD = 15  # frames


def find_calibration(video_path: Path) -> dict | None:
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


def get_args():
    parser = argparse.ArgumentParser(description="Single-frisbee tracker with trajectory")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", type=str, default=str(V3_MODEL))
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-visualize", action="store_true", default=False)
    return parser.parse_args()


def draw_trajectory(frame: np.ndarray, trajectory: Trajectory, window: int = 50) -> None:
    pts = trajectory.get_window(window)
    if len(pts) < 2:
        return
    n = len(pts)
    for i in range(n - 1):
        p0 = (int(round(pts[i][0])), int(round(pts[i][1])))
        p1 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
        t = i / max(n - 1, 1)
        g = int(255 * (1.0 - t))
        b = int(127 * (1.0 - t))
        cv2.line(frame, p0, p1, (b, g, 255), 2, cv2.LINE_AA)
    last = (int(round(pts[-1][0])), int(round(pts[-1][1])))
    cv2.circle(frame, last, 5, (0, 255, 0), -1)


def main():
    args = get_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print(f"Video:  {video_path.name}")
    print(f"Model:  {args.model}")
    print(f"Conf:   {args.conf}")

    matrix = None
    if args.calibration:
        matrix = load_calibration(Path(args.calibration))["matrix"]
        print(f"Calib:  {args.calibration}")
    else:
        calib = find_calibration(video_path)
        if calib:
            matrix = calib["matrix"]
            print("Calib:  auto-detected")
        else:
            print("Calib:  NONE — pixel-only")

    output_dir = Path(args.output_dir or f"runs/track/{video_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    model = YOLO(args.model)
    kf = init_kalman()
    trajectory = Trajectory()
    status = "searching"
    lost_counter = 0
    print(f"\nTracking (single-frisbee)...")

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

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=args.conf, verbose=False)
        r = results[0]
        candidates = []
        if r.boxes is not None:
            for box, conf_val in zip(r.boxes.xyxy, r.boxes.conf):
                candidates.append({
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "conf": float(conf_val),
                })

        current_status = status
        if not candidates:
            lost_counter += 1
            if lost_counter > LOST_TRACK_THRESHOLD:
                status = "searching"
                kf = init_kalman()
            else:
                status = "predicting"
                prediction = kf.predict()
                px = float(prediction[0])
                py = float(prediction[1])
                if 0 <= px <= w and 0 <= py <= h:
                    trajectory.push(px, py, 0)
                current_status = "predicting"
        else:
            prediction_arg = kf.predict() if status == "tracking" else None
            pred_pt = (float(prediction_arg[0]), float(prediction_arg[1])) if prediction_arg is not None else None

            best_idx = score_candidates(candidates, trajectory if status == "tracking" else None, pred_pt)
            if best_idx >= 0:
                best = candidates[best_idx]
                bx = best["box"]
                cx = (bx[0] + bx[2]) / 2.0
                cy = (bx[1] + bx[3]) / 2.0
                bw = bx[2] - bx[0]
                bh = bx[3] - bx[1]
                area = bw * bh
                meas = np.array([[cx], [cy]], dtype=np.float32)
                kf.correct(meas)
                status = "tracking"
                lost_counter = 0
                trajectory.push(cx, cy, area)
                current_status = "tracking"

                vx = float(kf.statePost[2])
                vy = float(kf.statePost[3])
                row = {
                    "frame": frame_idx, "px": round(cx, 1), "py": round(cy, 1),
                    "vx": round(vx, 2), "vy": round(vy, 2),
                    "conf": round(float(best["conf"]), 4), "status": status,
                }
                if matrix is not None:
                    try:
                        wx, wy = pixel_to_world(matrix, cx, cy)
                        row["wx"] = round(wx, 2)
                        row["wy"] = round(wy, 2)
                    except Exception:
                        row["wx"] = None
                        row["wy"] = None
                all_rows.append(row)

                if out_video is not None:
                    x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Frisbee", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if out_video is not None:
            draw_trajectory(frame, trajectory)
            info_line = f"Frame: {frame_idx}  Status: {current_status}"
            cv2.putText(frame, info_line, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            det_info = f"Dets: {len(candidates)}"
            cv2.putText(frame, det_info, (w - 150, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            out_video.write(frame)

        if frame_idx % 100 == 0:
            print(f"  {frame_idx} frames, {len(all_rows)} tracking rows")

    cap.release()
    if out_video is not None:
        out_video.release()

    if all_rows:
        csv_path = output_dir / f"{video_path.stem}_tracks.csv"
        fieldnames = ["frame", "px", "py", "vx", "vy", "conf", "status", "wx", "wy"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDone: {frame_idx} frames, {len(all_rows)} tracking rows")
        print(f"CSV:   {csv_path}")
    else:
        print("\nDone: 0 frames tracked")

    if out_video is not None:
        print(f"Video: {output_dir / f'{video_path.stem}_tracked.mp4'}")


if __name__ == "__main__":
    main()
