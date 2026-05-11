from ultralytics import YOLO
from pathlib import Path
import os
import sys


def evaluate_trained_model(model_path, video_path, conf_threshold=0.25):
    model = YOLO(model_path)
    print(f"\nModel: {model_path}")
    print(f"Inference on: {video_path}")
    print(f"Conf threshold: {conf_threshold}")
    results = model.predict(
        source=video_path,
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
    per_frame_counts = {}
    all_confs = []

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

    rate = frames_with_det / max(total_frames, 1)
    avg_per_frame = total_detections / max(total_frames, 1)

    print(f"\n{'='*50}")
    print(f"Trained Model Evaluation")
    print(f"Total frames:            {total_frames}")
    print(f"Frames with detections:  {frames_with_det} ({rate:.1%})")
    print(f"Total detections:        {total_detections}")
    print(f"Avg detections/frame:    {avg_per_frame:.2f}")
    print(f"")
    print(f"Per-frame distribution:")
    for n in sorted(per_frame_counts.keys()):
        count = per_frame_counts[n]
        pct = count / total_frames * 100
        bar = "#" * int(pct / 2)
        print(f"  {n:>3} dets: {count:>5} ({pct:5.1f}%) {bar}")
    if all_confs:
        all_confs.sort()
        n = len(all_confs)
        print(f"")
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


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/runs/detect/frisbee_det_s_v2/weights/best.pt"
    video_path = sys.argv[2] if len(sys.argv) > 2 else "/mnt/e/firsbee/03_datasets/frisbee-tracking/clip-5.mp4"
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    evaluate_trained_model(model_path, video_path, conf)
