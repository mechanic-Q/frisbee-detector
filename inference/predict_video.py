from ultralytics import YOLO
from pathlib import Path
import os
import sys


def evaluate_trained_model(model_path, video_path, conf_threshold=0.25):
    model = YOLO(model_path)
    print(f"\nModel: {model_path}")
    print(f"Inference on: {video_path}")
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/eval",
        name=Path(model_path).parent.parent.name if "weights" in str(model_path) else "eval",
    )
    frisbee_det = 0
    total_frames = 0
    for r in results:
        total_frames += 1
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # class 0 = frisbee
                    frisbee_det += 1
    rate = frisbee_det / max(total_frames, 1)
    print(f"\n{'='*50}")
    print(f"Trained Model Evaluation")
    print(f"Total frames: {total_frames}")
    print(f"Frames with frisbee: {frisbee_det}")
    print(f"Frisbee detection rate: {rate:.2%}")
    print(f"{'='*50}")
    return rate


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/frisbee_det_s/weights/best.pt"
    video_path = sys.argv[2] if len(sys.argv) > 2 else "/mnt/e/firsbee/03_datasets/frisbee-tracking/clip-5.mp4"
    evaluate_trained_model(model_path, video_path)
