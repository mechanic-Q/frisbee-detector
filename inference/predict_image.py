from ultralytics import YOLO
import argparse
import os


def predict_image(model_path, image_path, conf=0.25, save_dir="runs/predict"):
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        project=save_dir,
        name="frisbee",
    )
    for r in results:
        if r.boxes is not None:
            print(f"Detected {len(r.boxes)} frisbee(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Image path or directory")
    parser.add_argument("--model", default="runs/detect/frisbee_det_s/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-dir", default="runs/predict")
    args = parser.parse_args()
    if not os.path.exists(args.model):
        candidates = [
            args.model,
            "/mnt/e/frisbee-detector/runs/detect/runs/detect/frisbee_det_s/weights/best.pt",
        ]
        args.model = next((c for c in candidates if os.path.exists(c)), args.model)
    predict_image(args.model, args.source, args.conf, args.save_dir)
