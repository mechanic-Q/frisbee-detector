from ultralytics import YOLO
from pathlib import Path
import os
import sys

def evaluate_model(model_path, video_path, conf_threshold=0.25, save_dir="runs/eval"):
    model = YOLO(model_path)
    print(f"\nModel: {model_path}")
    print(f"Model class mapping: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name=Path(model_path).stem.replace(".", "_"),
    )
    frisbee_det = 0
    total_frames = 0
    for r in results:
        total_frames += 1
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                if "frisbee" in cls_name.lower() or "disc" in cls_name.lower():
                    frisbee_det += 1
    rate = frisbee_det / max(total_frames, 1)
    print(f"\n{'='*50}")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with frisbee: {frisbee_det}")
    print(f"Frisbee detection rate: {rate:.2%}")
    print(f"{'='*50}\n")
    return {"model": model_path, "frames": total_frames,
            "frisbee_detections": frisbee_det, "rate": rate}

if __name__ == "__main__":
    models = [
        "E:/firsbee/03_datasets/UltimateML/models/best.pt",
        "E:/firsbee/03_datasets/ultimate_analytics/web-app/src/data/model/best.pt",
    ]
    candidates = [
        "E:/firsbee/03_datasets/frisbee-tracking/clip-5.mp4",
        "E:/firsbee/03_datasets/frisbee-vision-project/Footage/backhand_2.mp4",
    ]
    video = sys.argv[1] if len(sys.argv) > 1 else None
    if video:
        candidates = [video] + candidates
    video = next((c for c in candidates if os.path.exists(c)), None)
    if not video:
        print("No video found for evaluation.")
        sys.exit(1)
    print(f"Using video: {video}\n")
    all_results = []
    for mp in models:
        if not os.path.exists(mp):
            print(f"Model not found: {mp}, skipping.\n")
            continue
        result = evaluate_model(mp, video)
        all_results.append(result)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"  {Path(r['model']).name}: {r['frisbee_detections']}/{r['frames']} frames ({r['rate']:.2%})")
