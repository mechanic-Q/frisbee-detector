"""Export model inference to yololabeler predictions format for desktop review.

Usage:
  # Export predictions from a model run on images/videos
  python3 tools/review_desktop.py --model runs/detect/frisbee_det_s_v3/weights/best.pt \\
      --source data/fp_spotcheck_50/ --conf 0.35

  # Then launch the desktop reviewer:
  yololabeler data/fp_spotcheck_50/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO predictions to yololabeler format")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--source", required=True, help="Path to images or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: {source} not found")
        sys.exit(1)

    model = YOLO(args.model)

    pred_dir = source / "predictions" / "detect"
    os.makedirs(pred_dir, exist_ok=True)

    # Create classes.json
    classes = {"0": {"name": "frisbee", "color": "#3cb44b"}}
    os.makedirs(source / "state", exist_ok=True)
    with open(source / "state" / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)

    if source.is_dir():
        exts = (".jpg", ".jpeg", ".png")
        images = sorted([p for p in source.iterdir() if p.suffix.lower() in exts])
    else:
        images = [source]

    total_dets = 0
    for img_path in images:
        results = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        pred_file = pred_dir / f"{img_path.stem}.txt"
        lines = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"{cls_id} {conf:.4f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                total_dets += 1
        pred_file.write_text("\n".join(lines) + "\n" if lines else "")

    print(f"Processed {len(images)} images, {total_dets} detections")
    print(f"Predictions saved to: {pred_dir}")
    print(f"\nLaunch review: yololabeler {source.resolve()}")


if __name__ == "__main__":
    main()
