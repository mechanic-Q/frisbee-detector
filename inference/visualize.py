import cv2
from pathlib import Path
import os
import sys


def draw_labels(image_path, labels_dir, output_dir, class_names=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return
    stem = Path(image_path).stem
    label_path = os.path.join(labels_dir, stem + ".txt")
    if not os.path.exists(label_path):
        print(f"No label file: {label_path}")
        return
    os.makedirs(output_dir, exist_ok=True)
    h, w = img.shape[:2]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = class_names[cls_id] if class_names and cls_id < len(class_names) else f"cls{cls_id}"
            conf = float(parts[5]) if len(parts) > 5 else None
            text = f"{label} {conf:.2f}" if conf else label
            cv2.putText(img, text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    output_path = os.path.join(output_dir, stem + "_detected.jpg")
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <image_path> <labels_dir> [output_dir]")
        sys.exit(1)
    image_path = sys.argv[1]
    labels_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "runs/visualize"
    draw_labels(image_path, labels_dir, output_dir, class_names=["frisbee"])
