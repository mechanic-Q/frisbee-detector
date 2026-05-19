"""Auto-label game frames using GroundedSAM through autodistill.

Usage:
    python3 tools/auto_label_gsam.py --frames-dir data/datasets/game1080/frames
    
Output: YOLO format labels in data/datasets/game1080/labels_gsam/
"""

import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import cv2
import numpy as np

from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology


def nms(boxes, scores, iou_threshold=0.5):
    """Apply NMS to consolidate overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def yolo_format(cx, cy, w, h, cls=0):
    """Convert normalized xywh to YOLO txt line."""
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/datasets/game1080/frames")
    parser.add_argument("--output-dir", default="data/datasets/game1080/labels_gsam")
    parser.add_argument("--conf", type=float, default=0.3, help="Min confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    return parser.parse_known_args()[0]


def main():
    args = get_args()
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.jpg"))
    if args.end:
        frames = frames[args.start:args.end]
    else:
        frames = frames[args.start:]
    
    print(f"Processing {len(frames)} frames...")
    print("Loading GroundedSAM (first load downloads ~2GB weights)...")
    
    base_model = GroundedSAM(ontology=CaptionOntology({
        "frisbee": "frisbee",
        "flying disc": "frisbee",
        "disc": "frisbee",
    }))
    
    img_w = 1920
    img_h = 1080
    total_det = 0
    total_frames = 0
    
    for i, fpath in enumerate(frames):
        t0 = time.time()
        detections = base_model.predict(str(fpath))
        dt = time.time() - t0
        
        boxes = detections.xyxy
        scores = detections.confidence
        
        visible_frisbee = 0
        if len(boxes) > 0:
            # NMS
            keep = nms(boxes, np.array(scores), args.iou)
            keep_scores = [scores[k] for k in keep]
            
            # Filter by confidence
            conf_mask = [s >= args.conf for s in keep_scores]
            keep = [k for k, m in zip(keep, conf_mask) if m]
            
            if keep:
                best = keep[0]  # highest score after NMS
                x1, y1, x2, y2 = boxes[best]
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                
                # Save YOLO format
                txt_path = output_dir / fpath.with_suffix(".txt").name
                txt_path.write_text(yolo_format(cx, cy, bw, bh) + "\n")
                visible_frisbee = 1
                total_det += 1
        else:
            # No detection - save empty label
            txt_path = output_dir / fpath.with_suffix(".txt").name
            txt_path.write_text("")
        
        total_frames += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(frames)} frames ({dt:.1f}s/frame, {total_det} detections)")
    
    print(f"\nDone! {total_frames} frames, {total_det} detections ({total_det/total_frames*100:.0f}% detection rate)")


if __name__ == "__main__":
    main()
