"""Collect hard negatives by frame-level VLM filtering.

Strategy: process each test frame through v3, then skip cropping of individual
bboxes. Instead, ask VLM once per frame: "does this frame contain a frisbee?"
If NO → save the frame as a hard negative background image.
If YES → skip (has real frisbee, don't use as negative).

This is MUCH faster (~10x fewer VLM calls) and still provides strong background
context frames (not just small crops but full game scenes with no frisbee).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import base64
import os
import cv2
import time
from openai import OpenAI
from ultralytics import YOLO

GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
GLM_CLIENT = OpenAI(api_key=GLM_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")

VLM_CACHE = {}


def vlm_has_frisbee(frame, frame_id):
    key = str(frame_id)
    if key in VLM_CACHE:
        return VLM_CACHE[key]
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()
    try:
        resp = GLM_CLIENT.chat.completions.create(
            model="glm-4v-flash",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": "Is there a frisbee (flying disc) visible anywhere in this image? A frisbee is a round plastic disc thrown in the sport of ultimate frisbee. Answer only YES or NO."}
                ]
            }]
        )
        answer = resp.choices[0].message.content.strip().upper()
        result = "YES" in answer
        VLM_CACHE[key] = result
        return result
    except Exception as e:
        print(f"  VLM error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/detect/frisbee_det_s_v3/weights/best.pt")
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--output", default="data/datasets/frisbee_merged/images/train_hard_neg")
    parser.add_argument("--stride", type=int, default=5,
                        help="Process every Nth frame (default: 5 = 300 frames for ~55s video)")
    parser.add_argument("--max-negatives", type=int, default=400,
                        help="Stop after collecting this many hard negatives")
    args = parser.parse_args()

    model = YOLO(args.model)
    os.makedirs(args.output, exist_ok=True)
    labels_dir = args.output.replace("/images/", "/labels/")
    os.makedirs(labels_dir, exist_ok=True)

    neg_count = 0
    frames_checked = 0
    vlm_calls = 0

    for video_path in args.videos:
        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open {video_path}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nProcessing {video_name} ({total} frames, stride={args.stride})")

        frame_idx = 0
        while cap.isOpened() and neg_count < args.max_negatives:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            results = model.predict(frame, conf=args.conf, verbose=False, imgsz=1280)
            has_detections = False
            high_conf = 0
            for r in results:
                for box in r.boxes:
                    has_detections = True
                    high_conf = max(high_conf, float(box.conf[0]))
                    break

            frames_checked += 1

            if not has_detections:
                frame_idx += 1
                continue

            if high_conf > 0.35:
                frame_idx += 1
                continue

            vlm_calls += 1
            is_frisbee = vlm_has_frisbee(frame, f"{video_name}_f{frame_idx}")

            if is_frisbee is False:
                basename = f"hardneg_{neg_count:04d}.jpg"
                dst = os.path.join(args.output, basename)
                cv2.imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                Path(os.path.join(labels_dir, basename.replace(".jpg", ".txt"))).touch()
                neg_count += 1
                print(f"  [{neg_count}/{args.max_negatives}] NO frisbee → saved ({video_name} frame {frame_idx})")

            frame_idx += 1

        cap.release()

    print(f"\n{'='*50}")
    print(f"Frames checked: {frames_checked}")
    print(f"VLM calls: {vlm_calls}")
    print(f"Hard negatives collected: {neg_count}")
    print(f"Saved to: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
