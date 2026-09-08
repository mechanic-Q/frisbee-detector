"""VLM-based FP spot-check: classify detection crops as TP or FP using GLM-4V-Flash."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import base64
import csv
import os
import cv2
import time
from openai import OpenAI

GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
GLM_CLIENT = OpenAI(api_key=GLM_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")


def vlm_is_frisbee(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, "unreadable"
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()
    try:
        resp = GLM_CLIENT.chat.completions.create(
            model="glm-4v-flash",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "Is the main object in this image a frisbee (flying disc)? "
                        "A frisbee is a round plastic disc used in the sport of ultimate frisbee. "
                        "Consider only the central/cropped object, not the background. "
                        "Answer only YES or NO."
                    )}
                ]
            }]
        )
        answer = resp.choices[0].message.content.strip().upper()
        if "YES" in answer:
            return True, answer
        elif "NO" in answer:
            return False, answer
        else:
            return None, answer
    except Exception as e:
        print(f"  VLM error: {e}")
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="VLM FP spot-check on detection crops")
    parser.add_argument("--crop-dir", required=True, help="Directory with crop images")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <crop-dir>/vlm_review.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Max crops to review")
    args = parser.parse_args()

    crop_dir = Path(args.crop_dir)
    output = Path(args.output) if args.output else crop_dir / "vlm_review.csv"

    crops = sorted([f for f in crop_dir.iterdir() if f.suffix.lower() in (".jpg", ".png") and f.is_file()])
    if args.limit:
        crops = crops[:args.limit]

    print(f"Reviewing {len(crops)} crops from {crop_dir}")

    tp = 0
    fp = 0
    errors = 0
    results = []

    for i, crop_path in enumerate(crops):
        is_frisbee, raw = vlm_is_frisbee(str(crop_path))
        label = "tp" if is_frisbee else "fp" if is_frisbee is False else "error"
        results.append((crop_path.name, label, raw))

        if is_frisbee is True:
            tp += 1
        elif is_frisbee is False:
            fp += 1
        else:
            errors += 1

        if (i + 1) % 10 == 0 or i == len(crops) - 1:
            total = tp + fp + errors
            fp_rate = fp / total * 100 if total else 0
            print(f"  [{i+1}/{len(crops)}] TP={tp} FP={fp} Err={errors} FP_rate={fp_rate:.1f}%")

        time.sleep(0.1)

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "vlm_response"])
        for row in results:
            writer.writerow(row)

    total = tp + fp + errors
    fp_rate = fp / total * 100 if total else 0
    print(f"\n=== VLM Review Results ===")
    print(f"Total: {total}")
    print(f"TP (frisbee): {tp} ({tp/total*100:.1f}%)")
    print(f"FP (not frisbee): {fp} ({fp_rate:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Saved to {output}")

    if fp_rate < 20:
        print("\n✓ FP rate < 20% — PASS! Proceed to Phase 2.")
    elif fp_rate < 50:
        print(f"\n~ FP rate {fp_rate:.1f}% — improved but not <20%. Consider more hard negatives.")
    else:
        print(f"\n✗ FP rate {fp_rate:.1f}% — still high. Need more aggressive approach.")


if __name__ == "__main__":
    main()
