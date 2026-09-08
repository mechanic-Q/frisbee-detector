"""合成场数据生成 CLI（解耦后的编排入口）。
预览: python3 tools/render_synth_field.py --count 12 --out results/synth_preview
批量: python3 tools/render_synth_field.py --count 100000 --out data/synth_field
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_field import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--out", type=str, default="results/synth_preview")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.count):
        img, meta = render.render_one(rng)
        stem = f"synth_{i + 1:06d}"
        cv2.imwrite(str(out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
        if (i + 1) % 100 == 0 or i + 1 == args.count:
            print(f"{i + 1}/{args.count}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
