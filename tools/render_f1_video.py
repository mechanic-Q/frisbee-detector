#!/usr/bin/env python3
"""F1 阶段成果可视化：tracks.json + 原视频 → 带 status 配色的叠加视频（mp4）。

配色（盘）：tracking=绿 / predicting=黄 / gated=红 / rejected=紫 / raw(无融合)=青
球员框：淡灰半透明。左上角状态计数条。

用法:
    python tools/render_f1_video.py --tracks results/f1_matrix/c3_shadow_fusion_segcal/tracks.json \
        --video movie/25866279684-1-192_55-56min.mp4 --out results/f1_matrix/c3_overlay.mp4 \
        [--max-frames 300] [--fps 25]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

STATUS_COLOR = {
    "tracking":   (0, 220, 0),     # 绿
    "predicting": (0, 200, 240),   # 黄
    "gated":      (0, 0, 255),     # 红
    "rejected":   (200, 0, 255),   # 紫
    "raw":        (240, 160, 0),   # 青（无融合模式）
}
STATUS_CN = {"tracking": "TRACK", "predicting": "PRED", "gated": "GATE",
             "rejected": "REJ", "raw": "RAW"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--show-players", action="store_true", default=True)
    args = ap.parse_args()

    doc = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    frames: dict = doc.get("frames", {})
    disc: dict = doc.get("disc_frames", {})
    fps = args.fps or doc.get("fps") or 25.0
    w, h = doc.get("width", 1280), doc.get("height", 720)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), args.video
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    status_counter: Counter = Counter()
    fi = -1
    while fi < args.max_frames - 1:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        key = str(fi)

        # 球员框（淡灰）
        if args.show_players:
            for d in frames.get(key, []):
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)

        # 盘框（status 配色 + 状态字）
        d = disc.get(key)
        if d is not None:
            st = d.get("status", "raw")
            status_counter[st] += 1
            c = STATUS_COLOR.get(st, (240, 160, 0))
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), c, 2)
            label = f"{STATUS_CN.get(st, st)}"
            if d.get("speed_ms") is not None:
                label += f" {d['speed_ms']:.0f}m/s"
            if d.get("d2") is not None:
                label += f" d2={d['d2']:.0f}"
            ty = max(y1 - 22, 14)
            cv2.putText(frame, label, (x1 - 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, c, 2, cv2.LINE_AA)
            # tracking 画短轨迹尾迹（前 10 帧 tracking 点）
            trail = []
            for back in range(1, 11):
                pd = disc.get(str(fi - back))
                if pd and pd.get("status") == "tracking":
                    trail.append((int(pd["cx"]), int(pd["cy"])))
            for a, b in zip(trail, trail[1:]):
                cv2.line(frame, a, b, c, 2)

        # 状态计数条
        bar = f"f{fi}  " + "  ".join(f"{STATUS_CN[k]}:{v}" for k, v in sorted(status_counter.items()))
        cv2.rectangle(frame, (0, 0), (w, 26), (20, 20, 20), -1)
        cv2.putText(frame, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    size_kb = out.stat().st_size // 1024
    print(f"saved: {out} ({fi + 1} frames, {size_kb} KB)")
    print("status counts:", dict(status_counter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
