"""把飞盘检测 CSV 的高置信度检测画回视频帧做视觉验证。
在 WSL 运行: python3 data/bili_final_test/viz_frisbee_dets.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/mnt/e/frisbee-detector")
CLIP = ROOT / "data/bili_final_test/testclip_60_120s.mp4"
CSV = ROOT / "data/bili_final_test/out/frisbee_sahi_clip.csv"
OUT = ROOT / "data/bili_final_test/out"

by_frame = defaultdict(list)
with open(CSV) as f:
    for row in csv.DictReader(f):
        by_frame[int(row["frame"])].append(row)

# 按置信度取 top10（不同帧）
rows = sorted((r for rs in by_frame.values() for r in rs), key=lambda r: -float(r["conf"]))
picked, seen = [], set()
for r in rows:
    fr = int(r["frame"])
    if fr not in seen:
        picked.append(fr)
        seen.add(fr)
    if len(picked) >= 10:
        break

cap = cv2.VideoCapture(str(CLIP))
i = 0
target = set(picked)
tiles = {}
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i in target:
        for r in by_frame[i]:
            x1, y1, x2, y2 = (int(float(r[k])) for k in ("minx", "miny", "maxx", "maxy"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{float(r["conf"]):.2f}', (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, f"processed_frame {i}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles[i] = frame
    i += 1
cap.release()

tiles_sorted = [tiles[k] for k in sorted(tiles)]
half = (len(tiles_sorted) + 1) // 2
for part, group in enumerate((tiles_sorted[:half], tiles_sorted[half:])):
    if not group:
        continue
    canvas = []
    for start in range(0, len(group), 2):
        pair = group[start:start + 2]
        while len(pair) < 2:
            pair.append(255 * np.ones_like(group[0]))
        canvas.append(np.hstack(pair))
    grid = np.vstack(canvas)
    grid = cv2.resize(grid, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / f"frisbee_top_dets_part{part}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved part", part)
print("picked frames:", sorted(tiles))
