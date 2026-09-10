"""E2 测试：COCO person 检测 + BoT-SORT 多目标跟踪，60s 测试片段。
在 WSL 运行: python3 data/bili_final_test/track_players.py
输出: 标注视频 + 每帧 CSV + ID 稳定性统计。
"""
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path("/mnt/e/frisbee-detector")
CLIP = str(ROOT / "data/bili_final_test/testclip_60_120s.mp4")
OUT = ROOT / "data/bili_final_test/out"
MODEL = str(ROOT / "yolo26x.pt")

# BoT-SORT：固定机位+摇镜，先用默认配置（gmc=sparseOptFlow 有助摇镜补偿）
model = YOLO(MODEL)
cap = cv2.VideoCapture(CLIP)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

writer = cv2.VideoWriter(str(OUT / "track_26x_botsort_60s.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

rows = []
id_first_seen = {}
t_start = time.time()
n_frames = 0
for res in model.track(source=CLIP, conf=0.25, imgsz=1280, classes=[0],
                       tracker="botsort.yaml", persist=True, stream=True, verbose=False):
    frame = res.orig_img
    ids = res.boxes.id.int().cpu().tolist() if res.boxes is not None and res.boxes.id is not None else []
    confs = res.boxes.conf.cpu().tolist() if res.boxes is not None else []
    for tid, c in zip(ids, confs):
        rows.append((n_frames, tid, round(c, 3)))
        if tid not in id_first_seen:
            id_first_seen[tid] = n_frames
    if res.boxes is not None:
        for box, tid in zip(res.boxes.xyxy.cpu().numpy(), ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
            cv2.putText(frame, f"#{tid}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"f{n_frames} objs={len(ids)}", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    writer.write(frame)
    n_frames += 1
    if n_frames % 300 == 0:
        print(f"frame {n_frames}, elapsed {time.time()-t_start:.0f}s", flush=True)
writer.release()

with open(OUT / "track_26x_botsort_60s.csv", "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["frame", "track_id", "conf"])
    wcsv.writerows(rows)

per_len = Counter()
by_id = defaultdict(int)
for _, tid, _ in rows:
    by_id[tid] += 1
for tid, n in by_id.items():
    per_len[n] += 1

print("=" * 50)
print(f"frames: {n_frames}, elapsed: {time.time()-t_start:.0f}s ({n_frames/(time.time()-t_start):.1f} fps)")
print(f"unique track IDs: {len(by_id)}  (场上一场累计人数+切分误差)")
print(f"frames with 0 tracks: {n_frames - len(set(r[0] for r in rows))}")
print(f"track length dist (frames: count) top: {per_len.most_common(8)}")
print(f"longest tracks: {sorted(by_id.values(), reverse=True)[:5]}")
print("saved -> out/track_26x_botsort_60s.mp4 / .csv")
