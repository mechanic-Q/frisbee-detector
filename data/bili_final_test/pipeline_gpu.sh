#!/bin/bash
# 1080P 高清管线三连：飞盘 SAHI 重测 / 球员抽帧+预标 / 硬负样本候选收集
set -e
cd /mnt/e/frisbee-detector
OUT=data/bili_final_test
mkdir -p "$OUT/player_frames" "$OUT/player_labels" "$OUT/hardneg_candidates"

echo "=== [A/5] 切 1080P 60-120s 测试片段 ==="
ffmpeg -v error -ss 60 -t 60 -i "movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4" \
  -c:v libx264 -preset veryfast -crf 20 -an "$OUT/testclip1080_60_120s.mp4" -y

echo "=== [B/5] 飞盘 SAHI 1080P 重测 ==="
python3 inference/predict_video.py --video "$OUT/testclip1080_60_120s.mp4" --conf 0.35 --sahi \
  --frame-skip 2 --output-csv "$OUT/frisbee_sahi_clip1080.csv" 2>&1 | tail -12

echo "=== [C/5] 全视频 2 秒抽帧（球员预标注底图） ==="
ffmpeg -v error -i "movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4" \
  -vf "select=not(mod(n\,60))" -vsync vfr -q:v 2 "$OUT/player_frames/f_%06d.jpg" -y
ls "$OUT/player_frames" | wc -l

echo "=== [D/5] yolo26x 球员预标注 ==="
python3 - <<'PYEOF'
from ultralytics import YOLO
from pathlib import Path
import json
ROOT = Path("/mnt/e/frisbee-detector")
frames = sorted((ROOT/"data/bili_final_test/player_frames").glob("f_*.jpg"))
m = YOLO(str(ROOT/"yolo26x.pt"))
n_img = n_box = 0
for i, fp in enumerate(frames):
    r = m.predict(str(fp), conf=0.30, imgsz=1280, classes=[0], verbose=False)[0]
    h, w = r.orig_shape
    lines = []
    if r.boxes is not None:
        for b in r.boxes.xywhn.cpu().numpy():
            lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(*b))
    (ROOT/"data/bili_final_test/player_labels"/(fp.stem+".txt")).write_text("\n".join(lines))
    n_img += 1; n_box += len(lines)
    if i % 300 == 0: print(f"  {i}/{len(frames)}", flush=True)
print(f"prelabel done: {n_img} imgs, {n_box} boxes")
stats = {"imgs": n_img, "boxes": n_box, "model": "yolo26x", "conf": 0.30, "imgsz": 1280, "cls": "0=person"}
(ROOT/"data/bili_final_test/player_labels/_prelabel_stats.json").write_text(json.dumps(stats, indent=2))
PYEOF

echo "=== [E/5] 480P 全片飞盘误检裁剪收集（1fps） ==="
python3 - <<'PYEOF'
import cv2
from ultralytics import YOLO
from pathlib import Path
ROOT = Path("/mnt/e/frisbee-detector")
OUT = ROOT/"data/bili_final_test/hardneg_candidates"
m = YOLO(str(ROOT/"runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt"))
cap = cv2.VideoCapture(str(ROOT/"movie/2024城市飞盘决赛_北京VS沪蛙_480p.mp4"))
idx = 0; n_crop = 0
rows = ["crop_file,src_frame,x1,y1,x2,y2,conf"]
while True:
    ok, frame = cap.read()
    if not ok: break
    if idx % 30 == 0:  # 1fps
        r = m.predict(frame, conf=0.35, imgsz=1280, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = map(int, b)
                pad = max(12, int(0.15 * max(x2-x1, y2-y1)))
                H, W = frame.shape[:2]
                cx1, cy1, cx2, cy2 = max(x1-pad,0), max(y1-pad,0), min(x2+pad,W), min(y2+pad,H)
                crop = frame[cy1:cy2, cx1:cx2]
                name = f"crop_{n_crop:05d}.jpg"
                cv2.imwrite(str(OUT/name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(f"{name},{idx},{x1},{y1},{x2},{y2},{c:.3f}")
                n_crop += 1
    idx += 1
    if idx % 3000 == 0: print(f"  frame {idx}, crops {n_crop}", flush=True)
cap.release()
(OUT/"crops.csv").write_text("\n".join(rows))
print(f"hardneg candidates: {n_crop} crops")
PYEOF

echo "=== ALL DONE ==="
date
