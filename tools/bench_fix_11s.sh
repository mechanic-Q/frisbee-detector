#!/bin/bash
# 11s 权重遗失补测：重训 30ep → 从 ComfyUI 复制权重 → 仅重评 11s 并更新 bench_summary.json
set -u
cd /mnt/e/frisbee-detector
export MLFLOW_ALLOW_FILE_STORE=true
LOG=data/bili_final_test

wait_gpu_free () {
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
    echo "[$(date +%H:%M:%S)] $1: 等待 GPU 空闲..." >> "$LOG/bench_status.log"; sleep 60
  done
}

wait_gpu_free "bench-11s-retrain"
bash tools/gpu_run.sh bench-11s-re \
  python3 models/train.py --data configs/frisbee_merged_v2.yaml --model yolo11s.pt \
    --box 5 --epochs 30 --patience 8 --close-mosaic 5 --name bench_11s --no-plots \
    --cache ram --workers 4 \
  > "$LOG/train_bench_11s.log" 2>&1
D=/home/lmr/comfy/ComfyUI/runs/detect/runs/bench_11s
if [ -d "$D" ]; then rm -rf "runs/detect/bench_11s"; cp -r "$D" "runs/detect/bench_11s"; fi
echo "[$(date +%H:%M:%S)] retrained bench_11s" >> "$LOG/bench_status.log"

wait_gpu_free "eval-11s"
python3 - <<'PYEOF' > "$LOG/eval_11s_only.log" 2>&1
import json
from pathlib import Path
from ultralytics import YOLO
ROOT = Path("/mnt/e/frisbee-detector")
W = ROOT / "runs/detect/bench_11s/weights/best.pt"
m = YOLO(str(W))
r = m.val(data="configs/frisbee_merged.yaml", split="test", imgsz=1280, batch=2, verbose=False)
import cv2, numpy as np, time
cap = cv2.VideoCapture(str(ROOT / "movie/2024城市飞盘俱乐部锦标赛『决赛』【北京大哥 VS 上海沪蛙】高清版3-1.mp4"))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = np.linspace(0, total - 1, 200).astype(int)
t0 = time.time(); n = 0
for k, i in enumerate(idxs):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
    if not ok: continue
    m.predict(f, imgsz=1280, conf=0.35, verbose=False)
    if k == 10: t0 = time.time(); n = 0
    n += 1
cap.release()
row = {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
       "P": round(float(r.box.mp), 4), "R": round(float(r.box.mr), 4),
       "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
       "fps_1080p": round(n / max(time.time() - t0, 1e-6), 1),
       "weights": str(W), "note": "retrained after weights loss; seed/recipe identical"}
s = json.loads((ROOT / "results/bench_summary.json").read_text())
s["yolo11s"] = row
(ROOT / "results/bench_summary.json").write_text(json.dumps(s, indent=1))
print("UPDATED 11s ->", row)
PYEOF
echo "[$(date +%H:%M:%S)] 11s re-eval done, bench_summary updated" >> "$LOG/bench_status.log"
