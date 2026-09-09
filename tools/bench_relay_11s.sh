#!/bin/bash
# 接力队列：等 pitch_kp 训练完成 → 跑 11s 重训修复版 → 刷新 11s 评测
set -u
cd /mnt/e/frisbee-detector
export MLFLOW_ALLOW_FILE_STORE=true
LOG=data/bili_final_test

# 等 pitch_kp 完成（its weights/best.pt 出现且 30 分钟无更新，或 results.csv 出现 60 行）
while true; do
  n=$(wc -l < /mnt/e/frisbee-detector/runs/detect/pitch_kp/results.csv 2>/dev/null || echo 0)
  if [ "$n" -ge 60 ]; then break; fi
  sleep 120
done
echo "[$(date +%H:%M:%S)] pitch_kp 完成，开始 11s 重训" >> "$LOG/bench_status.log"

wait_gpu_free () {
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
    echo "[$(date +%H:%M:%S)] $1: 等待 GPU..." >> "$LOG/bench_status.log"; sleep 60
  done
}

wait_gpu_free "bench-11s-re"
bash tools/gpu_run.sh bench-11s-re \
  python3 models/train.py --data configs/frisbee_merged_v2.yaml --model yolo11s.pt \
    --box 5 --epochs 30 --patience 8 --close-mosaic 5 --name bench_11s --no-plots \
    --cache --workers 4 \
  > "$LOG/train_bench_11s.log" 2>&1
rc=$?
D=/home/lmr/comfy/ComfyUI/runs/detect/runs/bench_11s
if [ $rc -eq 0 ] && [ -s "$D/weights/best.pt" ]; then
  bash tools/collect_weights.sh bench_11s
  echo "[$(date +%H:%M:%S)] retrained bench_11s OK" >> "$LOG/bench_status.log"
else
  echo "[$(date +%H:%M:%S)] bench_11s retrain FAILED (rc=$rc)" >> "$LOG/bench_status.log"
  exit 1
fi

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
       "weights": str(W), "note": "retrained on windows-native wsl"}
s = json.loads((ROOT / "results/bench_summary.json").read_text())
s["yolo11s"] = row
(ROOT / "results/bench_summary.json").write_text(json.dumps(s, indent=1))
print("UPDATED 11s ->", row)
PYEOF
echo "[$(date +%H:%M:%S)] 11s re-eval done, bench_summary updated" >> "$LOG/bench_status.log"
