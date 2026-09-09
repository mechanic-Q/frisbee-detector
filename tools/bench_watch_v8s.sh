#!/bin/bash
# v8s 接力 watcher：等 26s 完成标记 → 立即接管（阻止 bench_final 启动无 cache 的 v8s）
# → 用 --cache ram + workers 4 跑 v8s → 统一评测 → 恢复生产训练
set -u
cd /mnt/e/frisbee-detector
LOG=data/bili_final_test
PROD_WEIGHTS=/home/lmr/comfy/ComfyUI/runs/detect/runs/bili_prod_v8sp2/weights

while ! grep -q "trained bench_26s" "$LOG/bench_status.log" 2>/dev/null; do sleep 15; done
tmux kill-session -t bench-final 2>/dev/null
wait_gpu_free () {
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
    echo "[$(date +%H:%M:%S)] 等待 GPU 空闲..." >> "$LOG/bench_status.log"; sleep 30
  done
}
echo "[$(date +%H:%M:%S)] watcher 接管：v8s(cache) 开始" >> "$LOG/bench_status.log"
wait_gpu_free
bash tools/gpu_run.sh bench-v8s \
  python3 models/train.py --data configs/frisbee_merged_v2.yaml --model yolov8s.pt \
    --box 5 --epochs 30 --patience 8 --close-mosaic 5 --name bench_v8s --no-plots \
    --cache ram --workers 4 \
  > "$LOG/train_bench_v8s.log" 2>&1
D=/home/lmr/comfy/ComfyUI/runs/detect/runs/bench_v8s
bash tools/collect_weights.sh bench_v8s
echo "[$(date +%H:%M:%S)] trained bench_v8s(cache)" >> "$LOG/bench_status.log"

wait_gpu_free
echo "[$(date +%H:%M:%S)] unified eval" >> "$LOG/bench_status.log"
python3 tools/bench_eval.py > "$LOG/bench_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] eval done" >> "$LOG/bench_status.log"

if [ -f "$PROD_WEIGHTS/last.pt" ]; then
  wait_gpu_free
  bash tools/gpu_run.sh prod-resume \
    python3 -c "from ultralytics import YOLO; YOLO(r'$PROD_WEIGHTS/last.pt').train(resume=True)" \
    > "$LOG/train_bili_prod_resume.log" 2>&1
  echo "[$(date +%H:%M:%S)] prod resumed" >> "$LOG/bench_status.log"
fi
echo "[$(date +%H:%M:%S)] ALL DONE (含 v8s cache 接力)" >> "$LOG/bench_status.log"
