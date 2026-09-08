#!/bin/bash
# 最终串行队列（严格单 CUDA 进程纪律）：
#   11s → 26s → v8s → rfdetr评测 → 统一评测 → 生产训练断点恢复
# 每阶段前等待 GPU 完全空闲（包括 GUI 会话等未走队列的占用者）。
set -u
cd /mnt/e/frisbee-detector
LOG=data/bili_final_test
PROD_WEIGHTS=/home/lmr/comfy/ComfyUI/runs/detect/runs/bili_prod_v8sp2/weights

wait_gpu_free () {
  local who="$1"
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
    echo "[$(date +%H:%M:%S)] $who: GPU 被占用，等待 60s..." >> "$LOG/bench_status.log"
    sleep 60
  done
}

train_one () {
  local WEIGHTS=$1
  local NAME=$2
  wait_gpu_free "bench-$NAME"
  bash tools/gpu_run.sh "bench-$NAME" \
    python3 models/train.py --data configs/frisbee_merged_v2.yaml --model "$WEIGHTS" \
      --box 5 --epochs 30 --patience 8 --close-mosaic 5 --name "bench_$NAME" --no-plots \
    > "$LOG/train_bench_$NAME.log" 2>&1
  local D=/home/lmr/comfy/ComfyUI/runs/detect/runs/"bench_$NAME"
  if [ -d "$D" ]; then rm -rf "runs/detect/bench_$NAME"; cp -r "$D" "runs/detect/bench_$NAME"; fi
  echo "[$(date +%H:%M:%S)] trained bench_$NAME" >> "$LOG/bench_status.log"
}

echo "[$(date +%H:%M:%S)] bench-final start" >> "$LOG/bench_status.log"
train_one yolo11s.pt 11s
train_one yolo26s.pt 26s
train_one yolov8s.pt v8s

# rfdetr 已训完，补齐其同标尺评测（在 bench_eval 内统一做）
wait_gpu_free "eval"
echo "[$(date +%H:%M:%S)] unified eval" >> "$LOG/bench_status.log"
python3 tools/bench_eval.py > "$LOG/bench_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] eval done" >> "$LOG/bench_status.log"

# 恢复生产级重训（python API，避开 yolo CLI 的 GreenSocket 崩溃）
if [ -f "$PROD_WEIGHTS/last.pt" ] || [ -f "$PROD_WEIGHTS/last.pt.deferred" ]; then
  [ -f "$PROD_WEIGHTS/last.pt.deferred" ] && mv "$PROD_WEIGHTS/last.pt.deferred" "$PROD_WEIGHTS/last.pt"
  wait_gpu_free "prod-resume"
  bash tools/gpu_run.sh prod-resume \
    python3 -c "from ultralytics import YOLO; YOLO(r'$PROD_WEIGHTS/last.pt').train(resume=True)" \
    > "$LOG/train_bili_prod_resume.log" 2>&1
  echo "[$(date +%H:%M:%S)] prod resumed" >> "$LOG/bench_status.log"
fi
echo "[$(date +%H:%M:%S)] bench-final ALL DONE" >> "$LOG/bench_status.log"
