#!/bin/bash
# 串行收尾队列：等 lane1 的 26s 完成后，串行补跑崩溃的三个基准 + 评测 + 恢复生产训练
# （WSL2 GPU-PV 多进程 CUDA 并发不稳定——实测两车道/三车道均互崩，故全部回退串行+全局锁）
set -u
cd /mnt/e/frisbee-detector
unset GPU_LOCK
LOG=data/bili_final_test
PROD_LAST=/home/lmr/comfy/ComfyUI/runs/detect/runs/bili_prod_v8sp2/weights/last.pt

while [ ! -f "$LOG/.lane1_done" ]; do sleep 60; done

train_one () {
  local WEIGHTS=$1
  local NAME=$2
  bash tools/gpu_run.sh "bench-$NAME" \
    python3 models/train.py --data configs/frisbee_merged_v2.yaml --model "$WEIGHTS" \
      --box 5 --epochs 30 --patience 8 --close-mosaic 5 --name "bench_$NAME" --no-plots \
    > "$LOG/train_bench_$NAME.log" 2>&1
  local D=/home/lmr/comfy/ComfyUI/runs/detect/runs/"bench_$NAME"
  if [ -d "$D" ]; then rm -rf "runs/detect/bench_$NAME"; cp -r "$D" "runs/detect/bench_$NAME"; fi
  echo "[$(date +%H:%M:%S)] trained bench_$NAME" >> "$LOG/bench_status.log"
}

train_one yolov8s.pt v8s
train_one yolo11s.pt 11s
bash tools/gpu_run.sh bench-rfdetr python3 tools/train_rfdetr.py \
  > "$LOG/train_bench_rfdetr.log" 2>&1
echo "[$(date +%H:%M:%S)] trained bench_rfdetr" >> "$LOG/bench_status.log"

echo "[$(date +%H:%M:%S)] unified eval" >> "$LOG/bench_status.log"
python3 tools/bench_eval.py > "$LOG/bench_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] eval done" >> "$LOG/bench_status.log"

if [ -f "$PROD_LAST" ]; then
  bash tools/gpu_run.sh prod-resume \
    yolo train resume model="$PROD_LAST" \
    > "$LOG/train_bili_prod_resume.log" 2>&1
  echo "[$(date +%H:%M:%S)] prod resumed" >> "$LOG/bench_status.log"
fi
echo "[$(date +%H:%M:%S)] bench ALL DONE" >> "$LOG/bench_status.log"
