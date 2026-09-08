#!/bin/bash
# 基准双车道流水线 · 车道1：v8s → 26s
set -u
cd /mnt/e/frisbee-detector
export GPU_LOCK=/tmp/frisbee_gpu_lane1.lock
LOG=data/bili_final_test

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
train_one yolo26s.pt 26s
touch "$LOG/.lane1_done"
