#!/bin/bash
# 基准三车道流水线 · 车道3：RF-DETR（低显存配置 bs2@560，错峰并行）
set -u
cd /mnt/e/frisbee-detector
export GPU_LOCK=/tmp/frisbee_gpu_lane3.lock
export RF_BS=2
export RF_RES=576
LOG=data/bili_final_test

bash tools/gpu_run.sh bench-rfdetr python3 tools/train_rfdetr.py \
  > "$LOG/train_bench_rfdetr.log" 2>&1
echo "[$(date +%H:%M:%S)] trained bench_rfdetr(lane3)" >> "$LOG/bench_status.log"
touch "$LOG/.lane3_done"
