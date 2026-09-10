#!/bin/bash
# 架构基准训练：v8s-p2 vs yolo26s-p2，同数据(v2)同超参
set -x
cd /mnt/e/frisbee-detector
LOG=data/bili_final_test

train_one () {
  local MODEL=$1
  local NAME=$2
  python3 models/train.py \
    --data configs/frisbee_merged_v2.yaml \
    --model "$MODEL" \
    --box 5 --epochs 30 --patience 8 --close-mosaic 5 \
    --name "$NAME" --no-plots > "$LOG/train_${NAME}.log" 2>&1
  # 双层目录 bug 修复
  if [ -d "runs/detect/runs/detect/$NAME" ]; then
    rm -rf "runs/detect/$NAME"
    mv "runs/detect/runs/detect/$NAME" "runs/detect/$NAME"
  fi
  echo "[$(date)] done $NAME" >> "$LOG/bench_status.log"
}

echo "[$(date)] bench start" >> "$LOG/bench_status.log"
train_one "yolov8s-p2.yaml"  "bili_bench_v8sp2"
train_one "yolo26s-p2.yaml"  "bili_bench_26sp2"
echo "[$(date)] bench ALL DONE" >> "$LOG/bench_status.log"
