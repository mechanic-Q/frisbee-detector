#!/bin/bash
# D-FINE-S 基准训练启动器（Windows 侧排队）：等 fix11s 完成标记 → 走 GPU 空闲守卫 → 训练 30ep
set -u
cd /mnt/e/frisbee-detector/D-FINE
LOG=/mnt/e/frisbee-detector/data/bili_final_test

# 等 fix11s 先完成（它在前面排队）
while ! grep -q "11s re-eval done" "$LOG/bench_status.log" 2>/dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] D-FINE 开始（fix11s 已完成）" >> "$LOG/bench_status.log"

while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
  echo "[$(date +%H:%M:%S)] D-FINE: 等 GPU 空闲" >> "$LOG/bench_status.log"; sleep 60
done

export MLFLOW_ALLOW_FILE_STORE=true
python train.py -c configs/dfine/dfine_hgnetv2_s_frisbee.yml \
  --use-amp --seed=42 \
  -t /mnt/e/frisbee-detector/data/bili_final_test/dfine_s_coco.pth \
  > "$LOG/train_bench_dfine.log" 2>&1
echo "[$(date +%H:%M:%S)] trained dfine_s (exit=$?)" >> "$LOG/bench_status.log"
