#!/bin/bash
# 等两条车道都完成 → 统一评测 → 恢复生产级重训
set -u
cd /mnt/e/frisbee-detector
LOG=data/bili_final_test
PROD_LAST=/home/lmr/comfy/ComfyUI/runs/detect/runs/bili_prod_v8sp2/weights/last.pt

while [ ! -f "$LOG/.lane1_done" ] || [ ! -f "$LOG/.lane2_done" ]; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] 两车道完成，开始统一评测" >> "$LOG/bench_status.log"
python3 tools/bench_eval.py > "$LOG/bench_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] eval done" >> "$LOG/bench_status.log"

if [ -f "$PROD_LAST" ]; then
  bash tools/gpu_run.sh prod-resume \
    yolo train resume model="$PROD_LAST" \
    > "$LOG/train_bili_prod_resume.log" 2>&1
  echo "[$(date +%H:%M:%S)] prod resumed" >> "$LOG/bench_status.log"
fi
echo "[$(date +%H:%M:%S)] bench ALL DONE" >> "$LOG/bench_status.log"
