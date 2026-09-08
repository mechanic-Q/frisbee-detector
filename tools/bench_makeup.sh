#!/bin/bash
# 基准补测队列 v2：等主队列(11s→rfdetr→eval)结束后，补跑因 GUI 抢卡崩溃的 11s/26s/v8s，再刷新统一对比表。
# 最后自动恢复生产级重训（断点 last.pt.deferred 改回后断点续训）。
set -u
cd /mnt/e/frisbee-detector
LOG=data/bili_final_test
PROD_WEIGHTS=/home/lmr/comfy/ComfyUI/runs/detect/runs/bili_prod_v8sp2/weights

# 等主队列完全结束（bench_rest 写入 ALL DONE）
while ! grep -q "bench ALL DONE" "$LOG/bench_status.log" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] makeup start (11s/26s/v8s 补测)" >> "$LOG/bench_status.log"

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

train_one yolo11s.pt 11s
train_one yolo26s.pt 26s
train_one yolov8s.pt v8s

echo "[$(date +%H:%M:%S)] makeup re-eval (刷新含 11s/26s/v8s 的统一对比表)" >> "$LOG/bench_status.log"
python3 tools/bench_eval.py > "$LOG/bench_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] makeup re-eval done" >> "$LOG/bench_status.log"

# 恢复生产级重训断点续训
if [ -f "$PROD_WEIGHTS/last.pt.deferred" ]; then
  mv "$PROD_WEIGHTS/last.pt.deferred" "$PROD_WEIGHTS/last.pt"
  bash tools/gpu_run.sh prod-resume \
    yolo train resume model="$PROD_WEIGHTS/last.pt" \
    > "$LOG/train_bili_prod_resume.log" 2>&1
  echo "[$(date +%H:%M:%S)] prod resumed" >> "$LOG/bench_status.log"
fi
echo "[$(date +%H:%M:%S)] makeup ALL DONE" >> "$LOG/bench_status.log"
