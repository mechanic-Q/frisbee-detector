#!/bin/bash
# 检测器架构基准队列：官方 COCO 默认权重同起跑线，同配方微调 30 epochs，之后统一评测。
# 全部经过 gpu_run.sh flock 排队；队列末尾恢复被暂停的生产级重训。
set -u
cd /mnt/e/frisbee-detector
LOG=data/bili_final_test
PROD_LAST=/home/lmr/comfy/ComfyUI/runs/detect/runs/bili_prod_v8sp2/weights/last.pt

train_one () {
  local WEIGHTS=$1
  local NAME=$2
  bash tools/gpu_run.sh "bench-$NAME" \
    python3 models/train.py --data configs/frisbee_merged_v2.yaml --model "$WEIGHTS" \
      --box 5 --epochs 30 --patience 8 --close-mosaic 5 --name "bench_$NAME" --no-plots \
    > "$LOG/train_bench_$NAME.log" 2>&1
  # 统一从 ComfyUI 路径复制回项目
  local D=/home/lmr/comfy/ComfyUI/runs/detect/runs/"bench_$NAME"
  if [ -d "$D" ]; then rm -rf "runs/detect/bench_$NAME"; cp -r "$D" "runs/detect/bench_$NAME"; fi
  echo "[$(date +%H:%M:%S)] trained bench_$NAME" >> "$LOG/bench_status.log"
}

echo "[$(date +%H:%M:%S)] bench queue start" >> "$LOG/bench_status.log"
train_one yolov8s.pt v8s
train_one yolo11s.pt 11s
train_one yolo26s.pt 26s
bash tools/gpu_run.sh bench-rfdetr python3 tools/train_rfdetr.py \
  > "$LOG/train_bench_rfdetr.log" 2>&1
echo "[$(date +%H:%M:%S)] trained bench_rfdetr" >> "$LOG/bench_status.log"

echo "[$(date +%H:%M:%S)] unified eval" >> "$LOG/bench_status.log"
python3 tools/bench_eval.py > "$LOG/bench_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] eval done" >> "$LOG/bench_status.log"

# 恢复生产级重训（从 last.pt 续训剩余 epochs）
if [ -f "$PROD_LAST" ]; then
  bash tools/gpu_run.sh prod-resume \
    yolo train resume model="$PROD_LAST" \
    > "$LOG/train_bili_prod_resume.log" 2>&1
  echo "[$(date +%H:%M:%S)] prod resumed" >> "$LOG/bench_status.log"
fi
echo "[$(date +%H:%M:%S)] bench queue ALL DONE" >> "$LOG/bench_status.log"
