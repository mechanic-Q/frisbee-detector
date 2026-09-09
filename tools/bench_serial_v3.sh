#!/bin/bash
# 串行恢复队列 v3：等 GPU 空闲（GUI 会话 pipeline 完成后）→ 补齐全部剩余任务。
# 任务清单：11s 重训 → D-FINE-S 训练 → 统一评测（v8s/11s/26s/rfdetr 四模型+FPS）→ prod 断点恢复。
# 纪律：全程 MLFLOW env + flock 锁 + wait_gpu_free 守卫（杜绝多进程并发崩溃）。
set -u
cd /mnt/e/frisbee-detector
export MLFLOW_ALLOW_FILE_STORE=true
LOG=data/bili_final_test
PROD_WEIGHTS=/home/lmr/comfy/ComfyUI/runs/detect/runs/bili_prod_v8sp2/weights

wait_gpu_free () {
  local who="$1"
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
    echo "[$(date +%H:%M:%S)] $who: GPU 占用中（GUI/其他会话），等待..." >> "$LOG/bench_status.log"
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
  local rc=$?
  local D=/home/lmr/comfy/ComfyUI/runs/detect/runs/"bench_$NAME"
  if [ $rc -eq 0 ] && [ -s "$D/weights/best.pt" ]; then
    bash tools/collect_weights.sh bench_"$NAME"
    echo "[$(date +%H:%M:%S)] trained bench_$NAME OK" >> "$LOG/bench_status.log"
  else
    echo "[$(date +%H:%M:%S)] bench_$NAME FAILED (rc=$rc)" >> "$LOG/bench_status.log"
  fi
}

echo "[$(date +%H:%M:%S)] bench-serial-v3 start（等 GUI 会话 GPU 任务完成）" >> "$LOG/bench_status.log"

# 1) 11s 重训（上次 OOM 崩溃）
train_one yolo11s.pt 11s

# 2) D-FINE-S 训练（WSL 路径已修正）
wait_gpu_free "dfine"
bash tools/gpu_run.sh bench-dfine \
  python3 D-FINE/train.py -c D-FINE/configs/dfine/dfine_hgnetv2_s_frisbee.yml \
    --use-amp --seed=42 \
    -t /mnt/e/frisbee-detector/data/bili_final_test/dfine_s_coco.pth \
  > "$LOG/train_bench_dfine.log" 2>&1
D=/mnt/e/frisbee-detector/D-FINE/output/dfine_hgnetv2_s_frisbee
if [ -s "$D/checkpoint_best_ema.pth" ]; then
  echo "[$(date +%H:%M:%S)] trained bench-dfine OK" >> "$LOG/bench_status.log"
else
  echo "[$(date +%H:%M:%S)] bench-dfine FAILED" >> "$LOG/bench_status.log"
fi

# 3) 统一评测：v8s/11s/26s/rfdetr 四模型 + FPS（rfdetr 已训完）
wait_gpu_free "eval"
python3 tools/bench_eval.py > "$LOG/bench_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] eval done" >> "$LOG/bench_status.log"

# 4) prod 断点恢复（makeup 已把 deferred 改回 last.pt；若在则恢复）
if [ -f "$PROD_WEIGHTS/last.pt" ] || [ -f "$PROD_WEIGHTS/last.pt.deferred" ]; then
  [ -f "$PROD_WEIGHTS/last.pt.deferred" ] && mv "$PROD_WEIGHTS/last.pt.deferred" "$PROD_WEIGHTS/last.pt"
  wait_gpu_free "prod-resume"
  bash tools/gpu_run.sh prod-resume \
    python3 -c "from ultralytics import YOLO; YOLO(r'$PROD_WEIGHTS/last.pt').train(resume=True)" \
    > "$LOG/train_bili_prod_resume.log" 2>&1
  echo "[$(date +%H:%M:%S)] prod resumed" >> "$LOG/bench_status.log"
fi
echo "[$(date +%H:%M:%S)] bench-serial-v3 ALL DONE" >> "$LOG/bench_status.log"
