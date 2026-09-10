#!/bin/bash
# 基准训练评测：固定标尺(原 merged test mAP) + 1080P 草纹误检复测
set -x
cd /mnt/e/frisbee-detector
LOG=data/bili_final_test
CLIP=data/bili_final_test/testclip1080_60_120s.mp4

echo "[$(date)] eval start" >> "$LOG/eval_status.log"

eval_one () {
  local NAME=$1
  local W=runs/detect/$NAME/weights/best.pt
  [ -f "$W" ] || { echo "missing $W" >> "$LOG/eval_status.log"; return; }
  # 1) 固定标尺：原 frisbee_merged 测试集
  python3 - <<PYEOF > "$LOG/eval_${NAME}_map.log" 2>&1
from ultralytics import YOLO
m = YOLO("$W")
r = m.val(data="configs/frisbee_merged.yaml", split="test", imgsz=1280, batch=2, verbose=True)
print("SUMMARY", "$NAME", "mAP50", r.box.map50, "mAP50-95", r.box.map, "P", r.box.mp, "R", r.box.mr)
PYEOF
  # 2) 1080P 草纹误检复测（SAHI, 与旧模型同条件）
  python3 inference/predict_video.py --video "$CLIP" --model "$W" --conf 0.35 --sahi \
    --frame-skip 2 --output-csv "$LOG/frisbee1080_${NAME}.csv" > "$LOG/eval_${NAME}_sahi.log" 2>&1
  echo "[$(date)] eval done $NAME" >> "$LOG/eval_status.log"
}

eval_one bili_bench_v8sp2
eval_one bili_bench_26sp2

# 3) 基线对照：现役 p2_shadow_v1 在同一固定标尺上的 mAP
python3 - <<'PYEOF' > "$LOG/eval_baseline_shadow_v1_map.log" 2>&1
from ultralytics import YOLO
m = YOLO("runs/detect/frisbee_det_p2_shadow_v1/weights/best.pt")
r = m.val(data="configs/frisbee_merged.yaml", split="test", imgsz=1280, batch=2, verbose=True)
print("SUMMARY shadow_v1 mAP50", r.box.map50, "mAP50-95", r.box.map, "P", r.box.mp, "R", r.box.mr)
PYEOF
echo "[$(date)] baseline eval done" >> "$LOG/eval_status.log"
echo "[$(date)] eval ALL DONE" >> "$LOG/eval_status.log"
