#!/bin/bash
# D-FINE 训练完成后的自动接力：test 评测 → prod 恢复 → 最终表
set -u
cd /mnt/e/frisbee-detector
export MLFLOW_ALLOW_FILE_STORE=true

wait_gpu_free () {
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
    sleep 60
  done
}

# 1) 等 D-FINE 完成
wait_gpu_free
echo "[$(date +%H:%M:%S)] GPU 空闲，开始评测" >> data/bili_final_test/after_dfine.log

# 2) D-FINE test 评测
python3 tools/eval_dfine_test.py > data/bili_final_test/eval_dfine_test.log 2>&1
echo "[$(date +%H:%M:%S)] dfine test eval done" >> data/bili_final_test/after_dfine.log

# 3) 11s test 评测
python3 -c "
from ultralytics import YOLO
m = YOLO('runs/detect/bench_11s/weights/best.pt')
r = m.val(data='configs/frisbee_merged_v2_win.yaml', split='test', imgsz=1280, batch=4, verbose=False)
print('11s TEST: mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f' % (r.box.map50, r.box.map, r.box.mp, r.box.mr))
" > data/bili_final_test/eval_11s_test.log 2>&1
echo "[$(date +%H:%M:%S)] 11s test eval done" >> data/bili_final_test/after_dfine.log

# 4) prod 恢复
PROD_LAST=runs/detect/bili_prod_v8sp2/weights/last.pt
if [ -f "$PROD_LAST" ]; then
  wait_gpu_free
  python3 -c "from ultralytics import YOLO; YOLO('$PROD_LAST').train(resume=True)" \
    > data/bili_final_test/prod_resume.log 2>&1
  echo "[$(date +%H:%M:%S)] prod resumed" >> data/bili_final_test/after_dfine.log
fi

echo "[$(date +%H:%M:%S)] after_dfine ALL DONE" >> data/bili_final_test/after_dfine.log
