#!/bin/bash
# GPU 排队执行器：flock 保证同一时刻只有一个 GPU 任务在跑（跨 worktree 共享同一把锁）。
# 用法: bash tools/gpu_run.sh <task_name> <command...>
# 例:   bash tools/gpu_run.sh funasr-asr python3 tools/asr_run.py
set -u
LOCK_FILE="${GPU_LOCK:-/tmp/frisbee_gpu.lock}"
TASK="$1"; shift
mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
echo "[gpu-queue] '$TASK' 排队等待 GPU ... $(date +%H:%M:%S)"
flock 9
echo "[gpu-queue] '$TASK' 获得 GPU $(date +%H:%M:%S)"
"$@"
rc=$?
echo "[gpu-queue] '$TASK' 释放 GPU (exit=$rc) $(date +%H:%M:%S)"
exit $rc
