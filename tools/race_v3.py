#!/usr/bin/env python3
"""v3 成长性赛马队列（Windows 原生，串行 GPU）：三家架构喂同一套 merged_v3 数据，
各用各自已验证的完整成长配方，训练完自动跑 test + 金标准 v2 双标尺评测。

Horse A: v8s-P2    @1280 100ep  (ultralytics, models/train.py)
Horse B: D-FINE-S  @640  72ep   (D-FINE repo 官方配方)
每阶段完成写标记文件（results/race_v3/markers/），阶段失败记日志并继续。
运行: python tools/race_v3.py            （整体串行，内部经 tools/gpu_run.py 排队锁）
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = ROOT / "results/race_v3/markers"
LOGD = ROOT / "results/race_v3/logs"
PY = sys.executable


def run_stage(name: str, cmd: list[str], cwd: Path = ROOT) -> bool:
    mark = MARK / f"{name}.done"
    if mark.exists():
        print(f"[race] {name} 已完成，跳过", flush=True)
        return True
    log = open(LOGD / f"{name}.log", "w", encoding="utf-8")
    print(f"[race] START {name}: {' '.join(map(str, cmd))}", flush=True)
    # 经 gpu_run.py 排队锁（与其他 GPU 任务互斥）
    rc = subprocess.call([PY, str(ROOT / "tools/gpu_run.py"), name, *map(str, cmd)],
                         cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if rc == 0:
        mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        print(f"[race] DONE {name}", flush=True)
        return True
    print(f"[race] FAIL {name} (exit={rc})，继续下一阶段", flush=True)
    return False


def main() -> int:
    MARK.mkdir(parents=True, exist_ok=True)
    LOGD.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Horse A: v8s-P2 @1280 100ep on v3 ──
    run_stage("train_v8sp2_v3",
              [PY, "models/train.py", "--data", "configs/frisbee_merged_v3.yaml",
               "--box", "5", "--epochs", "100", "--name", "race_v8sp2_v3",
               "--workers", "2", "--batch", "2"])

    # ── Horse B: D-FINE-S @640 72ep(官方收敛档) on v3 ──
    dfine_cfg = "configs/dfine/dfine_hgnetv2_s_frisbee_v3.yml"
    run_stage("train_dfine_v3",
              [PY, "train.py", "-c", dfine_cfg,
               "-t", "E:/frisbee-detector/data/bili_final_test/dfine_s_coco.pth"],
              cwd=ROOT / "D-FINE")

    # ── 评测：金标准 100 帧推理（新权重）──
    run_stage("golden_infer_v8sp2_v3",
              [PY, "tools/golden_inference.py", "--models", "race_v8sp2_v3"])
    v3_ckpt = ROOT / "data/bili_final_test/dfine_v3_out/best_stg1.pth"
    if v3_ckpt.exists():
        run_stage("golden_infer_dfine_v3",
                  [PY, "tools/golden_inference_detr.py",
                   "--dfine-ckpt", str(v3_ckpt), "--dfine-name", "dfine_s_v3",
                   "--skip-rfdetr"])

    # ── 终评：金标准 v2 指标重算（含新模型 json）──
    run_stage("golden_metrics_v2", [PY, "tools/golden_metrics.py", "--gt-v2"])

    print(f"[race] ALL STAGES FINISHED in {(time.time()-t0)/3600:.1f}h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
