#!/usr/bin/env python3
"""v3.5 bbox 负样本形态实验队列: v8s-P2@1280/100ep on merged_v35 → 金标准推理 → --gt-v2 终评。"""
import subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MARK = ROOT / "results/race_v35/markers"; LOGD = ROOT / "results/race_v35/logs"
PY = sys.executable

def run(name, cmd):
    mark = MARK / f"{name}.done"
    if mark.exists():
        print(f"[q] {name} skip", flush=True); return
    log = open(LOGD / f"{name}.log", "w", encoding="utf-8")
    print(f"[q] START {name}", flush=True)
    rc = subprocess.call([PY, str(ROOT/"tools/gpu_run.py"), name, *map(str, cmd)],
                         cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if rc == 0:
        mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        print(f"[q] DONE {name}", flush=True)
    else:
        print(f"[q] FAIL {name} exit={rc}", flush=True)

run("train_v8sp2_v35", [PY, "models/train.py", "--data", "configs/frisbee_merged_v35.yaml",
                        "--box", "5", "--epochs", "100", "--name", "race_v8sp2_v35",
                        "--workers", "2", "--batch", "2"])
run("golden_infer_v35", [PY, "tools/golden_inference.py", "--models", "race_v8sp2_v35"])
run("metrics_v2_v35", [PY, "tools/golden_metrics.py", "--gt-v2"])
print("[race_v35] ALL DONE", flush=True)
