"""最终 GPU 编排器（极简直线版，串行自驱）：
11s 重训 → D-FINE test 评测 → prod 断点恢复。每步前 GPU 空闲守卫。
运行: python tools/bench_final_orchestrator.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data/bili_final_test"
PROD_LAST = ROOT / "runs/detect/bili_prod_v8sp2/weights/last.pt"
PY = sys.executable


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "final_orch.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_gpu_free(tag, timeout_min=480):
    t0 = time.time()
    while True:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            if not r.stdout.strip():
                return
        except Exception:
            return
        log(f"{tag}: GPU 占用，等 60s...")
        if time.time() - t0 > timeout_min * 60:
            return
        time.sleep(60)


def run(cmd, tag, timeout_s):
    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    log(f"{tag} START")
    try:
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env,
                           capture_output=True, text=True, timeout=timeout_s)
        out = (r.stdout or "")[-5000:]
        err = (r.stderr or "")[-5000:]
        (LOG_DIR / f"{tag}.out.log").write_text(out, encoding="utf-8")
        (LOG_DIR / f"{tag}.err.log").write_text(err, encoding="utf-8")
        log(f"{tag} EXIT={r.returncode}")
        return r.returncode
    except subprocess.TimeoutExpired:
        log(f"{tag} TIMEOUT(>{timeout_s}s)")
        return -1
    except Exception as e:
        log(f"{tag} ERROR: {e}")
        return -2


def main():
    log("=== bench_final_orchestrator start ===")

    # 1) 11s 重训
    run([PY, "models/train.py",
         "--data", "configs/frisbee_merged_v2_win.yaml",
         "--model", "yolo11s.pt",
         "--box", "5", "--epochs", "30", "--patience", "8", "--close-mosaic", "5",
         "--name", "bench_11s", "--workers", "4"], "11s-retrain", timeout_s=7200)

    # 2) D-FINE test 评测
    run([PY, "tools/eval_dfine_test.py"], "dfine-eval", timeout_s=1800)

    # 3) prod 断点恢复
    if PROD_LAST.exists():
        run([PY, "-c",
             f"from ultralytics import YOLO; YOLO(r'{PROD_LAST}').train(resume=True)"],
            "prod-resume", timeout_s=14400)

    log("=== bench_final_orchestrator ALL DONE ===")


if __name__ == "__main__":
    main()
