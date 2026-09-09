"""Windows 原生 GPU 任务队列：11s 重训 → D-FINE-S 训练 → 统一评测。
运行: python tools/win_gpu_queue.py
每个任务前等待 GPU 空闲（含 WSL 侧占用者），避免多进程 CUDA 并发崩溃。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/bili_final_test/win_queue.log"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_busy() -> bool:
    """判定是否有真实训练任务在跑：查 python 进程命令行是否含 train.py
    （Windows 的 nvidia-smi 无法可靠区分图形进程/缓存占用，故用进程判据）。"""
    import re
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
            capture_output=True, text=True, timeout=30)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        for l in lines[1:]:
            if re.search(r"train\.py|train_dfine|yolo.*train", l, re.I):
                return True
        return False
    except Exception:
        # wmic 不可用则退化为显存阈值（>10GB 视为有训练）
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
            return int(r.stdout.strip().split("\n")[0]) > 10000
        except Exception:
            return False


def wait_gpu_free(tag: str, timeout_min: int = 600):
    t0 = time.time()
    while gpu_busy():
        if time.time() - t0 > timeout_min * 60:
            log(f"{tag}: GPU 等待超时，继续执行")
            return
        log(f"{tag}: 等 GPU 空闲...")
        time.sleep(60)


def run(cmd, tag):
    log(f"{tag} START: {' '.join(str(c) for c in cmd[:5])}...")
    r = subprocess.run(cmd, cwd=str(ROOT))
    log(f"{tag} EXIT={r.returncode}")
    return r.returncode


def train_11s():
    wait_gpu_free("11s-retrain")
    return run([
        sys.executable, "models/train.py",
        "--data", "configs/frisbee_merged_v2_win.yaml",
        "--model", "yolo11s.pt",
        "--box", "5", "--epochs", "30", "--patience", "8", "--close-mosaic", "5",
        "--name", "bench_11s", "--workers", "4",
    ], "11s-retrain")


def train_dfine():
    wait_gpu_free("dfine")
    return run([
        sys.executable, "D-FINE/train.py",
        "-c", "D-FINE/configs/dfine/dfine_hgnetv2_s_frisbee.yml",
        "--use-amp", "--seed=42",
        "-t", "data/bili_final_test/dfine_s_coco.pth",
    ], "dfine")


def main():
    log("=== Windows 原生 GPU 队列启动 ===")
    train_11s()
    train_dfine()
    log("=== 队列完成 ===")


if __name__ == "__main__":
    main()
