#!/usr/bin/env python3
"""GPU 排队执行器（跨平台）——tools/gpu_run.sh 的 Windows 原生等价物。

用法:
    python tools/gpu_run.py <task_name> <command> [args...]
例:
    python tools/gpu_run.py gui-analysis python -m frisbee_analyzer.pipeline --video ...

同一时刻只放行一个 GPU 任务，后来的自动排队（锁文件 %TEMP%/frisbee_gpu.lock）:
    Windows: msvcrt.locking 独占锁（轮询等待）
    POSIX:   fcntl.flock LOCK_EX

与 gpu_run.sh 共享同一把锁的前提是两者对 /tmp 与 %TEMP% 指向同一物理目录
（Git Bash 默认如此）；跨 WSL/Windows 混跑不互斥——现在也不再需要。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Windows 双 OpenMP 运行时规避（libiomp5md.dll 重复初始化会 OMP Error #15 崩溃）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

LOCK_FILE = Path(os.environ.get("GPU_LOCK", Path(tempfile.gettempdir()) / "frisbee_gpu.lock"))


def _acquire(lock_fh) -> None:
    if os.name == "nt":
        import msvcrt

        while True:
            try:
                lock_fh.seek(0)
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(1)
    else:
        import fcntl

        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    task = sys.argv[1]
    cmd = sys.argv[2:]

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK_FILE, "a+")
    print(f"[gpu-queue] '{task}' 排队等待 GPU ... {time.strftime('%H:%M:%S')}", flush=True)
    _acquire(fh)
    print(f"[gpu-queue] '{task}' 获得 GPU {time.strftime('%H:%M:%S')}", flush=True)
    try:
        rc = subprocess.call(cmd)
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        print(f"[gpu-queue] '{task}' 释放 GPU (exit={rc}) {time.strftime('%H:%M:%S')}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
