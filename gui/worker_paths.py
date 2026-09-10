"""worker 启动参数构造（纯标准库，便于无 PySide6 环境单测）。

Windows 原生：worker 直接以 `python -m frisbee_analyzer.pipeline` 运行（不再经
wsl.exe），GPU 分析自动套 tools/gpu_run.py 排队锁（%TEMP%/frisbee_gpu.lock，
跨进程共享），避免与训练/基准任务撞卡。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_GUI = Path(__file__).resolve().parents[1]  # 仓库根（GUI 从任一 checkout 运行皆取自身）


def main_checkout_root() -> Path:
    """兼容别名：worktree 体系已移除，仓库根即主 checkout 根。"""
    return PROJECT_ROOT_GUI


def python_exe() -> str:
    """worker 用的 Python 解释器（需带 CUDA torch + ultralytics）。

    GUI 进程本身跑在 PySide6 专用解释器（py -3.11）上；worker 默认用 PATH 上的
    `python`（当前 = 3.13 + torch 2.11 cu128），可用环境变量 FRISBEE_PYTHON 覆盖。
    """
    return os.environ.get("FRISBEE_PYTHON", "python")


def build_worker_argv(video_win: str, output_dir_win: str, weights_win: str | None = None,
                      max_frames: int | None = None, task_name: str = "gui-analysis",
                      use_gpu_queue: bool = True, team_only: str | None = None,
                      classes: str | None = None, team_from_cls: bool = False) -> list[str]:
    """构造 worker 命令行（Windows 原生路径原样传递）。

    GPU 任务一律经 gpu_run.py 排队（use_gpu_queue=False 需显式说明理由）。
    """
    cmd = [python_exe(), "-m", "frisbee_analyzer.pipeline"]
    if team_only:
        cmd += ["--team-only", str(team_only)]
    else:
        cmd += ["--video", str(video_win), "--output-dir", str(output_dir_win)]
    if weights_win:
        cmd += ["--weights", str(weights_win)]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    if classes:
        cmd += ["--classes", classes]
    if team_from_cls:
        cmd += ["--team-from-cls"]
    if use_gpu_queue:
        return [python_exe(), str(PROJECT_ROOT_GUI / "tools" / "gpu_run.py"), task_name, *cmd]
    return cmd
