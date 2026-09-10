"""worker 启动参数构造（纯标准库，便于无 PySide6 环境单测）。

GUI 发起的每一步 GPU 分析都自动套 tools/gpu_run.sh 排队锁（/tmp/frisbee_gpu.lock，
跨会话共享），避免与基准/训练队列撞卡。
"""

from __future__ import annotations

from pathlib import Path

from frisbee_analyzer import protocol

PROJECT_ROOT_GUI = Path(__file__).resolve().parents[1]  # 所在 worktree/仓库根（Windows 路径）


def main_checkout_root() -> Path:
    """从 worktree 运行时返回主 checkout 根；否则返回当前仓库根。"""
    if PROJECT_ROOT_GUI.parent.name == ".worktrees":
        return PROJECT_ROOT_GUI.parent.parent
    return PROJECT_ROOT_GUI


def gpu_queue_script() -> str:
    """主 checkout 的 gpu_run.sh（WSL 路径）——锁文件在 /tmp，脚本用任意 checkout 的副本均可。"""
    return protocol.win_to_wsl(str(main_checkout_root() / "tools" / "gpu_run.sh"))


def _fwd(path: str) -> str:
    """Windows 路径统一为正斜杠，避免 wsl.exe 传参时的反斜杠转义问题。"""
    return str(path).replace("\\", "/")


def build_worker_argv(video_win: str, output_dir_win: str, weights_win: str | None = None,
                      max_frames: int | None = None, task_name: str = "gui-analysis",
                      use_gpu_queue: bool = True, team_only: str | None = None,
                      classes: str | None = None, team_from_cls: bool = False) -> list[str]:
    """构造 wsl.exe 参数列表。GPU 任务一律经 gpu_run.sh 排队（use_gpu_queue=False 需显式说明理由）。"""
    wsl_root = protocol.win_to_wsl(str(PROJECT_ROOT_GUI))
    cmd = ["python3", "-m", "frisbee_analyzer.pipeline"]
    if team_only:
        cmd += ["--team-only", _fwd(team_only)]
    else:
        cmd += ["--video", _fwd(video_win), "--output-dir", _fwd(output_dir_win)]
    if weights_win:
        cmd += ["--weights", protocol.win_to_wsl(_fwd(weights_win))]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    if classes:
        cmd += ["--classes", classes]
    if team_from_cls:
        cmd += ["--team-from-cls"]
    if use_gpu_queue:
        return ["--cd", wsl_root, "-e", "bash", gpu_queue_script(), task_name, *cmd]
    return ["--cd", wsl_root, "-e", *cmd]
