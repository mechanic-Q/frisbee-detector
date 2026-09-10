"""JSON-lines 协议：分析 worker 与 PySide6 GUI 之间的消息 schema。

纯标准库——worker 与 GUI 进程都能直接导入，不引入 cv2/torch/Qt。
消息格式见 docs/superpowers/plans/2026-09-09-gui-v0.md §3。

历史注：win_to_wsl/wsl_to_win 是 2026-09-11 之前"Windows GUI + WSL worker"架构的
路径转换器，现已不再使用（worker 与 GUI 同在 Windows 原生运行），仅为兼容保留。
"""

from __future__ import annotations

import json
import re
import sys

# ── 消息构造（worker → GUI）──────────────────────────────────────────


def meta(total_frames: int, fps: float, width: int, height: int) -> dict:
    return {"type": "meta", "total_frames": int(total_frames), "fps": float(fps),
            "width": int(width), "height": int(height)}


def progress(frame: int, total_frames: int) -> dict:
    return {"type": "progress", "frame": int(frame), "total": int(total_frames)}


def log(message: str) -> dict:
    return {"type": "log", "msg": str(message)}


def result(payload_path: str) -> dict:
    return {"type": "result", "path": str(payload_path)}


def error(message: str) -> dict:
    return {"type": "error", "msg": str(message)}


def emit(message: dict, stream=None) -> None:
    """写一行 JSON 并立即 flush——QProcess 逐行读取依赖 flush。"""
    stream = stream if stream is not None else sys.stdout
    stream.write(json.dumps(message, ensure_ascii=False) + "\n")
    stream.flush()


def parse_line(line: str) -> dict | None:
    """GUI 侧解析一行 stdout。空行 → None；非 JSON 输出降级为 log 消息。"""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return log(line)


# ── 路径转换（已废弃：Windows GUI ↔ WSL worker 时代遗留，无调用方）──

_WIN_DRIVE = re.compile(r"^([A-Za-z]):[/\\](.*)$")
_WSL_MOUNT = re.compile(r"^/mnt/([a-z])(/.*)?$")


def win_to_wsl(path: str) -> str:
    """`E:\\a\\b.mp4` / `E:/a/b.mp4` → `/mnt/e/a/b.mp4`。WSL 路径原样返回。"""
    p = path.replace("\\", "/")
    m = _WIN_DRIVE.match(p)
    if m:
        rest = m.group(2)
        return f"/mnt/{m.group(1).lower()}/{rest}" if rest else f"/mnt/{m.group(1).lower()}"
    return path


def wsl_to_win(path: str) -> str:
    """`/mnt/e/a/b.mp4` → `E:/a/b.mp4`。Windows 路径原样返回。"""
    m = _WSL_MOUNT.match(path)
    if m:
        rest = m.group(2) or ""
        return f"{m.group(1).upper()}:{rest}"
    return path
