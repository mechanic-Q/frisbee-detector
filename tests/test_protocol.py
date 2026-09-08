"""protocol.py 单测：消息 schema、JSON 行编解码、Windows↔WSL 路径转换。纯标准库。"""

import io
import json

from frisbee_analyzer import protocol


# ── 消息构造 ──────────────────────────────────────────────

def test_meta_schema():
    m = protocol.meta(1800, 30.0, 1920, 1080)
    assert m["type"] == "meta"
    assert m["total_frames"] == 1800 and m["fps"] == 30.0
    assert m["width"] == 1920 and m["height"] == 1080


def test_progress_and_result_and_error():
    assert protocol.progress(30, 1800) == {"type": "progress", "frame": 30, "total": 1800}
    assert protocol.result("E:/x/tracks.json") == {"type": "result", "path": "E:/x/tracks.json"}
    assert protocol.error("boom") == {"type": "error", "msg": "boom"}


def test_emit_writes_single_flushed_line():
    buf = io.StringIO()
    protocol.emit(protocol.log("hello"), buf)
    protocol.emit(protocol.progress(1, 2), buf)
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"type": "log", "msg": "hello"}


def test_parse_line_roundtrip_and_degradation():
    assert protocol.parse_line('{"type": "log", "msg": "hi"}') == {"type": "log", "msg": "hi"}
    assert protocol.parse_line("") is None
    assert protocol.parse_line("   \n") is None
    # 非 JSON 输出（如库打印的警告）降级为 log，不炸 GUI 解析循环
    degraded = protocol.parse_line("some library warning: value=0.5")
    assert degraded["type"] == "log" and "warning" in degraded["msg"]


# ── 路径转换 ──────────────────────────────────────────────

def test_win_to_wsl():
    assert protocol.win_to_wsl(r"E:\frisbee-detector\movie\a.mp4") == "/mnt/e/frisbee-detector/movie/a.mp4"
    assert protocol.win_to_wsl("E:/a/b.mp4") == "/mnt/e/a/b.mp4"
    assert protocol.win_to_wsl("D:\\x") == "/mnt/d/x"
    # WSL 路径原样通过
    assert protocol.win_to_wsl("/mnt/e/a.mp4") == "/mnt/e/a.mp4"
    assert protocol.win_to_wsl("relative/video.mp4") == "relative/video.mp4"


def test_wsl_to_win():
    assert protocol.wsl_to_win("/mnt/e/a/b.mp4") == "E:/a/b.mp4"
    assert protocol.wsl_to_win("/mnt/d") == "D:"
    # Windows 路径原样通过
    assert protocol.wsl_to_win("E:/a/b.mp4") == "E:/a/b.mp4"
    # 非 /mnt 挂载点的 WSL 路径不动（如 ~ 或 /home）
    assert protocol.wsl_to_win("/home/lmr/x.pt") == "/home/lmr/x.pt"


def test_roundtrip_win_wsl_win():
    orig = r"E:\frisbee-detector\movie\2024决赛.mp4"
    assert protocol.wsl_to_win(protocol.win_to_wsl(orig)) == "E:/frisbee-detector/movie/2024决赛.mp4"
