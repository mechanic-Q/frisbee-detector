"""分析 worker 进程管理：QProcess → wsl.exe → python3 -m frisbee_analyzer.pipeline。

跨约定：GUI 传 Windows 路径（统一正斜杠防 wsl.exe 引号坑），worker 内部转换；
stdout 逐行 JSON（协议 frisbee_analyzer/protocol.py），强制 UTF-8。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Signal

from frisbee_analyzer import protocol

PROJECT_ROOT_GUI = Path(__file__).resolve().parents[1]  # 所在 worktree/仓库根（Windows 路径）


class AnalysisWorker(QProcess):
    metaReceived = Signal(dict)
    progressChanged = Signal(int, int)
    logLine = Signal(str)
    resultReady = Signal(str)   # tracks.json 的 Windows 路径
    failed = Signal(str)        # 出错或被取消（msg 区分）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer = b""
        self._cancel_requested = False
        self.readyReadStandardOutput.connect(self._on_stdout)
        self.readyReadStandardError.connect(self._on_stderr)
        self.finished.connect(self._on_finished)

    def start_analysis(self, video_win_path: str, output_dir_win: str,
                       weights_win_path: str | None = None,
                       max_frames: int | None = None) -> None:
        self._cancel_requested = False
        self._buffer = b""
        wsl_root = protocol.win_to_wsl(str(PROJECT_ROOT_GUI))
        argv = ["--cd", wsl_root, "-e", "python3", "-m", "frisbee_analyzer.pipeline",
                "--video", _fwd(video_win_path),
                "--output-dir", _fwd(output_dir_win)]
        if weights_win_path:
            argv += ["--weights", protocol.win_to_wsl(_fwd(weights_win_path))]
        if max_frames:
            argv += ["--max-frames", str(max_frames)]

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONUNBUFFERED", "1")
        self.setProcessEnvironment(env)
        self.setProgram("wsl.exe")
        self.setArguments(argv)
        self.logLine.emit(f"$ wsl.exe {' '.join(argv)}")
        self.start()

    def cancel(self) -> None:
        """先礼貌 terminate，3 秒后强杀。worker 退出码 2 = 已取消。"""
        self._cancel_requested = True
        self.terminate()
        QTimer.singleShot(3000, self.kill)

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def _on_stdout(self) -> None:
        self._buffer += bytes(self.readAllStandardOutput())
        while b"\n" in self._buffer:
            raw, _, rest = self._buffer.partition(b"\n")
            self._buffer = rest
            msg = protocol.parse_line(raw.decode("utf-8", errors="replace"))
            if msg is None:
                continue
            kind = msg.get("type")
            if kind == "meta":
                self.metaReceived.emit(msg)
            elif kind == "progress":
                self.progressChanged.emit(int(msg["frame"]), int(msg["total"]))
            elif kind == "log":
                self.logLine.emit(msg["msg"])
            elif kind == "result":
                self.resultReady.emit(msg["path"])
            elif kind == "error":
                self.logLine.emit(f"[worker error] {msg['msg']}")

    def _on_stderr(self) -> None:
        text = bytes(self.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self.logLine.emit(f"[stderr] {text[-2000:]}")

    def _on_finished(self, exit_code, _status) -> None:
        if self._cancel_requested or exit_code == 2:
            self.failed.emit("分析已取消")
        elif exit_code != 0:
            self.failed.emit(f"worker 退出码 {exit_code}（详情见日志）")


def _fwd(path: str) -> str:
    """Windows 路径统一为正斜杠，避免 wsl.exe 传参时的反斜杠转义问题。"""
    return str(path).replace("\\", "/")
