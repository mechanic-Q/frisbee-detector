"""GUI 入口。

    python -m gui.main                     # 正常启动
    python -m gui.main --smoke VIDEO [TRACKS_JSON]
                                           # 自动化烟雾测试：加载后渲染一帧并退出，
                                           # 打印 "SMOKE PASS/FAIL ..."（配 QT_QPA_PLATFORM=offscreen）
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    smoke = "--smoke" in argv
    if smoke:
        argv.remove("--smoke")
        smoke_video = argv[1] if len(argv) > 1 else None
        smoke_tracks = argv[2] if len(argv) > 2 else None
        argv = argv[:1]

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(argv)
    window = MainWindow()
    window.resize(1360, 820)
    window.show()

    if not smoke:
        return app.exec()

    state = {"ok": True}

    def _check():
        try:
            if smoke_video:
                window.open_video(smoke_video)
            if smoke_tracks:
                window.load_tracks(smoke_tracks)
                state["ok"] &= window.doc is not None
            pixmap = window.player.grab()  # 触发一次完整渲染（叠加层 paint 路径）
            state["ok"] &= not pixmap.isNull()
        except Exception as e:  # noqa: BLE001
            print(f"SMOKE FAIL exception: {type(e).__name__}: {e}", flush=True)
            app.quit()
            return
        print(f"SMOKE {'PASS' if state['ok'] else 'FAIL'} "
              f"(doc={'yes' if window.doc else 'no'}, rendered={not pixmap.isNull()})", flush=True)
        app.quit()

    QTimer.singleShot(1500, _check)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
