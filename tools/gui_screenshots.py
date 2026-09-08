"""离屏渲染 GUI 各层级截图（无需真人交互、不占 GPU）。

在项目根目录运行（Windows）：
    py -3.11 tools/gui_screenshots.py [VIDEO TRACKS_JSON]
默认用 accept_10min 的视频与结果。输出 runs/gui_analysis/screenshots/*.png。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")  # 离屏平台默认无 CJK 字体

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEO = sys.argv[1] if len(sys.argv) > 1 else str(Path("E:/frisbee-detector/data/bili_final_test/accept_10min.mp4"))
TRACKS = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "runs/gui_analysis/accept_10min/tracks.json")
OUT = ROOT / "runs/gui_analysis/screenshots"


def pump(app, seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.03)


def wait_until(fn, timeout: float, app):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if fn():
            return True
        time.sleep(0.03)
    return False


def main() -> int:
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from gui.calibrate_dialog import CalibrateDialog
    from gui.main_window import MainWindow
    from gui.team_override import TeamOverrideDialog
    from gui.theme import apply_theme

    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    apply_theme(app)
    win = MainWindow()
    win.resize(1360, 820)
    win.show()
    pump(app, 0.8)
    win.grab().save(str(OUT / "01_main_empty.png"))

    win.open_video(VIDEO)
    win.player.play()
    got_frame = wait_until(lambda: win.player.grab_current_frame() is not None, 8.0, app)
    win.player.pause()
    if got_frame and win.doc:
        win.player.seek_frame(9000, win.fps)
        pump(app, 1.2)
    win.log_view.appendPlainText('[worker] $ wsl.exe --cd /mnt/e/frisbee-detector/.worktrees/gui-v0 '
                                 '-e bash /mnt/e/frisbee-detector/tools/gpu_run.sh gui-analysis python3 -m frisbee_analyzer.pipeline ...')
    win.log_view.appendPlainText('[gpu-queue] \'gui-analysis\' 获得 GPU')
    win.log_view.appendPlainText('{"type": "progress", "frame": 11490, "total": 18004}')
    win.log_view.appendPlainText('{"type": "result", "path": ".../accept_10min/tracks.json"}')
    win.log_dock.show()
    pump(app, 0.5)
    win.grab().save(str(OUT / "02_main_overlay.png"))

    frame = win.player.grab_current_frame() or QImage(1280, 720, QImage.Format.Format_RGB32)
    dlg = CalibrateDialog(frame, win.player.video_size() or (1920, 1080), VIDEO, 9000, win)
    dlg.show()
    for i, px in enumerate([(120, 950), (1790, 950), (1790, 420), (120, 420)]):
        dlg.combo.setCurrentIndex(i)  # 预设 0-3 = 四个场角
        dlg._on_canvas_click(px)
    pump(app, 0.4)
    dlg.grab().save(str(OUT / "03_calibrate_dialog.png"))
    dlg.close()

    if win.doc:
        tdlg = TeamOverrideDialog(win.doc, win.doc_path, win)
        tdlg.resize(640, 520)
        tdlg.show()
        pump(app, 0.5)
        tdlg.grab().save(str(OUT / "04_team_override.png"))
        tdlg.close()

    saved = sorted(p.name for p in OUT.glob("*.png"))
    print("SAVED:", ", ".join(saved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
