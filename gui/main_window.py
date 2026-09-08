"""主窗口：视频播放 + 结果叠加 + 分析启动/取消 + 标定 + 队伍改判。"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QDockWidget, QHBoxLayout, QSlider,
                               QStyle, QVBoxLayout, QWidget)

from utils.homography import FIELD_LINES_WORLD, load_calibration, world_to_pixel

from .calibrate_dialog import CalibrateDialog
from .team_override import TeamOverrideDialog
from .video_player import VideoPlayerWidget
from .worker_paths import PROJECT_ROOT_GUI, main_checkout_root
from .worker_process import AnalysisWorker


VIDEO_FILTER = "视频 (*.mp4 *.mov *.mkv *.avi *.webm);;所有文件 (*)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("飞盘赛事分析 GUI v0")
        self.video_path: str | None = None
        self.doc: dict | None = None
        self.doc_path: str | None = None
        self.fps: float = 30.0
        self.calib_matrix = None

        self.player = VideoPlayerWidget(self)
        self.setCentralWidget(self.player)

        self.worker = AnalysisWorker(self)
        self.worker.metaReceived.connect(self._on_worker_meta)
        self.worker.progressChanged.connect(self._on_worker_progress)
        self.worker.logLine.connect(self._log)
        self.worker.resultReady.connect(self._on_worker_result)
        self.worker.failed.connect(self._on_worker_failed)

        self._build_actions()
        self._build_bottom_bar()
        self._build_log_dock()
        self.statusBar().showMessage("就绪 — Ctrl+O 打开视频")

    # ── UI 构建 ──────────────────────────────────────────
    def _build_actions(self):
        self.act_open = QAction("打开视频", self)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open.triggered.connect(self.open_video_dialog)

        self.act_open_result = QAction("打开分析结果", self)
        self.act_open_result.triggered.connect(self.open_tracks_dialog)

        self.act_start = QAction("启动分析", self)
        self.act_start.triggered.connect(self.start_analysis)

        self.act_cancel = QAction("取消分析", self)
        self.act_cancel.setEnabled(False)
        self.act_cancel.triggered.connect(self.worker.cancel)

        self.act_calib = QAction("场地标定", self)
        self.act_calib.triggered.connect(self.open_calibration)

        self.act_team = QAction("队伍改判", self)
        self.act_team.setEnabled(False)
        self.act_team.triggered.connect(self.open_team_override)

        self.act_field_only = QAction("只看场内球员", self)
        self.act_field_only.setCheckable(True)
        self.act_field_only.setChecked(True)
        self.act_field_only.triggered.connect(self._update_overlay)

        self.act_show_unassigned = QAction("显示未分配", self)
        self.act_show_unassigned.setCheckable(True)
        self.act_show_unassigned.setChecked(False)
        self.act_show_unassigned.triggered.connect(self._update_overlay)

        for act in (self.act_open, self.act_open_result, self.act_start,
                    self.act_cancel, self.act_calib, self.act_team):
            self.addAction(act)

        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_open_result)
        toolbar.addSeparator()
        toolbar.addAction(self.act_start)
        toolbar.addAction(self.act_cancel)
        toolbar.addSeparator()
        toolbar.addAction(self.act_calib)
        toolbar.addAction(self.act_team)
        toolbar.addSeparator()
        toolbar.addAction(self.act_field_only)
        toolbar.addAction(self.act_show_unassigned)

        # 键盘：空格播放/暂停，←→ 逐帧
        for keys, fn in (("Space", self.player.toggle_play), ("Left", lambda: self._step(-1)),
                         ("Right", lambda: self._step(+1))):
            act = QAction(self)
            act.setShortcut(QKeySequence(keys))
            act.triggered.connect(fn)
            self.addAction(act)

    def _build_bottom_bar(self):
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        self.btn_play = QPushButton("播放/暂停")
        self.btn_play.clicked.connect(self.player.toggle_play)
        self.btn_prev = QPushButton("◀ 帧")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = QPushButton("帧 ▶")
        self.btn_next.clicked.connect(lambda: self._step(+1))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.player.seek_ms)
        self.player.positionChangedMs.connect(self._on_position_ms)
        lay.addWidget(self.btn_prev)
        lay.addWidget(self.btn_play)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.slider, stretch=1)
        dock = QDockWidget("播放控制", self)
        dock.setWidget(bar)
        dock.setTitleBarWidget(QWidget())  # 无标题栏
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.frame_label = QLabel("")
        self.statusBar().addPermanentWidget(self.frame_label)

    def _build_log_dock(self):
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_dock = QDockWidget("分析日志", self)
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        self.log_dock.hide()

    def _log(self, text: str):
        self.log_view.appendPlainText(text)
        self.statusBar().showMessage(text[:160], 8000)

    # ── 打开视频 / 结果 ──────────────────────────────────
    def open_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", str(main_checkout_root() / "movie"), VIDEO_FILTER)
        if path:
            self.open_video(path)

    def open_video(self, path: str):
        self.video_path = path
        self.player.load(path)
        stem = Path(path).stem
        default_tracks = PROJECT_ROOT_GUI / "runs" / "gui_analysis" / stem / "tracks.json"
        if default_tracks.exists():
            self.load_tracks(str(default_tracks))
            self._log(f"已自动加载分析结果: {default_tracks}")
        default_calib = PROJECT_ROOT_GUI / "configs" / "homography" / f"{stem}.json"
        if default_calib.exists():
            self._apply_calibration(str(default_calib))
            self._log(f"已自动加载场地标定: {default_calib}")
        self._log(f"视频已打开: {path}")

    def open_tracks_dialog(self):
        start = str(PROJECT_ROOT_GUI / "runs" / "gui_analysis")
        path, _ = QFileDialog.getOpenFileName(self, "打开分析结果", start, "tracks.json (tracks.json);;JSON (*.json)")
        if path:
            self.load_tracks(path)

    def load_tracks(self, path: str):
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        self.doc = doc
        self.doc_path = path
        self.fps = float(doc.get("fps") or 30.0)
        self.act_team.setEnabled(True)
        self.player.set_team_colors(doc.get("team_colors"))
        self._update_overlay()

    # ── 播放与叠加同步 ───────────────────────────────────
    def _on_position_ms(self, ms: int):
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self._update_overlay()

    def _step(self, delta: int):
        self.player.pause()
        fps = self.fps or 30.0
        cur = self.player.current_frame_index(fps)
        self.player.seek_frame(max(0, cur + delta), fps)

    def _update_overlay(self):
        if self.doc is None:
            return
        idx = self.player.current_frame_index(self.fps)
        dets = self.doc.get("frames", {}).get(str(idx))
        if dets:
            from frisbee_analyzer.filters import FIELD_POLYGON_M, filter_dets

            if self.act_field_only.isChecked():
                dets = filter_dets(dets, self.doc.get("height") or 1080,
                                   matrix=self.calib_matrix, polygon=FIELD_POLYGON_M,
                                   require_team=not self.act_show_unassigned.isChecked())
            elif not self.act_show_unassigned.isChecked():
                dets = [d for d in dets if d.get("team_id") is not None]
        self.player.set_overlay(dets)
        self.frame_label.setText(f"f{idx}")

    # ── 分析任务 ─────────────────────────────────────────
    def start_analysis(self):
        if not self.video_path:
            QMessageBox.information(self, "提示", "请先打开视频（Ctrl+O）")
            return
        stem = Path(self.video_path).stem
        out_dir = PROJECT_ROOT_GUI / "runs" / "gui_analysis" / stem
        main_root = main_checkout_root()
        weights = str(main_root / "yolo26x.pt") if (main_root / "yolo26x.pt").exists() else None
        self.act_start.setEnabled(False)
        self.act_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # 收到 meta 后改确定范围
        self._log(f"启动分析: {self.video_path}")
        self.worker.start_analysis(self.video_path, str(out_dir), weights)

    def _on_worker_meta(self, msg: dict):
        self.progress.setRange(0, int(msg.get("total_frames") or 0))
        self.fps = float(msg.get("fps") or self.fps)
        self._log(f"worker 元信息: {msg['total_frames']} 帧 @ {msg['fps']:.1f} fps")

    def _on_worker_progress(self, frame: int, total: int):
        self.progress.setValue(frame)
        self.progress.setFormat(f"{frame}/{total}")

    def _on_worker_result(self, path: str):
        self.act_start.setEnabled(True)
        self.act_cancel.setEnabled(False)
        self.progress.setVisible(False)
        self.load_tracks(path)
        self._log(f"分析完成: {path}")

    def _on_worker_failed(self, msg: str):
        self.act_start.setEnabled(True)
        self.act_cancel.setEnabled(False)
        self.progress.setVisible(False)
        self._log(f"[失败] {msg}")
        QMessageBox.warning(self, "分析", msg)

    # ── 标定 ─────────────────────────────────────────────
    def open_calibration(self):
        if not self.video_path:
            QMessageBox.information(self, "提示", "请先打开视频")
            return
        self.player.pause()
        frame = self.player.grab_current_frame()
        if frame is None:
            QMessageBox.information(self, "提示", "尚无视频帧（等待解码或点击播放一帧后再试）")
            return
        dlg = CalibrateDialog(frame, self.player.video_size(), self.video_path,
                              self.player.current_frame_index(self.fps), self)
        if dlg.exec() and dlg.saved_path:
            self._apply_calibration(dlg.saved_path)
            self._log(f"标定已保存: {dlg.saved_path}")

    def _apply_calibration(self, path: str):
        calib = load_calibration(path)
        self.calib_matrix = calib["matrix"]
        lines_px = []
        for a, b in FIELD_LINES_WORLD:
            seg = []
            for i in range(41):
                t = i / 40
                px, py = world_to_pixel(calib["matrix"], a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                if px == px and py == py:
                    seg.append((px, py))
            if len(seg) > 1:
                lines_px.append(seg)
        self.player.set_field_polylines(lines_px)

    # ── 队伍改判 ─────────────────────────────────────────
    def open_team_override(self):
        if self.doc is None or not self.doc_path:
            QMessageBox.information(self, "提示", "尚无分析结果")
            return
        dlg = TeamOverrideDialog(self.doc, self.doc_path, self)
        dlg.teamsChanged.connect(self._update_overlay)
        dlg.exec()
