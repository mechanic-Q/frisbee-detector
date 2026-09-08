"""视频播放组件：QMediaPlayer + QVideoSink → QImage → QPainter，支持叠加层与点击取点。"""

from __future__ import annotations

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink
from PySide6.QtWidgets import QWidget

from . import overlay


class VideoPlayerWidget(QWidget):
    """当前帧渲染 + 检测叠加 + 可选的取点回调（标定模式）。"""

    positionChangedMs = Signal("qlonglong")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self._frame: QImage | None = None
        self._video_size = (0, 0)
        self._overlay_dets: list[dict] | None = None
        self._team_colors: dict = {}
        self._field_polylines: list[list[tuple[float, float]]] = []
        self._calib_points: list[tuple[float, float]] = []
        self._click_handler = None  # callable(video_x, video_y)

        self._player = QMediaPlayer(self)
        self._sink = QVideoSink(self)
        self._player.setVideoSink(self._sink)
        self._sink.videoFrameChanged.connect(self._on_video_frame)
        self._player.positionChanged.connect(self.positionChangedMs)

    # ── 播放控制 ─────────────────────────────────────────
    def load(self, video_path: str) -> None:
        self._player.setSource(QUrl.fromLocalFile(str(video_path)))

    def play(self) -> None:
        self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def position_ms(self) -> int:
        return self._player.position()

    def seek_ms(self, ms: int) -> None:
        self._player.setPosition(int(max(0, ms)))

    def seek_frame(self, frame_idx: int, fps: float) -> None:
        self.seek_ms(round(frame_idx * 1000.0 / fps))

    def current_frame_index(self, fps: float) -> int:
        return round(self.position_ms() * fps / 1000.0)

    def grab_current_frame(self) -> QImage | None:
        return self._frame

    def video_size(self) -> tuple[int, int]:
        return self._video_size

    # ── 叠加内容 ─────────────────────────────────────────
    def set_overlay(self, dets: list[dict] | None) -> None:
        self._overlay_dets = dets
        self.update()

    def set_team_colors(self, colors: dict | None) -> None:
        """doc["team_colors"]：{"0":"red","1":"blue"}——队伍实际球衣色，用于框着色。"""
        self._team_colors = colors or {}
        self.update()

    def set_field_polylines(self, polylines: list[list[tuple[float, float]]]) -> None:
        self._field_polylines = polylines
        self.update()

    def set_calib_points(self, points: list[tuple[float, float]]) -> None:
        self._calib_points = points
        self.update()

    def set_click_handler(self, handler) -> None:
        """标定模式：handler(video_x, video_y)；传 None 退出取点模式。"""
        self._click_handler = handler

    # ── 渲染 ─────────────────────────────────────────────
    def _on_video_frame(self, frame) -> None:
        if frame.isValid():
            img = frame.toImage()
            if not img.isNull():
                self._frame = img
                self._video_size = (img.width(), img.height())
                self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(12, 12, 12))
        vw, vh = self._video_size
        if self._frame is None or vw == 0:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), 0x0084, "打开视频开始（Ctrl+O）")
            return
        scale, dx, dy = overlay.letterbox(vw, vh, self.width(), self.height())
        target = self._frame.scaled(int(vw * scale), int(vh * scale))
        painter.drawImage(int(dx), int(dy), target)
        if self._field_polylines:
            overlay.draw_polylines(painter, self._field_polylines, vw, vh, self.width(), self.height())
        if self._overlay_dets:
            overlay.draw_detections(painter, self._overlay_dets, vw, vh, self.width(), self.height(),
                                    team_colors=self._team_colors)
        if self._calib_points:
            overlay.draw_points(painter, self._calib_points, vw, vh, self.width(), self.height())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._click_handler is None or not self._video_size[0]:
            return
        vx, vy = overlay.widget_to_video(event.position().x(), event.position().y(),
                                         *self._video_size, self.width(), self.height())
        self._click_handler(vx, vy)
