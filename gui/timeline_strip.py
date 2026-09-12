"""事件时间线条（§13.16 Phase D）：按 doc["events"] 画刻度，点击 seek。

布局约定：横条控件放底部播放 dock（slider 下方一行）。candidate 事件画醒色
刻度（score 候选=橙、其他候选=黄），counted 事件画青色三角；当前位置画白色
游标。点击/拖动 → seekRequested(frame) 信号给主窗口接 player.seek_frame。
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

CAND_SCORE = QColor(255, 140, 0)   # 橙：得分候选
CAND_OTHER = QColor(230, 210, 60)  # 黄：其他候选
COUNTED = QColor(60, 200, 220)     # 青：counted 事件
CURSOR = QColor(255, 255, 255)


class EventTimelineStrip(QWidget):
    seekRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._events: list[dict] = []
        self._total_frames: int = 0
        self._current: int = 0
        self._dragging = False
        self.setMinimumHeight(26)

    # ── 数据 ──
    def set_events(self, events: list[dict], total_frames: int):
        self._events = sorted(events, key=lambda e: e.get("frame", 0))
        self._total_frames = max(int(total_frames), 1)
        self.update()

    def set_position(self, frame: int):
        self._current = int(frame)
        self.update()

    # ── 交互 ──
    def _frame_at(self, x: int) -> int:
        w = max(self.width() - 8, 1)
        frac = min(max((x - 4) / w, 0.0), 1.0)
        return int(frac * self._total_frames)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._total_frames:
            self._dragging = True
            self.seekRequested.emit(self._frame_at(ev.position().toPoint().x()))

    def mouseMoveEvent(self, ev):
        if self._dragging and self._total_frames:
            self.seekRequested.emit(self._frame_at(ev.position().toPoint().x()))

    def mouseReleaseEvent(self, ev):
        self._dragging = False

    # ── 绘制 ──
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(QRectF(0, 0, w, h), QColor(24, 24, 28))
        p.setPen(QPen(QColor(70, 70, 76), 1))
        p.drawRect(0, 0, w - 1, h - 1)
        if not self._total_frames:
            return

        def x_of(frame: int) -> float:
            return 4 + (w - 8) * (frame / self._total_frames)

        # 事件刻度
        for e in self._events:
            x = x_of(e.get("frame", 0))
            if e.get("candidate"):
                color = CAND_SCORE if e.get("type") == "score" else CAND_OTHER
                p.setPen(QPen(color, 2))
                top, bottom = 4.0, float(h - 4)
            else:
                color = COUNTED
                p.setPen(QPen(color, 2))
                top, bottom = h * 0.45, float(h - 4)
            p.drawLine(int(x), int(top), int(x), int(bottom))

        # 当前位置游标
        cx = x_of(min(self._current, self._total_frames))
        p.setPen(QPen(CURSOR, 1))
        p.drawLine(int(cx), 2, int(cx), h - 2)
