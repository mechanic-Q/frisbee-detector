"""场地标定对话框：在当前帧上点选 ≥4 个标定点 → 单应性 → 保存 configs/homography/<stem>.json。

世界坐标模板为 WFDF 标准场 100×37m；点选顺序自由，每点先在下拉框选世界坐标再点像素。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from utils.homography import (FIELD_LINES_WORLD, compute_homography, save_calibration,
                              world_to_pixel)

from . import overlay

# (wx, wy, 显示名) —— WFDF 100x37m 的角点/中点/中圈
WORLD_PRESETS = [
    (0, 0, "左下角 (0,0)"),
    (100, 0, "右下角 (100,0)"),
    (100, 37, "右上角 (100,37)"),
    (0, 37, "左上角 (0,37)"),
    (50, 0, "下边线中点 (50,0)"),
    (50, 37, "上边线中点 (50,37)"),
    (0, 18.5, "左边线中点 (0,18.5)"),
    (100, 18.5, "右边线中点 (100,18.5)"),
    (50, 18.5, "中圈点 (50,18.5)"),
]


class _Canvas(QWidget):
    """绘制帧 + 已选点 + 场地线预览；点击上报。"""

    def __init__(self, frame: QImage, video_size: tuple[int, int], parent=None):
        super().__init__(parent)
        self._frame = frame
        self._video_size = video_size
        self.points: list[tuple[float, float, float, float]] = []
        self.polylines: list[list[tuple[float, float]]] = []
        self.on_click = None
        max_w = 1080
        vw, vh = video_size
        self.setFixedSize(min(max_w, vw), int(min(max_w, vw) * vh / vw))

    def add_point(self, pixel, world):
        self.points.append((pixel[0], pixel[1], world[0], world[1]))
        self.update()

    def pop_point(self):
        if self.points:
            self.points.pop()
            self.update()

    def clear(self):
        self.points.clear()
        self.polylines.clear()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.on_click:
            vx, vy = overlay.widget_to_video(event.position().x(), event.position().y(),
                                             *self._video_size, self.width(), self.height())
            self.on_click((vx, vy))

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        vw, vh = self._video_size
        scale, dx, dy = overlay.letterbox(vw, vh, self.width(), self.height())
        painter.drawImage(int(dx), int(dy),
                          self._frame.scaled(int(vw * scale), int(vh * scale)))
        if self.polylines:
            overlay.draw_polylines(painter, self.polylines, vw, vh, self.width(), self.height())
        overlay.draw_points(painter, [(p[0], p[1]) for p in self.points], vw, vh,
                            self.width(), self.height())
        if not self.points:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "先选世界坐标，再点击图中对应位置（至少 4 个点）")


class CalibrateDialog(QDialog):
    def __init__(self, frame: QImage, video_size: tuple[int, int], video_path: str,
                 frame_idx: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("场地标定（WFDF 100×37m）")
        self.video_path = video_path
        self.frame_idx = frame_idx
        self.video_size = video_size
        self.matrix = None
        self.rmse_px = 0.0
        self.saved_path: str | None = None
        self._picked: list[tuple[float, float]] = []

        self.canvas = _Canvas(frame, video_size, self)
        self.canvas.on_click = self._on_canvas_click

        self.combo = QComboBox()
        for wx, wy, name in WORLD_PRESETS:
            self.combo.addItem(name, (wx, wy))
        self.status = QLabel("已选 0 点（≥4 后可计算）")
        self.undo_btn = QPushButton("撤销上一点")
        self.clear_btn = QPushButton("清空")
        self.compute_btn = QPushButton("计算单应性")
        self.compute_btn.setEnabled(False)
        self.save_btn = QPushButton("保存标定")
        self.save_btn.setEnabled(False)
        self.rmse_label = QLabel("")

        btns = QHBoxLayout()
        for w in (self.undo_btn, self.clear_btn, self.compute_btn, self.save_btn):
            btns.addWidget(w)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("第 1 步：下拉框选世界坐标 → 第 2 步：点击画面中对应位置"))
        lay.addWidget(self.combo)
        lay.addWidget(self.canvas)
        lay.addWidget(self.status)
        lay.addWidget(self.rmse_label)
        lay.addLayout(btns)

        self.undo_btn.clicked.connect(self._undo)
        self.clear_btn.clicked.connect(self._clear)
        self.compute_btn.clicked.connect(self._compute)
        self.save_btn.clicked.connect(self._save)

    # ── 交互 ─────────────────────────────────────────────
    def _on_canvas_click(self, pixel: tuple[float, float]) -> None:
        world = self.combo.currentData()
        self.canvas.add_point(pixel, world)
        self._picked.append((self.combo.currentText(), pixel))
        self.status.setText(f"已选 {len(self.canvas.points)} 点（≥4 后可计算）")
        self.compute_btn.setEnabled(len(self.canvas.points) >= 4)
        self.polylines_preview()
        self.canvas.polylines.clear()
        self.canvas.update()

    def _undo(self):
        self.canvas.pop_point()
        if self._picked:
            self._picked.pop()
        self.compute_btn.setEnabled(len(self.canvas.points) >= 4)
        self.status.setText(f"已选 {len(self.canvas.points)} 点（≥4 后可计算）")

    def _clear(self):
        self.canvas.clear()
        self._picked.clear()
        self.matrix = None
        self.compute_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.rmse_label.setText("")

    def polylines_preview(self):
        """已选点单独预览（复用 overlay.draw_points，无需场地线）。"""
        self.canvas.update()

    # ── 计算与保存 ───────────────────────────────────────
    def _compute(self):
        matrix, rmse = compute_homography(self.canvas.points)
        if matrix is None:
            self.rmse_label.setText("计算失败（点共线或退化），请调整点位")
            return
        self.matrix = matrix
        self.rmse_px = float(rmse)
        self.rmse_label.setText(f"重投影 RMSE = {rmse:.2f} px —— 满意则保存，否则撤销调整")
        # 场地线预览
        lines_px = []
        for a, b in FIELD_LINES_WORLD:
            seg = []
            for i in range(41):
                t = i / 40
                wx = a[0] + t * (b[0] - a[0])
                wy = a[1] + t * (b[1] - a[1])
                px, py = world_to_pixel(matrix, wx, wy)
                if px == px and py == py:  # 非 NaN
                    seg.append((px, py))
            if len(seg) > 1:
                lines_px.append(seg)
        self.canvas.polylines = lines_px
        self.canvas.update()
        self.save_btn.setEnabled(True)

    def _save(self):
        stem = Path(self.video_path).stem
        out = Path(__file__).resolve().parents[1] / "configs" / "homography" / f"{stem}.json"
        save_calibration(
            out,
            video=Path(self.video_path).name,
            image_size=list(self.video_size),
            field_size_m=[100.0, 37.0],
            calibration_frame=self.frame_idx,
            points=[{"pixel": [p[0], p[1]], "world": [p[2], p[3]], "label": self._picked[i][0]}
                    for i, p in enumerate(self.canvas.points)],
            matrix=self.matrix,
            reprojection_error_px=self.rmse_px,
        )
        self.saved_path = str(out)
        self.accept()
