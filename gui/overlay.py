"""叠加层绘制：球员框+队伍色、标定点、场地线。坐标换算（letterbox）供点击反查复用。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF

# team_id 编码与 frisbee_analyzer/tools 惯例一致：0=红 1=蓝 2=裁判 3=旁观，None=未分配
TEAM_COLORS = {
    0: QColor(255, 72, 72),
    1: QColor(80, 148, 255),
    2: QColor(255, 214, 64),
    3: QColor(170, 170, 170),
}
TEAM_NAMES = {0: "红队", 1: "蓝队", 2: "裁判", 3: "旁观"}
UNKNOWN_COLOR = QColor(230, 230, 230)


def letterbox(video_w: float, video_h: float, widget_w: float, widget_h: float):
    """等比缩放并居中：返回 (scale, offset_x, offset_y)。"""
    if video_w <= 0 or video_h <= 0:
        return 1.0, 0.0, 0.0
    scale = min(widget_w / video_w, widget_h / video_h)
    return scale, (widget_w - video_w * scale) / 2, (widget_h - video_h * scale) / 2


def widget_to_video(px: float, py: float, video_w: float, video_h: float,
                    widget_w: float, widget_h: float) -> tuple[float, float]:
    scale, dx, dy = letterbox(video_w, video_h, widget_w, widget_h)
    return (px - dx) / scale, (py - dy) / scale


def draw_detections(painter: QPainter, dets: list[dict], video_w: float, video_h: float,
                    widget_w: float, widget_h: float, team_colors: dict | None = None,
                    line_width: float = 2.0) -> None:
    """按 widget 尺寸画当前帧的球员框。team_colors = doc["team_colors"]（如 {"0":"red"}）。"""
    from PySide6.QtGui import QFontMetrics

    scale, dx, dy = letterbox(video_w, video_h, widget_w, widget_h)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont()
    font.setPixelSize(max(11, int(12 * scale * 0.6) + 8))
    painter.setFont(font)
    metrics = QFontMetrics(font)
    named = {"red": QColor(255, 72, 72), "blue": QColor(80, 148, 255),
             "other": QColor(170, 170, 170)}
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        team = det.get("team_id")
        color = named.get((team_colors or {}).get(str(team)))
        if color is None:
            color = TEAM_COLORS.get(team, UNKNOWN_COLOR)
        pen = QPen(color, line_width)
        painter.setPen(pen)
        rect = QRectF(dx + x1 * scale, dy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale)
        painter.drawRect(rect)
        team_label = TEAM_NAMES.get(team, "未分配") if team is not None else "未分配"
        text = f"#{det['track_id']} {team_label} {det.get('conf', 0):.2f}"
        # 深色药丸底 + 彩色左缘：保证亮画面上的标签可读
        tw = metrics.horizontalAdvance(text)
        th = metrics.height()
        pill_top = rect.top() - th - 8
        if pill_top < 2:
            pill_top = rect.top() + 2
        pill = QRectF(rect.left(), pill_top, tw + 12, th + 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 175))
        painter.drawRoundedRect(pill, 4, 4)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 1))
        painter.drawLine(QPointF(pill.left(), pill.top()), QPointF(pill.left(), pill.bottom()))
        painter.setPen(QColor("#F2F4F8"))
        painter.drawText(QPointF(pill.left() + 6, pill_top + th), text)


def draw_points(painter: QPainter, points: list[tuple[float, float]], video_w, video_h,
                widget_w, widget_h, color: QColor = QColor(64, 255, 128)) -> None:
    """标定点（视频坐标）绘制为圆点+序号。"""
    scale, dx, dy = letterbox(video_w, video_h, widget_w, widget_h)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont()
    font.setPixelSize(13)
    painter.setFont(font)
    for i, (vx, vy) in enumerate(points, start=1):
        cx, cy = dx + vx * scale, dy + vy * scale
        painter.setPen(QPen(color, 2))
        painter.drawEllipse(QPointF(cx, cy), 6, 6)
        painter.drawText(QPointF(cx + 9, cy - 6), str(i))


def draw_endzones(painter: QPainter, matrix, video_w: float, video_h: float,
                  widget_w: float, widget_h: float) -> None:
    """标定存在时渲染两端得分区（半透明填充）与得分线（goal lines）。

    matrix 为像素→世界单应性（utils.homography 的 3x3）。
    """
    from utils.homography import ENDZONES_WORLD, world_to_pixel

    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for i, poly in enumerate(ENDZONES_WORLD):
        pts_px = []
        for wx, wy in poly:
            px, py = world_to_pixel(matrix, wx, wy)
            if px == px and py == py:
                pts_px.append((px, py))
        if len(pts_px) < 3:
            continue
        scale, dx, dy = letterbox(video_w, video_h, widget_w, widget_h)
        fill = QColor(255, 72, 72, 26) if i == 0 else QColor(80, 148, 255, 26)
        painter.setPen(QPen(QColor(242, 244, 248, 160), 2))
        painter.setBrush(fill)
        polygon = QPolygonF([QPointF(dx + x * scale, dy + y * scale) for x, y in pts_px])
        painter.drawPolygon(polygon)
    # 得分线加亮（边线已由 draw_polylines 的场地线覆盖）
    goal_px = []
    for a, b in [(18, 18), (82, 82)]:
        seg = []
        for i in range(21):
            t = i / 20
            px, py = world_to_pixel(matrix, a, t * 37)
            if px == px and py == py:
                seg.append((px, py))
        if len(seg) > 1:
            goal_px.append(seg)
    draw_polylines(painter, goal_px, video_w, video_h, widget_w, widget_h,
                   color=QColor(242, 244, 248, 200), width=2.6)


def draw_polylines(painter: QPainter, polylines_px: list[list[tuple[float, float]]],
                   video_w, video_h, widget_w, widget_h,
                   color: QColor = QColor(64, 255, 128), width: float = 1.6) -> None:
    """画若干条视频坐标折线（如标定后的场地线投影）。"""
    scale, dx, dy = letterbox(video_w, video_h, widget_w, widget_h)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(color, width))
    for line in polylines_px:
        for a, b in zip(line, line[1:]):
            painter.drawLine(QPointF(dx + a[0] * scale, dy + a[1] * scale),
                             QPointF(dx + b[0] * scale, dy + b[1] * scale))
