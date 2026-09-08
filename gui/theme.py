"""暗色 + macOS 毛玻璃风格主题（Fusion 基底 + 全局 QSS + Win10 Acrylic）。

apply_theme(app) 在创建 QApplication 后立即调用；enable_windows_acrylic(widget)
在主窗口 show 之后调用（ctypes 调 DWM，失败静默降级为普通深色，不影响功能）。
"""

from __future__ import annotations

ACCENT = "#0A84FF"          # macOS 系统蓝
TEXT = "#E8EAF0"
TEXT_DIM = "#9AA0AC"
GLASS = "rgba(28, 30, 38, 0.72)"        # 毛玻璃面板底（配合 DWM Acrylic）
GLASS_LIGHT = "rgba(255, 255, 255, 0.07)"
GLASS_HOVER = "rgba(255, 255, 255, 0.13)"
HAIRLINE = "rgba(255, 255, 255, 0.10)"
DARK = "#141619"


def build_qss() -> str:
    return f"""
* {{ font-family: "Microsoft YaHei UI", "PingFang SC", sans-serif; font-size: 13px; color: {TEXT}; }}

QMainWindow, QDialog {{
    background-color: rgba(22, 24, 30, 0.82);
}}
QWidget {{ color: {TEXT}; }}

QToolBar {{
    background: {GLASS};
    border: none;
    border-bottom: 1px solid {HAIRLINE};
    padding: 6px 10px;
    spacing: 6px;
}}
QToolButton {{
    background: {GLASS_LIGHT};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 6px 14px;
}}
QToolButton:hover {{ background: {GLASS_HOVER}; }}
QToolButton:pressed {{ background: rgba(10, 132, 255, 0.35); }}
QToolButton:disabled {{ color: {TEXT_DIM}; background: rgba(255, 255, 255, 0.03); }}

QPushButton {{
    background: {GLASS_LIGHT};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 7px 16px;
}}
QPushButton:hover {{ background: {GLASS_HOVER}; }}
QPushButton:pressed {{ background: rgba(10, 132, 255, 0.35); }}
QPushButton:disabled {{ color: {TEXT_DIM}; background: rgba(255, 255, 255, 0.03); }}
QPushButton#accent {{ background: {ACCENT}; border: none; color: white; }}
QPushButton#accent:hover {{ background: #2E96FF; }}

QDockWidget {{
    color: {TEXT_DIM};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background: transparent;
    border-bottom: 1px solid {HAIRLINE};
    padding: 4px 8px;
}}

QPlainTextEdit, QTextEdit, QTableWidget, QListView {{
    background: rgba(16, 18, 23, 0.72);
    border: 1px solid {HAIRLINE};
    border-radius: 10px;
    padding: 4px;
    selection-background-color: rgba(10, 132, 255, 0.45);
}}
QHeaderView::section {{
    background: rgba(255, 255, 255, 0.05);
    border: none;
    border-bottom: 1px solid {HAIRLINE};
    padding: 5px 8px;
    color: {TEXT_DIM};
}}
QTableWidget {{ gridline-color: rgba(255, 255, 255, 0.06); }}

QComboBox {{
    background: {GLASS_LIGHT};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 5px 10px;
}}
QComboBox:hover {{ background: {GLASS_HOVER}; }}
QComboBox QAbstractItemView {{
    background: #1D2027;
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    selection-background-color: rgba(10, 132, 255, 0.45);
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: rgba(255, 255, 255, 0.14);
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    background: #F2F4F8;
}}
QSlider::handle:horizontal:hover {{ background: white; }}

QProgressBar {{
    background: rgba(255, 255, 255, 0.10);
    border: none;
    border-radius: 5px;
    text-align: center;
    color: {TEXT};
    min-height: 10px;
    max-height: 14px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

QStatusBar {{
    background: {GLASS};
    border-top: 1px solid {HAIRLINE};
}}
QStatusBar QLabel {{ color: {TEXT_DIM}; }}

QToolTip {{
    background: #1D2027;
    color: {TEXT};
    border: 1px solid {HAIRLINE};
    border-radius: 6px;
    padding: 4px 8px;
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: rgba(255, 255, 255, 0.18); border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: rgba(255, 255, 255, 0.30); }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: rgba(255, 255, 255, 0.18); border-radius: 5px; min-width: 30px; }}
"""


def apply_theme(app) -> None:
    """Fusion 基底（QSS 跨原生风格一致）+ 雅黑 + 全局暗色毛玻璃 QSS。"""
    from PySide6.QtGui import QFont

    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(build_qss())


def enable_windows_acrylic(widget, tint_rgb=(20, 22, 28), tint_alpha: int = 168) -> None:
    """Win10/11 DWM Acrylic 毛玻璃（SetWindowCompositionAttribute）。

    失败（旧系统/精简版/离屏）静默忽略——QSS 的半透明面板仍呈现暗色玻璃观感。
    """
    import ctypes

    class _AccentPolicy(ctypes.Structure):
        _fields_ = [("AccentState", ctypes.c_int), ("AccentFlags", ctypes.c_int),
                    ("GradientColor", ctypes.c_uint), ("AnimationId", ctypes.c_int)]

    class _CompositionAttrData(ctypes.Structure):
        _fields_ = [("Attribute", ctypes.c_int), ("Data", ctypes.c_void_p),
                    ("SizeOfData", ctypes.c_size_t)]

    try:
        hwnd = int(widget.winId())
        r, g, b = tint_rgb
        # GradientColor = AABBGGRR
        gradient = (tint_alpha << 24) | (b << 16) | (g << 8) | r
        accent = _AccentPolicy(AccentState=4, AccentFlags=2, GradientColor=gradient)  # 4 = ACRYLICBLURBEHIND
        data = _CompositionAttrData(Attribute=19,
                                    Data=ctypes.cast(ctypes.pointer(accent), ctypes.c_void_p),
                                    SizeOfData=ctypes.sizeof(accent))
        ctypes.windll.user32.SetWindowCompositionAttribute(ctypes.c_void_p(hwnd), ctypes.byref(data))
    except Exception:
        pass
