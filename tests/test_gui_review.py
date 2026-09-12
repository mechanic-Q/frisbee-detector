"""Phase D §13.16 GUI 组件单元测试（offscreen）。

运行: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_review.py -v
（PySide6 必须在 py -3.11；本测试假定运行环境已装）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

from frisbee_analyzer.event_review import review_of  # noqa: E402
from gui.review_queue import ReviewQueueDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


def _doc_with_candidates(tmp_path):
    doc = {
        "video": "v.mp4", "fps": 30.0, "width": 852, "height": 480,
        "frames": {str(i): [] for i in range(100)},
        "score": {"0": 1, "1": 0},
        "team_colors": {"0": "red", "1": "blue"},
        "events": [
            {"type": "score", "candidate": True, "frame": 30, "t_sec": 1.0,
             "endzone": "right", "detail": "carry"},
            {"type": "disc_on_ground", "candidate": False, "frame": 50, "t_sec": 1.7},
            {"type": "score", "candidate": True, "frame": 80, "t_sec": 2.7,
             "endzone": "left", "detail": "carry2"},
        ],
    }
    p = tmp_path / "tracks.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return doc, p


def test_review_dialog_lists_candidates_and_writes(app, tmp_path):
    doc, p = _doc_with_candidates(tmp_path)
    dlg = ReviewQueueDialog(doc, str(p))
    assert dlg.table.rowCount() == 2  # 只列 candidate
    # 默认应为"— 未复核 —"占位（防误写）
    combo = dlg.table.cellWidget(0, 4)
    assert combo.currentData() is None
    # 模拟选"确认得分"（占位项后 score=1）
    combo.setCurrentIndex(1)
    back = json.loads(p.read_text(encoding="utf-8"))
    rev = review_of(back, {"type": "score", "frame": 30})
    assert rev is not None and rev["reviewer_decision"] == "score"


def test_review_double_click_emits_seek(app, tmp_path):
    doc, p = _doc_with_candidates(tmp_path)
    dlg = ReviewQueueDialog(doc, str(p))
    got = []
    dlg.frameRequested.connect(got.append)
    dlg._on_double_click(1, 0)
    assert got == [80]  # 第二行候选的帧号
