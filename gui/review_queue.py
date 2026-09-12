"""事件复核队列（§13.16 Phase D）：克隆 team_override 的"表格+行内控件+原子写"模式。

列出 candidate=true 的事件（得分候选为主），行内下拉选 reviewer_decision
（score/not_score/uncertain）即写 doc["event_review"]（event_review.py schema）
并原子落盘；双击行跳转该帧。写回后 emit reviewChanged 供主窗口触发 CPU 重算
（events-only）与比分板/时间线刷新（R8 修复的接线点）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from frisbee_analyzer.event_review import event_key, review_of, write_event_review

_DECISION_LABELS = {"score": "确认得分", "not_score": "拒绝", "uncertain": "跳过"}


class ReviewQueueDialog(QDialog):
    reviewChanged = Signal()
    frameRequested = Signal(int)

    def __init__(self, doc: dict, doc_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("事件复核队列（候选得分）")
        self.resize(680, 420)
        self.doc = doc
        self.doc_path = doc_path
        self._loading = True  # 行控件初始化期间不触发写回

        lay = QVBoxLayout(self)
        cands = [e for e in (doc.get("events") or []) if e.get("candidate")]
        info = QLabel(f"候选事件 {len(cands)} 个 — 下拉选择结论（即时落盘），双击行跳转画面")
        lay.addWidget(info)

        self.table = QTableWidget(len(cands), 6)
        self.table.setHorizontalHeaderLabels(["帧", "t/s", "类型", "端区", "复核结论", "详情"])
        self.table.setColumnWidth(5, 280)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_double_click)

        for r, e in enumerate(sorted(cands, key=lambda x: x.get("frame", 0))):
            vals = [str(e.get("frame", "")), f"{e.get('t_sec', 0):.1f}",
                    e.get("type", ""), str(e.get("endzone", "")),
                    "", str(e.get("detail", ""))[:80]]
            for c, v in enumerate(vals):
                if c == 4:
                    continue
                self.table.setItem(r, c, QTableWidgetItem(v))
            combo = QComboBox()
            combo.addItem("— 未复核 —", None)  # 无记录时的真实占位（防默认误写 score）
            for dec, label in _DECISION_LABELS.items():
                combo.addItem(label, dec)
            rev = review_of(doc, e)
            if rev:
                cur = rev.get("reviewer_decision")
                if cur:
                    combo.setCurrentIndex(1 + list(_DECISION_LABELS).index(cur))
            combo.currentIndexChanged.connect(
                lambda _i, ev=e, cb=combo: self._on_decision(ev, cb))
            self.table.setCellWidget(r, 4, combo)
            self.table.item(r, 0).setData(Qt.ItemDataRole.UserRole, e.get("frame", 0))

        lay.addWidget(self.table)
        row = QHBoxLayout()
        self.summary_label = QLabel(self._summary_text())
        row.addWidget(self.summary_label)
        lay.addLayout(row)
        self._loading = False

    def _summary_text(self):
        from frisbee_analyzer.event_review import review_summary
        s = review_summary(self.doc)
        return f"已复核: {s}" if s else "尚无复核记录"

    def _on_decision(self, event: dict, combo: QComboBox):
        if self._loading:
            return
        dec = combo.currentData()
        if dec is None:
            return  # 占位"未复核"不写
        try:
            write_event_review(self.doc, event, dec, self.doc_path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "复核", f"写回失败: {e}")
            return
        self.summary_label.setText(self._summary_text())
        self.reviewChanged.emit()

    def _on_double_click(self, row: int, _col: int):
        frame = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if frame is not None:
            self.frameRequested.emit(int(frame))
