"""队伍改判对话框：track 级人工改队，改判立即写回 tracks.json（决策即落盘）。"""

from __future__ import annotations

import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from frisbee_analyzer.team import write_team_overrides

TEAM_OPTIONS = [(0, "红队"), (1, "蓝队"), (2, "裁判"), (3, "旁观")]
TEAM_NAMES = {0: "红队", 1: "蓝队", 2: "裁判", 3: "旁观"}


class TeamOverrideDialog(QDialog):
    teamsChanged = Signal()

    def __init__(self, doc: dict, doc_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("队伍改判（track 级，立即保存）")
        self.doc = doc
        self.doc_path = doc_path
        self.setMinimumSize(560, 480)

        stats = self._track_stats()
        hint = QLabel(f"共 {len(stats)} 个 track。改动下拉框立即生效并写回 JSON；"
                      "自动聚类的队号 0/1 与红蓝衫的对应关系可在此修正。")
        hint.setWordWrap(True)

        self.table = QTableWidget(len(stats), 4)
        self.table.setHorizontalHeaderLabels(["track_id", "出现帧数", "平均置信度", "队伍"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        for row, (tid, n_frames, mean_conf, team) in enumerate(stats):
            self.table.setItem(row, 0, QTableWidgetItem(str(tid)))
            self.table.setItem(row, 1, QTableWidgetItem(str(n_frames)))
            self.table.setItem(row, 2, QTableWidgetItem(f"{mean_conf:.2f}"))
            combo = QComboBox()
            combo.addItem("未分配", None)
            for value, name in TEAM_OPTIONS:
                combo.addItem(name, value)
            combo.setCurrentIndex(max(0, TEAM_OPTIONS.index(next(
                (o for o in TEAM_OPTIONS if o[0] == team), TEAM_OPTIONS[0])) + 1) if team in (0, 1, 2, 3) else 0)
            combo.currentIndexChanged.connect(lambda _i, t=tid, c=combo: self._on_change(t, c))
            self.table.setCellWidget(row, 3, combo)

        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(self.table)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.clicked.connect(self.close)
        lay.addWidget(btns)

    def _track_stats(self):
        """track_id → (帧数, 平均conf, 当前队)。"""
        acc: dict[int, dict] = {}
        for dets in self.doc.get("frames", {}).values():
            for det in dets:
                s = acc.setdefault(det["track_id"], {"n": 0, "conf": 0.0, "team": None})
                s["n"] += 1
                s["conf"] += det.get("conf", 0.0)
                if s["team"] is None and det.get("team_id") is not None:
                    s["team"] = det["team_id"]
        return [(tid, s["n"], s["conf"] / max(1, s["n"]), s["team"])
                for tid, s in sorted(acc.items())]

    def _on_change(self, track_id: int, combo: QComboBox):
        team = combo.currentData()  # None 或 0..3
        # 1) 内存 doc 全 track 生效
        for dets in self.doc.get("frames", {}).values():
            for det in dets:
                if det["track_id"] == track_id:
                    det["team_id"] = team
        # 2) overrides + 原子写盘
        if team is not None:
            write_team_overrides(self.doc, track_id, team, self.doc_path)
        else:
            self.doc.get("team_overrides", {}).pop(str(track_id), None)
            p = self.doc_path
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.doc, f, ensure_ascii=False)
            import os
            os.replace(tmp, p)
        self.teamsChanged.emit()
