"""事件复核写回（§13.16 B：event_review schema 冻结版）。

Schema（遵循 docs/conventions/naming-glossary.md 词汇）：
    doc["event_review"][str(event_key)] = {
        "review_status": "accepted" | "rejected" | "skipped",   # 流转状态
        "reviewer_decision": "score" | "not_score" | "uncertain",  # 语义结论
        "frame": int,           # 事件锚帧（冗余存，便于跨 run 对账）
        "event_type": str,      # 冗余存
        "note": str,            # 可选备注
    }

event_key = f"{event_type}:{frame}"（同帧同类型唯一；端区候选合并后以首帧为锚）。
统计层（tools/match_stats.py）消费：accepted/not_score 影响计数，skipped/缺失按
candidate 策略。纯函数+原子写，与 team.write_team_overrides 同模式。
"""
from __future__ import annotations

import json
from pathlib import Path

# reviewer_decision 允许值（事件域扩展 glossary 的 frisbee/not_frisbee 域）
DECISIONS = ("score", "not_score", "uncertain")
STATUSES = ("accepted", "rejected", "skipped")

_DECISION_TO_STATUS = {"score": "accepted", "not_score": "rejected", "uncertain": "skipped"}


def event_key(event: dict) -> str:
    return f"{event.get('type', '?')}:{int(event.get('frame', -1))}"


def write_event_review(doc: dict, event: dict, decision: str,
                       doc_path: str | Path | None = None, note: str = "") -> dict:
    """GUI 复核动作：写 event_review 并原子落盘（决策即落盘，同 team_overrides 模式）。

    decision: "score"（确认得分）/ "not_score"（拒绝）/ "uncertain"（跳过）。
    doc_path=None 时只改内存 doc（纯函数用法，供统计层/测试）。
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    key = event_key(event)
    doc.setdefault("event_review", {})[key] = {
        "review_status": _DECISION_TO_STATUS[decision],
        "reviewer_decision": decision,
        "frame": int(event.get("frame", -1)),
        "event_type": event.get("type", "?"),
        **({"note": note} if note else {}),
    }
    if doc_path is None:
        return doc
    p = Path(doc_path)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return doc


def review_of(doc: dict, event: dict) -> dict | None:
    """查某事件的复核结论（无则 None）。"""
    return (doc.get("event_review") or {}).get(event_key(event))


def review_summary(doc: dict) -> dict:
    """复核进度概览：各 decision 的计数。"""
    from collections import Counter
    entries = (doc.get("event_review") or {}).values()
    return dict(Counter(e.get("reviewer_decision") for e in entries))
