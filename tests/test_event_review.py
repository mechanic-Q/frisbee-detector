"""event_review schema（§13.16 B 冻结版）单元测试。

运行: python -m pytest tests/test_event_review.py -v
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.event_review import (  # noqa: E402
    DECISIONS,
    event_key,
    review_of,
    review_summary,
    write_event_review,
)


def _evt(frame=1000, etype="score"):
    return {"type": etype, "frame": frame, "endzone": "right"}


def test_event_key_format():
    assert event_key({"type": "score", "frame": 1000}) == "score:1000"


def test_write_and_read_roundtrip(tmp_path):
    p = tmp_path / "tracks.json"
    doc = {"events": [_evt()]}
    write_event_review(doc, _evt(), "score", p, note="clear carry")
    assert p.exists()
    back = json.loads(p.read_text(encoding="utf-8"))
    rev = review_of(back, _evt())
    assert rev["review_status"] == "accepted"
    assert rev["reviewer_decision"] == "score"
    assert rev["frame"] == 1000 and rev["note"] == "clear carry"


def test_decision_mapping():
    doc = {}
    write_event_review(doc, _evt(1), "not_score")   # doc_path=None 纯内存
    write_event_review(doc, _evt(2), "uncertain")
    assert doc["event_review"]["score:1"]["review_status"] == "rejected"
    assert doc["event_review"]["score:2"]["review_status"] == "skipped"


def test_invalid_decision_raises(tmp_path):
    try:
        write_event_review({}, _evt(), "maybe", tmp_path / "x.json")
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_summary():
    doc = {}
    write_event_review(doc, _evt(1), "score")
    write_event_review(doc, _evt(2), "not_score")
    assert review_summary(doc) == {"score": 1, "not_score": 1}
    assert DECISIONS == ("score", "not_score", "uncertain")
