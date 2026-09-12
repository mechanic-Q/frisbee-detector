"""源感知记分门单元测试（§13.14/13.15：imputed 观测保护窗内得分降级 candidate）。

运行: python -m pytest tests/test_events_source_guard.py -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.events import (  # noqa: E402
    DiscObservation,
    EndZone,
    EventType,
    MatchEventEngine,
    Player,
    PossessionConfig,
)

RIGHT_EZ = EndZone(team_attacking=1, polygon=[(82, 0), (100, 0), (100, 37), (82, 37)])
CFG = PossessionConfig(min_hold_frames=3, change_persist_frames=2)


def make_engine():
    return MatchEventEngine(end_zones=[RIGHT_EZ], team_of={201: 1}, config=CFG, fps=30.0)


def feed(eng, frames):
    out = []
    for f, disc_src in frames:
        disc = DiscObservation(x=90.0, y=19.3, conf=0.9, frame=f, source=disc_src)
        players = [Player(track_id=201, team_id=1, x=90.0, y=18.0)]
        out.extend(eng.update(f, players, disc))
    return out


def test_pure_real_possession_scores_normally():
    eng = make_engine()
    evs = feed(eng, [(f, None) for f in range(100, 110)])
    scores = [e for e in evs if e.type == EventType.SCORE]
    assert len(scores) == 1 and scores[0].candidate is False
    assert eng.score == {0: 0, 1: 1}
    assert eng._score_lock is True


def test_imputed_within_guard_demotes_score_to_candidate():
    eng = make_engine()
    frames = [(100, "possession_imputed")]          # streak 首帧为合成观测
    frames += [(f, None) for f in range(101, 112)]
    evs = feed(eng, frames)
    scores = [e for e in evs if e.type == EventType.SCORE]
    assert len(scores) == 1
    assert scores[0].candidate is True              # 降级为候选
    assert "imputed-guard" in scores[0].detail
    assert eng.score == {0: 0, 1: 0}                # 不计入比分
    assert eng._score_lock is True                  # 仍上锁（死球到 pull）


def test_guard_expires_after_window():
    eng = make_engine()
    feed(eng, [(100, "possession_imputed")])
    # 102 帧后（>90）才出现的持盘得分不受影响
    evs = feed(eng, [(f, None) for f in range(202, 212)])
    scores = [e for e in evs if e.type == EventType.SCORE]
    assert len(scores) == 1 and scores[0].candidate is False
    assert eng.score == {0: 0, 1: 1}


def test_pull_clears_guard():
    eng = make_engine()
    feed(eng, [(100, "possession_imputed")])
    eng.notify_pull(101)
    assert eng._imputed_seen_frame is None
    evs = feed(eng, [(f, None) for f in range(103, 110)])
    scores = [e for e in evs if e.type == EventType.SCORE]
    assert len(scores) == 1 and scores[0].candidate is False
    assert eng.score == {0: 0, 1: 1}
