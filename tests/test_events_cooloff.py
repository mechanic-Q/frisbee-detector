"""Phase C0 §13.16 得分锁冷却解锁单元测试。

运行: python -m pytest tests/test_events_runner.py::test_score_cooloff_unlock -v
   及全文件回归。
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


def _feed_with_cooloff(eng, n_frames, start=0):
    """模拟 events_runner 的冷却解锁循环：得分后 score_cooloff_frames 帧调 notify_pull。"""
    import math
    cfg = eng.cfg
    out = []
    last_score = None
    players = [Player(track_id=201, team_id=1, x=90.0, y=18.0)]
    for i in range(start, start + n_frames):
        for e in eng.update(i, players, DiscObservation(x=90.0, y=19.3, conf=0.9, frame=i)):
            out.append(e)
            if e.type is EventType.SCORE:
                last_score = i
        if (last_score is not None and i - last_score >= cfg.score_cooloff_frames
                and eng._score_lock):
            out.append(eng.notify_pull(i))
            last_score = None
    return out


def test_score_cooloff_unlocks_engine():
    """得分后冷却窗到期 → 自动 pull 解锁 → 引擎周而复始产出 possession/score。"""
    cfg = PossessionConfig(min_hold_frames=3, change_persist_frames=2,
                           score_cooloff_frames=50)
    eng = MatchEventEngine(end_zones=[RIGHT_EZ], team_of={201: 1}, config=cfg, fps=30.0)
    events = _feed_with_cooloff(eng, 200)
    types = [e.type for e in events]
    assert types.count(EventType.SCORE) >= 2, f"冷却解锁后应能再得分: {types}"
    # 末周期可能还在冷却中（窗口边缘），pull 数 = score 数或 score 数-1
    assert types.count(EventType.PULL) in (types.count(EventType.SCORE) - 1,
                                           types.count(EventType.SCORE))
    assert eng.score[1] == types.count(EventType.SCORE)
    # 每个周期内：score→pull 之间不应有 possession 泄漏（末周期无 pull 则跳过）
    pull_positions = [i for i, t in enumerate(types) if t is EventType.PULL]
    for i, t in enumerate(types):
        if t is EventType.SCORE:
            next_pulls = [j for j in pull_positions if j > i]
            if not next_pulls:
                break  # 末周期冷却未到期
            j = next_pulls[0]
            assert not any(x in (EventType.POSSESSION_START, EventType.TRANSFER)
                           for x in types[i + 1:j])


def test_lock_without_cooloff_is_permanent():
    """无冷却解锁的裸引擎 = 旧行为：首次得分后永久死球（C0 修复前的账面）。"""
    cfg = PossessionConfig(min_hold_frames=3, change_persist_frames=2)
    eng = MatchEventEngine(end_zones=[RIGHT_EZ], team_of={201: 1}, config=cfg, fps=30.0)
    players = [Player(track_id=201, team_id=1, x=90.0, y=18.0)]
    types = []
    for i in range(100):
        for e in eng.update(i, players, DiscObservation(x=90.0, y=19.3, conf=0.9, frame=i)):
            types.append(e.type)
    assert types.count(EventType.SCORE) == 1, "无解锁时只应有一次得分"
    assert eng._score_lock is True
