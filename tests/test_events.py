"""frisbee_analyzer 事件引擎单元测试（合成轨迹，无 CV 依赖）。

运行: python3 -m pytest tests/test_events.py -v
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

# WFDF 场地: 100m x 37m, 得分区各 18m 深
LEFT_EZ = EndZone(team_attacking=0, polygon=[(0, 0), (18, 0), (18, 37), (0, 37)])
RIGHT_EZ = EndZone(team_attacking=1, polygon=[(82, 0), (100, 0), (100, 37), (82, 37)])


def make_engine(**cfg) -> MatchEventEngine:
    # §13.16 B 后引擎会就地翻转 end_zones（USAU 9.B）——EZ 每次新建，
    # 防止测试间共享可变模块级单例造成状态泄漏。
    return MatchEventEngine(
        end_zones=[
            EndZone(team_attacking=0, polygon=[(0, 0), (18, 0), (18, 37), (0, 37)]),
            EndZone(team_attacking=1, polygon=[(82, 0), (100, 0), (100, 37), (82, 37)]),
        ],
        team_of={101: 0, 102: 0, 201: 1, 202: 1},
        config=PossessionConfig(**cfg) if cfg else PossessionConfig(),
    )


def feed(eng, frames, players_by_frame, disc_by_frame):
    out = []
    for f in frames:
        ev = eng.update(f, players_by_frame.get(f, []), disc_by_frame.get(f))
        out.extend(ev)
    return out


def player_at(tid, team, x, y):
    return Player(track_id=tid, team_id=team, x=x, y=y)


def disc_at(f, x, y, speed_ms=0.0):
    """speed_ms 通过相邻帧位移实现（1 帧间隔）。"""
    return DiscObservation(x=x, y=y, frame=f)


def test_completed_pass_counts_transfer():
    """同队 A(101) 持盘 10 帧 -> 飞行 -> 同队 B(102) 接住持 10 帧 = 1 次交换手, 0 turnover."""
    eng = make_engine()
    players = {f: [player_at(101, 0, 30, 18)] for f in range(0, 10)}
    disc = {f: disc_at(f, 30, 18 + 1.3) for f in range(0, 10)}  # 盘在手部点附近静止
    evs0 = feed(eng, range(0, 10), players, disc)
    assert eng.holder == 101
    assert [e.type for e in evs0].count(EventType.POSSESSION_START) == 1

    # 3 帧快速飞行到 40m 外的队友
    traj = [(32, 18.6), (36, 19.0), (40, 19.2)]
    evs = []
    for i, (x, y) in enumerate(traj):
        f = 10 + i
        evs += eng.update(f, [player_at(101, 0, 30, 18), player_at(102, 0, 40, 19)], disc_at(f, x, y))
    # 接住后静止持 10 帧
    for f in range(13, 24):
        evs += eng.update(f, [player_at(101, 0, 30, 18), player_at(102, 0, 40, 19)], disc_at(f, 40, 20.5))

    types = [e.type for e in evs]
    assert types.count(EventType.TRANSFER) == 1
    assert types.count(EventType.TURNOVER) == 0
    assert eng.holder == 102
    tr = next(e for e in evs if e.type == EventType.TRANSFER)
    assert tr.from_track == 101 and tr.to_track == 102


def test_drop_then_pickup_by_other_team_is_turnover():
    """101 持盘 -> 盘落地(低速无人) -> 对方 201 拾起 = 1 turnover."""
    eng = make_engine()
    players = {f: [player_at(101, 0, 30, 18), player_at(201, 1, 33, 18)] for f in range(0, 40)}
    # 0-9: 持盘
    disc = {f: disc_at(f, 30, 19.3) for f in range(0, 10)}
    # 10-11: 掉盘小位移
    disc[10] = disc_at(10, 31.0, 19.2)
    disc[11] = disc_at(11, 31.6, 19.2)
    # 12-30: 落地静止（在 201 脚边附近但手部点距离>阈值? 让 201 站远些）
    players[12] = [player_at(101, 0, 30, 18), player_at(201, 1, 60, 18)]
    for f in players:
        players[f] = players[12] if f >= 12 else players[f]
    for f in range(12, 31):
        disc[f] = disc_at(f, 31.6, 19.2)
    evs = feed(eng, range(0, 31), players, disc)
    types = [e.type for e in evs]
    assert types.count(EventType.DISC_ON_GROUND) == 1
    # 31-45: 201 跑回来拾起并持盘
    evs2 = []
    for f in range(31, 46):
        evs2 += eng.update(f, [player_at(101, 0, 30, 18), player_at(201, 1, 31.6, 18.2)],
                           disc_at(f, 31.6, 19.5))
    types2 = [e.type for e in evs2]
    assert types2.count(EventType.TURNOVER) == 1
    assert eng.holder == 201


def test_catch_in_end_zone_scores_once_and_locks():
    """0 队球员在其进攻端区(左区 x<18)内接住 = 1 分, 且得分锁防止重复计分直到 pull."""
    eng = make_engine()
    players = {f: [player_at(101, 0, 10, 18)] for f in range(0, 25)}
    disc = {f: disc_at(f, 10, 19.3) for f in range(0, 25)}
    evs = feed(eng, range(0, 25), players, disc)
    scores = [e for e in evs if e.type == EventType.SCORE]
    assert len(scores) == 1
    assert eng.score == {0: 1, 1: 0}
    # 得分后 holder 清空并加锁
    assert eng.holder is None
    # 继续 50 帧（盘还在端区里）不再计分
    evs2 = feed(eng, range(25, 75), players, disc)
    assert not [e for e in evs2 if e.type == EventType.SCORE]
    # pull 后解锁
    eng.notify_pull(80)
    assert eng._score_lock is False


def test_referee_never_holds_disc():
    """team_id=-1 的裁判不会被选为持盘人."""
    eng = make_engine()
    players = {f: [player_at(999, -1, 30, 18)] for f in range(0, 20)}
    disc = {f: disc_at(f, 30, 19.3) for f in range(0, 20)}
    evs = feed(eng, range(0, 20), players, disc)
    assert eng.holder is None
    assert not [e for e in evs if e.type == EventType.POSSESSION_START]


def test_transfer_flicker_is_suppressed():
    """单帧误匹配到别人不应切换持盘人（惯性 4 帧）."""
    eng = make_engine()
    players = {f: [player_at(101, 0, 30, 18), player_at(102, 0, 30.5, 18)] for f in range(0, 30)}
    disc = {f: disc_at(f, 30, 19.3) for f in range(0, 30)}
    feed(eng, range(0, 10), players, disc)
    assert eng.holder == 101
    # 第 10 帧瞬移到 102 处, 第 11 帧又回到 101 —— 不应 TRANSFER
    eng.update(10, players[10], disc_at(10, 30.5, 19.3))
    eng.update(11, players[11], disc_at(11, 30, 19.3))
    eng.update(12, players[12], disc_at(12, 30, 19.3))
    trans = [e for e in eng.last_events if e.type == EventType.TRANSFER]
    assert not trans
    assert eng.holder == 101


def test_score_counts_accumulate_over_multiple_possessions():
    """两次得分（pull 解锁后）累计 2:0。

    §13.16 B（USAU 9.B）：首次得分后方向翻转（team0 转攻右端区），
    第二次得分须在右端区完成。
    """
    eng = make_engine()
    players = {f: [player_at(101, 0, 10, 18)] for f in range(0, 80)}
    disc = {f: disc_at(f, 10, 19.3) for f in range(0, 80)}
    feed(eng, range(0, 25), players, disc)
    assert eng.score == {0: 1, 1: 0}
    assert eng.holder is None  # 死球期间不重选持盘人
    assert eng.ezs[0].team_attacking == 1  # team0 现攻右端区（方向已翻转）
    eng.notify_pull(30)
    players2 = {f: [player_at(101, 0, 91, 18)] for f in range(40, 80)}
    disc2 = {f: disc_at(f, 91, 19.3) for f in range(40, 80)}
    feed(eng, range(40, 65), players2, disc2)
    assert eng.score == {0: 2, 1: 0}
    assert eng.ezs[0].team_attacking == 0  # 再次翻转回来
