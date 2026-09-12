"""Phase B §13.16 方向管理单元测试：USAU 9.B 逐分翻转 / 半场恢复 / team_left 反推。

运行: python -m pytest tests/test_events_direction.py -v
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.events import DiscObservation, EventType, MatchEventEngine, Player, PossessionConfig  # noqa: E402
from frisbee_analyzer.events_runner import (  # noqa: E402
    build_end_zones,
    compute_events,
    flip_end_zones,
    infer_team_left,
)


def _ez_states(ezs):
    return [z.team_attacking for z in ezs]


def test_flip_end_zones_swaps_attacking():
    ezs = build_end_zones(100.0, 37.0, 18.0, team_left=0)
    assert _ez_states(ezs) == [0, 1]
    flip_end_zones(ezs)
    assert _ez_states(ezs) == [1, 0]
    flip_end_zones(ezs)
    assert _ez_states(ezs) == [0, 1]


def test_engine_flips_after_counted_score():
    """计入比分的得分后端区翻转（USAU 9.B）；candidate 得分不翻转。"""
    ezs = build_end_zones(100.0, 37.0, 18.0, team_left=0)
    cfg = PossessionConfig(min_hold_frames=2, change_persist_frames=1)
    eng = MatchEventEngine(end_zones=ezs, team_of={5: 0}, config=cfg, fps=30.0)
    # team0 持盘站左端区（初始 team0 攻左）→ 得分 → 翻转后 team0 攻右
    players = [Player(track_id=5, team_id=0, x=9.0, y=18.0)]
    got = []
    for i in range(6):
        got += eng.update(i, players, DiscObservation(x=9.0, y=19.3, conf=0.9, frame=i))
    assert any(e.type is EventType.SCORE for e in got)
    assert eng.score == {0: 1, 1: 0}
    assert _ez_states(ezs) == [1, 0], "counted 得分后应翻转"
    assert eng._score_lock is True  # 未配冷却时保持锁（C0 在 runner 层解锁）


def test_infer_team_left_voting():
    """确认候选反推 team_left：自洽对（left:t0 + right:t1）投 0；矛盾对平票 None。"""
    cands = [
        {"endzone": "left", "team": 0},
        {"endzone": "left", "team": 0},
        {"endzone": "right", "team": 1},  # team1 攻右 → team_left=0，与上两条自洽
    ]
    assert infer_team_left(cands) == 0
    # 同一队被记攻两端 = 证据矛盾 → 平票 None
    assert infer_team_left([{"endzone": "left", "team": 0},
                            {"endzone": "right", "team": 0}]) is None
    assert infer_team_left([]) is None


def test_compute_events_team_left_from_calib(tmp_path):
    """compute_events 读标定 json 的 team_left 可选键并透传输出。"""
    calib = tmp_path / "cal.json"
    calib.write_text(json.dumps({
        "image_size": [852, 480], "field_size_m": [100.0, 37.0],
        "matrix": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1]],  # px→world: /10
        "team_left": 1,
    }), encoding="utf-8")
    doc = {
        "video": "v.mp4", "fps": 30.0, "width": 852, "height": 480,
        "frames": {}, "disc_frames": {}, "team_overrides": {},
    }
    out = compute_events(doc, calib)
    assert out["team_left"] == 1
    assert out["events"] == []
