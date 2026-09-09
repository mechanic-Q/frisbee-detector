"""Phase 3 验证：事件引擎在真实管线输出上的端到端表现（纯 CPU）。
输入: GUI worker 产出的 tracks.json（真实检测+跟踪+分队）
输出: results/engine_validation.json（引擎事件 vs E6 自动GT 的一致性）

运行: python tools/validate_engine_on_tracks.py [tracks.json]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frisbee_analyzer.events import (  # noqa: E402
    DiscObservation, EndZone, EventType, MatchEventEngine, Player, PossessionConfig,
)

ROOT = Path(__file__).resolve().parents[1]
TRACKS = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / ".worktrees/gui-v0/runs/gui_analysis/accept_10min/tracks.json"
GT = ROOT / "results/auto_gt_events.json"

# WFDF 场地（100x37，得分区 18m）
LEFT_EZ = EndZone(team_attacking=0, polygon=[(0, 0), (18, 0), (18, 37), (0, 37)])
RIGHT_EZ = EndZone(team_attacking=1, polygon=[(82, 0), (100, 0), (100, 37), (82, 37)])


def load_tracks():
    d = json.loads(TRACKS.read_text(encoding="utf-8"))
    return d


def analyze(doc):
    fps = doc.get("fps") or 30.0
    W, H = doc.get("width") or 1920, doc.get("height") or 1080
    frames = doc["frames"]
    overrides = {int(k): v for k, v in (doc.get("team_overrides") or {}).items()}

    # 单应性：用场地 4 角在画面中的近似位置（真实项目里来自标定 json）
    # 这里做「无标定」的保守代理：像素→米 用画面底部假设平面线性映射（仅用于验证引擎逻辑，
    # 精度不足以做真实测速，但足以验证事件状态机在真实轨迹上的行为）
    def px_to_world(x, y):
        # 画面 y 越大越近；假设画面底部 (y=H) 对应场地近边线 y_w=0，
        # 画面顶部 (y=0.25H) 对应远边线 y_w=37；x 线性映射到 0..100
        wy = (1.0 - (y - 0.25 * H) / (0.75 * H)) * 37.0
        wx = (x / W) * 100.0
        return wx, wy

    team_of = {}
    for fr in frames.values():
        for det in fr:
            tid = det["track_id"]
            t = overrides.get(tid, det.get("team_id"))
            if t is not None:
                team_of[tid] = t

    eng = MatchEventEngine(end_zones=[LEFT_EZ, RIGHT_EZ], team_of=team_of,
                           config=PossessionConfig(), fps=fps)
    all_events = []
    frames_sorted = sorted(frames, key=int)
    for fk in frames_sorted:
        idx = int(fk)
        players = []
        for det in frames[fk]:
            x1, y1, x2, y2 = det["bbox"]
            wx, wy = px_to_world((x1 + x2) / 2, y2)  # 脚点
            players.append(Player(track_id=det["track_id"],
                                  team_id=team_of.get(det["track_id"]),
                                  x=wx, y=wy))
        # 盘观测：本 run 没有飞盘检测 → 用「最高置信度球员」近似（验证引擎状态机鲁棒性）
        disc = None
        if players:
            top = max(frames[fk], key=lambda d: d["conf"])
            x1, y1, x2, y2 = top["bbox"]
            wx, wy = px_to_world((x1 + x2) / 2, y2)
            disc = DiscObservation(x=wx, y=wy, conf=top["conf"], frame=idx)
        evs = eng.update(idx, players, disc)
        all_events.extend(evs)

    return eng, all_events, team_of, frames_sorted


def main():
    doc = load_tracks()
    eng, events, team_of, frames_sorted = analyze(doc)
    counts = Counter(e.type.value for e in events)
    duration_s = len(frames_sorted) / (doc.get("fps") or 30.0)

    gt = json.loads(GT.read_text(encoding="utf-8")) if GT.exists() else {"events": []}
    gt_events = gt.get("events", [])
    gt_counts = Counter(e["type"] for e in gt_events)

    out = {
        "input": str(TRACKS),
        "frames": len(frames_sorted),
        "duration_s": round(duration_s, 1),
        "tracks": len(team_of),
        "engine_events": dict(counts),
        "engine_total": len(events),
        "gt_events": dict(gt_counts),
        "gt_window_note": "E6 GT 覆盖整场 47min；本 tracks 只覆盖其前若干分钟，数量不可直接比",
        "score_state": eng.score,
        "sample_events": [{"type": e.type.value, "frame": e.frame, "from": e.from_track,
                           "to": e.to_track, "team": e.team, "detail": e.detail}
                          for e in events[:15]],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/engine_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "sample_events"}, ensure_ascii=False, indent=1))
    print("\n--- 前 15 个事件 ---")
    for e in out["sample_events"]:
        print(f"  f{e['frame']:6d} {e['type']:16s} from={e['from']} to={e['to']} team={e['team']} {e['detail'][:30]}")


if __name__ == "__main__":
    main()
