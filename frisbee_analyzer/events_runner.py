"""事件统计适配层：把 worker 产物（frames/disc_frames + 标定 json）喂给事件引擎。
纯 CPU、无 GPU 依赖——可在任何环境单测。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .events import DiscObservation, EndZone, EventType, MatchEventEngine, Player, PossessionConfig


def load_calibration(path: Path) -> tuple:
    """读项目标定 json → (H 单应矩阵 3x3, field_size_m, end_zone_depth_m)

    兼容字段名：homography / matrix（现有 tools/calibrate_field.py 产物用 matrix）。
    得分区深度：显式字段优先，否则按场地体系推断（WFDF 18m / USAU 25 码）。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = d.get("homography") or d.get("matrix")
    if raw is None:
        raise ValueError(f"calibration json missing homography/matrix: {path}")
    H = np.array(raw, dtype=np.float64).reshape(3, 3)
    fw, fh = d.get("field_size_m", [100.0, 37.0])
    if "end_zone_depth_m" in d:
        ez = float(d["end_zone_depth_m"])
    else:
        # WFDF 标准 18m；USAU 规则 25 码 ≈ 22.86m。默认 WFDF。
        ez = 22.86 if d.get("ruleset", "wfdf").lower() in ("usau", "usa") else 18.0
    return H, (float(fw), float(fh)), ez


def px_to_world(H: np.ndarray, x: float, y: float) -> tuple:
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def build_end_zones(field_w: float, field_h: float, ez_depth: float,
                    team_left: int = 0) -> list:
    """两端得分区：team_left 攻左区，另一队攻右区。"""
    right = 1 - team_left
    return [
        EndZone(team_attacking=team_left,
                polygon=[(0, 0), (ez_depth, 0), (ez_depth, field_h), (0, field_h)]),
        EndZone(team_attacking=right,
                polygon=[(field_w - ez_depth, 0), (field_w, 0),
                         (field_w, field_h), (field_w - ez_depth, field_h)]),
    ]


def compute_events(doc: dict, calibration_path: Path,
                   config: PossessionConfig | None = None) -> dict:
    """doc = worker 产物（含 frames/disc_frames/fps/team_overrides）。
    返回 {"events": [...], "score": {team: n}}。"""
    H, (fw, fh), ez = load_calibration(calibration_path)
    fps = float(doc.get("fps") or 30.0)
    frames = doc.get("frames", {})
    disc_frames = doc.get("disc_frames", {})
    overrides = {int(k): v for k, v in (doc.get("team_overrides") or {}).items()}

    team_of: dict[int, int] = {}
    for fr in frames.values():
        for det in fr:
            tid = det["track_id"]
            t = overrides.get(tid, det.get("team_id"))
            if t is not None:
                team_of[tid] = t

    eng = MatchEventEngine(
        end_zones=build_end_zones(fw, fh, ez),
        team_of=team_of,
        config=config or PossessionConfig(),
        fps=fps,
    )

    out_events = []
    for fk in sorted(frames, key=int):
        idx = int(fk)
        players = []
        for det in frames[fk]:
            x1, y1, x2, y2 = det["bbox"]
            wx, wy = px_to_world(H, (x1 + x2) / 2, y2)  # 脚点
            players.append(Player(track_id=det["track_id"],
                                  team_id=team_of.get(det["track_id"]),
                                  x=wx, y=wy))
        disc = None
        dd = disc_frames.get(fk)
        if dd is not None:
            wx, wy = px_to_world(H, dd["cx"], dd["cy"])
            disc = DiscObservation(x=wx, y=wy, conf=dd.get("conf", 1.0), frame=idx)
        for e in eng.update(idx, players, disc):
            out_events.append({
                "type": e.type.value, "frame": e.frame, "t_sec": round(e.frame / fps, 2),
                "from_track": e.from_track, "to_track": e.to_track, "team": e.team,
                "x": round(e.x, 2), "y": round(e.y, 2), "detail": e.detail,
            })
    return {"events": out_events, "score": eng.score}
