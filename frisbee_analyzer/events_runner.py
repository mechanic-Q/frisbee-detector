"""事件统计适配层：把 worker 产物（frames/disc_frames + 标定 json）喂给事件引擎。
纯 CPU、无 GPU 依赖——可在任何环境单测。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .events import DiscObservation, EndZone, EventType, MatchEventEngine, Player, PossessionConfig


def load_calibration(path: Path, image_size: tuple | None = None) -> tuple:
    """读项目标定 json → (H 单应矩阵 3x3, field_size_m, end_zone_depth_m)

    兼容字段名：homography / matrix（现有 tools/calibrate_field.py 产物用 matrix）。
    得分区深度：显式字段优先，否则按场地体系推断（WFDF 18m / USAU 25 码）。
    image_size 传入时校验标定分辨率，不一致则自动缩放 H（标定在 A 分辨率做的，
    用于 B 分辨率视频时必须换算，否则世界坐标完全错位）。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = d.get("homography") or d.get("matrix")
    if raw is None:
        raise ValueError(f"calibration json missing homography/matrix: {path}")
    H = np.array(raw, dtype=np.float64).reshape(3, 3)

    cal_size = d.get("image_size")
    if image_size and cal_size and len(cal_size) == 2:
        cw, ch = float(cal_size[0]), float(cal_size[1])
        vw, vh = float(image_size[0]), float(image_size[1])
        if abs(cw - vw) > 1 or abs(ch - vh) > 1:
            sx, sy = vw / cw, vh / ch
            S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]], dtype=np.float64)
            H = H @ S  # 像素缩放后作用于世界映射

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


def detect_endzone_carries(doc: dict, H: np.ndarray, fw: float, fh: float,
                           ez_depth: float, fps: float,
                           min_frames: int = 15, carry_speed_ms: float = 2.5,
                           margin_m: float = 1.0,
                           attended_radius_m: float = 2.5,
                           attended_frac: float = 0.0) -> list[dict]:
    """端区慢速持盘走段检测 → 得分候选（进复核队列，不自动记分）。

    多维互证的几何维度（F1 G4 实验 §13.10）：持盘状态机要求"盘-手部点 <1.2m"，
    但得分后持盘走到底线的动作中盘在腰侧、手部点模型不匹配 → 状态机漏检。
    本规则只看几何：盘在端区(±margin)内以步行速度持续 ≥min_frames 帧 →
    score_candidate（candidate=true，供人工/上层复核确认）。

    有人持有判别（§13.13 迭代2→3）：慢速端区段有三类伪影——落地躺盘、拉盘者
    端区等待、得分后走回。原计划用"盘 2.5m 内有球员"过滤躺盘，实测不可行：
    真得分段盘-最近球员世界距离实测 6.7-9.9m——单应性地平面视差（持盘离地
    ~1m，随相机距离放大）+ players-extent 标定端区区误差叠加，世界坐标近距
    判别噪声 ~7m，任何阈值都分不开持有/落地。故 attended 降级为**软信号**：
    恒计算恒上报（attended_frac 进 detail 供复核队列参考），不参与过滤
    （attended_frac=0 即关闭过滤，默认如此）。拉盘者等待/走回伪影由上层
    90s 新颖性链标记 followup 兜住。
    """
    import math

    disc = doc.get("disc_frames", {})
    if not disc:
        return []

    # 球员脚点世界坐标缓存（与 compute_events 同口径：bbox 底边中点）
    players_world: dict[int, list[tuple[float, float]]] = {}
    for fk, dets in doc.get("frames", {}).items():
        ws = []
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            ws.append(px_to_world(H, (x1 + x2) / 2, y2))
        players_world[int(fk)] = ws

    seq = []
    for fk in sorted(disc, key=int):
        d = disc[fk]
        if d.get("status") == "rejected":  # 速度物理拒绝的观测不可信
            continue
        wx, wy = px_to_world(H, d["cx"], d["cy"])
        near = any(math.hypot(wx - px, wy - py) <= attended_radius_m
                   for px, py in players_world.get(int(fk), ()))
        seq.append((int(fk), wx, wy, near))
    # 慢速连续段：帧号断裂 >2 或速度超限则断开
    segments, cur, prev = [], [], None
    for fn, wx, wy, _near in seq:
        if prev is not None:
            gap = fn - prev[0]
            sp = math.hypot(wx - prev[1], wy - prev[2]) / max(gap / fps, 1e-6)
            if gap > 2 or sp > carry_speed_ms:
                if len(cur) >= min_frames:
                    segments.append(cur)
                cur = []
        cur.append((fn, wx, wy, _near))
        prev = (fn, wx, wy)
    if len(cur) >= min_frames:
        segments.append(cur)

    lo, hi = ez_depth + margin_m, fw - ez_depth - margin_m
    cands = []
    for seg in segments:
        in_ez = [p for p in seg if p[1] < lo or p[1] > hi]
        if len(in_ez) < min_frames:
            continue
        attended = sum(1 for p in in_ez if p[3]) / len(in_ez)
        if attended < attended_frac:
            continue  # 落地躺盘等无人贴身伪影
        side = "left" if in_ez[len(in_ez) // 2][1] < lo else "right"
        cands.append({
            "type": "score", "candidate": True,
            "frame": seg[0][0], "t_sec": round(seg[0][0] / fps, 2),
            "end_frame": seg[-1][0], "endzone": side,
            "carry_frames": len(in_ez),
            "attended_frac": round(attended, 2),
            "detail": f"endzone carry: disc slow in {side} EZ x[{seg[0][1]:.0f}->{seg[-1][1]:.0f}] "
                      f"≥{min_frames}f attended={attended:.0%} — review for score",
        })
    # 合并：同端区 30s 内的碎片候选（行走中的速度毛刺会切段）聚为一个得分候选
    merged = []
    for c in sorted(cands, key=lambda x: x["t_sec"]):
        if merged and c["endzone"] == merged[-1]["endzone"] \
                and c["t_sec"] - merged[-1]["end_t_sec"] <= 30:
            m = merged[-1]
            m["end_t_sec"] = c["t_sec"]
            m["carry_frames"] += c["carry_frames"]
            m["attended_frac"] = round(min(m["attended_frac"], c["attended_frac"]), 2)
            m["segments"] = m.get("segments", 1) + 1
            m["detail"] = (f"endzone carry: {m['segments']} slow-carry segments in {m['endzone']} EZ, "
                           f"total {m['carry_frames']}f attended={m['attended_frac']:.0%} — review for score")
        else:
            c = dict(c)
            c["end_t_sec"] = c["t_sec"]
            c["segments"] = 1
            merged.append(c)
    for m in merged:
        m["end_t_sec"] = round(m["end_t_sec"], 2)
    return merged


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
    vsize = (doc.get("width"), doc.get("height")) if doc.get("width") else None
    H, (fw, fh), ez = load_calibration(calibration_path, image_size=vsize)
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
    margin = 5.0  # 米：场地外缓冲
    disc_rejected = 0
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
            if -margin <= wx <= fw + margin and -margin <= wy <= fh + margin:
                disc = DiscObservation(x=wx, y=wy, conf=dd.get("conf", 1.0), frame=idx,
                                       source=dd.get("source"))
            else:
                disc_rejected += 1
        for e in eng.update(idx, players, disc):
            out_events.append({
                "type": e.type.value, "frame": e.frame, "t_sec": round(e.frame / fps, 2),
                "from_track": e.from_track, "to_track": e.to_track, "team": e.team,
                "x": round(e.x, 2), "y": round(e.y, 2), "detail": e.detail,
                "candidate": e.candidate,
            })

    # F1 多维互证·几何维度: 端区慢速持盘走段 → 得分候选（复核队列）
    for cand in detect_endzone_carries(doc, H, fw, fh, ez, fps):
        out_events.append(cand)
    return {"events": out_events, "score": eng.score,
            "disc_rejected_out_of_field": disc_rejected}
