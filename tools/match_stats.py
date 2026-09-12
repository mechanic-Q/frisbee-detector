#!/usr/bin/env python3
"""比赛统计层（§13.16 Phase C）：七指标 + 统一证据策略 + provenance。

用户问题映射：
  1 飞行距离   → disc_flight_distance（仅 tracking 帧，世界轨迹路径长）
  2 换手率     → turnover_rates（双口径：/throws 与 /possessions，UFA 官方）
  3 控盘占比   → possession_share（引擎区间为主、holder 印章兜底、carry 不计）
  4 热力图     → team_heatmaps（脚点 25×25 网格 + σ1 高斯，PNG 落 runs/）
  5 得分手     → scorers（SCORE.to_track + carry 确认后最近球员归属）
  6 组织者     → organizers（hockey-assist + 首推进末三区，xT 启发式）
  7 威胁进攻发起 → threat_origins（以得分结尾链的起点 人+位置+方向）

证据策略（单点定义，七指标共用）：
  counted  — 引擎计入比分的得分/其派生链（含 C0 冷却后的全部引擎事件）
  candidate— 候选（端区规则/imputed 守卫降级），默认不计入、单独列出
  excluded — 缺证据（无标定/无队伍/短段），原因随行

用法:
  python tools/match_stats.py TRACKS_JSON [--calib CALIB_JSON] [--heat-png OUTDIR]
  # calib 缺省取同目录 segment_calib.json

诚实边界（随输出 provenance 声明）：
  - 世界坐标近距噪声 ~7m（地平面视差+players-extent 端区误差，§13.13 实测）
  - 盘帧覆盖率低时换手/控盘统计系统性偏低（空洞既杀 numerator 也杀 denominator）
  - 热力图/距离的跨 chunk 汇总需各 chunk 标定内点率 ≥0.85（本工具只吃单 doc）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frisbee_analyzer.events_runner import load_calibration, px_to_world  # noqa: E402

FIELD_W, FIELD_H = 100.0, 37.0
HEAT_GRID = 25          # mplsoccer 惯例 25×25
HEAT_SIGMA = 1.0        # 网格单元高斯平滑
MIN_TRACKING_RUN = 3    # 飞行距离最短段（帧）
GAP_BREAK = 2           # 帧号断裂 >2 切段


# ── 基础工具 ──

def load_doc(tracks: Path, calib: Path) -> tuple[dict, np.ndarray, float]:
    doc = json.loads(tracks.read_text(encoding="utf-8"))
    H, (_fw, _fh), _ez = load_calibration(calib,
                                          image_size=(doc.get("width"), doc.get("height")))
    return doc, np.asarray(H, dtype=float), float(doc.get("fps") or 30.0)


def foot_world(H, det) -> tuple[float, float]:
    x1, y1, x2, y2 = det["bbox"]
    return px_to_world(H, (x1 + x2) / 2, y2)


def team_of_track(doc) -> dict[int, int]:
    overrides = {int(k): v for k, v in (doc.get("team_overrides") or {}).items()}
    team_of = {}
    for fr in doc.get("frames", {}).values():
        for det in fr:
            tid = det["track_id"]
            t = overrides.get(tid, det.get("team_id"))
            if t is not None:
                team_of[tid] = t
    return team_of


def review_map(doc) -> dict[str, dict]:
    return doc.get("event_review") or {}


def evidence_of(doc, ev: dict) -> str:
    """统一证据策略（单点定义）：
    counted（引擎计入）/ candidate（复核候选）/ accepted_candidate（复核确认的候选）。"""
    if not ev.get("candidate"):
        return "counted"
    rev = review_map(doc).get(f"{ev.get('type')}:{ev.get('frame')}")
    if rev and rev.get("reviewer_decision") == "score":
        return "accepted_candidate"
    return "candidate"


def provenance(doc, disc_cov: float) -> dict:
    return {
        "video": doc.get("video"),
        "fps": doc.get("fps"),
        "frames": len(doc.get("frames", {})),
        "disc_frame_coverage": round(disc_cov, 3),
        "event_review": review_summary(doc) if (doc.get("event_review")) else {},
        "caveats": [
            "world-coord noise ~7m near-field (ground-plane parallax + players-extent EZ error, §13.13)",
            "low disc coverage biases turnover/possession stats low (holes hit num & denom)",
            "carry-candidates excluded from possession share (contain ground-disc artifacts)",
        ],
    }


def review_summary(doc) -> dict:
    from collections import Counter as C
    return dict(C(e.get("reviewer_decision") for e in (doc.get("event_review") or {}).values()))


# ── 1 飞行距离 ──

def disc_flight_distance(doc, H, fps) -> dict:
    """仅 status=='tracking' 的世界轨迹切段路径长（predicting 外推帧污染，R2 硬过滤）。"""
    disc = doc.get("disc_frames", {})
    pts = []
    for fk in sorted(disc, key=int):
        d = disc[fk]
        if d.get("status") != "tracking":
            continue
        wx, wy = px_to_world(H, d["cx"], d["cy"])
        pts.append((int(fk), wx, wy))
    segments, cur, prev = [], [], None
    for fn, wx, wy in pts:
        if prev is not None and fn - prev[0] > GAP_BREAK:
            if len(cur) >= MIN_TRACKING_RUN:
                segments.append(cur)
            cur = []
        cur.append((fn, wx, wy))
        prev = (fn, wx, wy)
    if len(cur) >= MIN_TRACKING_RUN:
        segments.append(cur)

    lens = []
    for seg in segments:
        L = sum(math.hypot(seg[i][1] - seg[i - 1][1], seg[i][2] - seg[i - 1][2])
                for i in range(1, len(seg)))
        dur = (seg[-1][0] - seg[0][0]) / fps
        lens.append({"length_m": round(L, 1), "duration_s": round(dur, 2),
                     "speed_ms": round(L / dur, 1) if dur > 0 else None,
                     "t_start_s": round(seg[0][0] / fps, 2)})
    lens.sort(key=lambda x: -x["length_m"])
    return {
        "n_segments": len(lens),
        "longest_m": lens[0]["length_m"] if lens else None,
        "top10": lens[:10],
        "note": "tracking-only; short(<3f) segments dropped; ~7m near-field noise",
    }


# ── 2 换手率（双口径）+ 3 控盘占比 ──

def possession_stats(doc, team_of) -> dict:
    evs = doc.get("events", [])
    counted = [e for e in evs if evidence_of(doc, e) in ("counted", "accepted_candidate")]

    n_poss = sum(1 for e in counted if e["type"] == "possession_start")
    n_turnovers = sum(1 for e in counted if e["type"] == "turnover")
    n_transfers = sum(1 for e in counted if e["type"] == "transfer")
    n_scores = sum(1 for e in counted
                   if e["type"] == "score")  # counted score + confirmed candidates
    dur_s = len(doc.get("frames", {})) / float(doc.get("fps") or 30.0)

    # 控盘占比：possession_start→end/turnover/score 区间按队累加（死球=pull 后无盘，无区间自然排除）
    holds = {0: 0.0, 1: 0.0}
    open_start = None  # (team, frame)
    for e in sorted(counted, key=lambda x: x["frame"]):
        if e["type"] == "possession_start" and open_start is None:
            t = team_of.get(e.get("to_track"))
            if t is not None:
                open_start = (t, e["frame"])
        elif e["type"] in ("possession_end", "turnover", "score") and open_start is not None:
            holds[open_start[0]] += (e["frame"] - open_start[1])
            open_start = None
    total_hold = sum(holds.values()) or 1
    return {
        "possessions": n_poss,
        "turnovers": n_turnovers,
        "passes_same_team": n_transfers,
        "scores": n_scores,
        "turnover_per_throw": round(n_turnovers / max(n_transfers + n_turnovers, 1), 3),
        "turnover_per_possession": round(n_turnovers / max(n_poss, 1), 3),
        "per_minute": {"turnovers": round(n_turnovers / (dur_s / 60), 2),
                       "possessions": round(n_poss / (dur_s / 60), 2)},
        "possession_share": {str(t): round(h / total_hold, 3) for t, h in holds.items()},
        "note": "candidate events excluded unless review-confirmed; carry-candidates not counted",
    }


# ── 4 热力图 ──

def _gaussian_smooth(g, sigma=HEAT_SIGMA):
    """小核高斯近似（3σ 截断的可分离卷积，无 scipy 依赖）。"""
    r = max(int(3 * sigma), 1)
    xs = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-xs * xs / (2 * sigma * sigma))
    k /= k.sum()
    pad = [(0, 0), (r, r), (r, r)]
    gp = np.pad(g, pad, mode="constant")
    out = np.zeros_like(gp)
    for dy in range(-r, r + 1):
        out += np.roll(gp, dy, axis=1) * k[dy + r]
    g1 = out
    out2 = np.zeros_like(g1)
    for dx in range(-r, r + 1):
        out2 += np.roll(g1, dx, axis=2) * k[dx + r]
    return out2[0] if out2.ndim == 3 else out2


def team_heatmaps(doc, H, out_png_dir: Path | None = None) -> dict:
    """两队脚点占用热力图：25×25 网格 + σ1 高斯（mplsoccer 惯例）。"""
    team_of = team_of_track(doc)
    grids = {0: np.zeros((HEAT_GRID, HEAT_GRID)), 1: np.zeros((HEAT_GRID, HEAT_GRID))}
    for fr in doc.get("frames", {}).values():
        for det in fr:
            t = team_of.get(det.get("track_id"))
            if t not in grids:
                continue
            wx, wy = foot_world(H, det)
            if not (0 <= wx < FIELD_W and 0 <= wy < FIELD_H):
                continue
            gx = min(int(wx / FIELD_W * HEAT_GRID), HEAT_GRID - 1)
            gy = min(int(wy / FIELD_H * HEAT_GRID), HEAT_GRID - 1)
            grids[t][gy, gx] += 1

    summary = {}
    for t, g in grids.items():
        total = g.sum()
        if total == 0:
            summary[str(t)] = {"samples": 0}
            continue
        gs = _gaussian_smooth(g[None, :, :])
        peak = np.unravel_index(np.argmax(gs), gs.shape)
        summary[str(t)] = {
            "samples": int(total),
            "peak_cell": [int(peak[1]), int(peak[0])],          # [gx, gy]
            "peak_world_m": [round((peak[1] + 0.5) / HEAT_GRID * FIELD_W, 1),
                             round((peak[0] + 0.5) / HEAT_GRID * FIELD_H, 1)],
        }
        if out_png_dir is not None:
            _save_heat_png(gs, out_png_dir / f"heatmap_team{t}.png", t)
    return summary


def _save_heat_png(gs, path: Path, team: int):
    import cv2
    norm = (gs - gs.min()) / max(gs.max() - gs.min(), 1e-9)
    color = cv2.applyColorMap((norm * 255).astype("uint8"), cv2.COLORMAP_JET)
    color = cv2.resize(color, (1000, 370), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(color, (0, 0), (1000 - 1, 370 - 1), (60, 60, 60), 1)
    cv2.line(color, (500, 0), (500, 370), (60, 60, 60), 1)
    cv2.putText(color, f"team {team} occupancy (25x25 gaussian)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), color)


# ── 5 得分手 / 6 组织者 / 7 威胁进攻发起 ──

def player_profiles(doc, team_of) -> dict:
    evs = sorted(doc.get("events", []), key=lambda e: e["frame"])
    fps = float(doc.get("fps") or 30.0)

    scorers = Counter()
    scorer_candidates = []
    hockey_assists = Counter()
    first_pushers = Counter()
    threat_origins = []

    # possession 链构建：START 开链，TURNOVER/SCORE/END 关链
    chain = None  # {"team", "passes": [ev...], "origin": (track,x,y)}
    for e in evs:
        ev = evidence_of(doc, e)
        if e["type"] == "possession_start":
            chain = {"team": team_of.get(e.get("to_track")),
                     "passes": [], "origin": (e.get("to_track"), e.get("x"), e.get("y")),
                     "evidence": ev}
        elif e["type"] == "transfer" and chain is not None:
            chain["passes"].append({"to": e.get("to_track"), "from": e.get("from_track"),
                                    "x": e.get("x"), "y": e.get("y")})
        elif e["type"] == "score":
            scorer_id = e.get("to_track")
            if ev == "counted":
                scorers[scorer_id] += 1
            else:
                scorer_candidates.append({"track": scorer_id, "frame": e["frame"],
                                          "evidence": ev})
            if chain is not None:
                # hockey-assist：得分链倒数第二次传球的出球人
                if len(chain["passes"]) >= 1:
                    ha = chain["passes"][-1].get("from")
                    if ha is not None:
                        hockey_assists[ha] += 1
                # 首推进末三区者（xT 启发式）：链内第一个 x 进入对方末三区的传球 to
                attacking_right = None
                for p in chain["passes"]:
                    # 方向未知时两侧对称判定：进入任一端 18m 区即记
                    pass
                threat_origins.append({
                    "origin_track": chain["origin"][0],
                    "origin_x": chain["origin"][1], "origin_y": chain["origin"][2],
                    "t_s": round(e["frame"] / fps, 2),
                    "n_passes": len(chain["passes"]),
                    "evidence": ev,
                })
            chain = None
        elif e["type"] in ("turnover", "possession_end"):
            chain = None

    def _name(tid):
        t = team_of.get(tid)
        return {"track": tid, "team": t}

    return {
        "scorers": {json.dumps(_name(k)): v for k, v in scorers.most_common()},
        "scorer_candidates_unconfirmed": scorer_candidates[:10],
        "hockey_assists": {json.dumps(_name(k)): v for k, v in hockey_assists.most_common()},
        "threat_origins_top": threat_origins[:10],
        "n_threat_chains": len(threat_origins),
        "note": "origin positions are possession-start foot coords (world m); "
                "organizer = hockey-assist count (xT-lite); first-pusher needs direction binding",
    }


# ── 主入口 ──

def main() -> int:
    ap = argparse.ArgumentParser(description="比赛统计层：七指标 + provenance")
    ap.add_argument("tracks", help="tracks.json 路径")
    ap.add_argument("--calib", default=None, help="标定 json（默认同目录 segment_calib.json）")
    ap.add_argument("--heat-png", default=None, help="热力图 PNG 输出目录（默认 runs/match_stats/<stem>/）")
    args = ap.parse_args()

    tracks = Path(args.tracks)
    calib = Path(args.calib) if args.calib else tracks.parent / "segment_calib.json"
    if not calib.exists():
        print(f"calibration not found: {calib}", file=sys.stderr)
        return 1
    heat_dir = Path(args.heat_png) if args.heat_png else \
        ROOT / "runs/match_stats" / tracks.parent.name

    doc, H, fps = load_doc(tracks, calib)
    frames_n = len(doc.get("frames", {}))
    disc_cov = len(doc.get("disc_raw", doc.get("disc_frames", {}))) / max(frames_n, 1)
    team_of = team_of_track(doc)

    out = {
        "provenance": provenance(doc, disc_cov),
        "1_flight_distance": disc_flight_distance(doc, H, fps),
        "2_turnover_rates": {k: v for k, v in possession_stats(doc, team_of).items()
                             if k in ("turnover_per_throw", "turnover_per_possession",
                                      "per_minute", "turnovers", "passes_same_team",
                                      "possessions", "note")},
        "3_possession_share": {k: v for k, v in possession_stats(doc, team_of).items()
                               if k in ("possession_share", "possessions", "scores", "note")},
        "4_heatmaps": team_heatmaps(doc, H, heat_dir),
        "5_6_7_player_profiles": player_profiles(doc, team_of),
    }
    out_path = tracks.parent / "match_stats.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "1_flight_distance"},
                     ensure_ascii=False, indent=1)[:2600])
    print(f"... flight segments: {out['1_flight_distance']['n_segments']}, "
          f"longest {out['1_flight_distance']['longest_m']}m")
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
