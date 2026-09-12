#!/usr/bin/env python3
"""Stage 2 分块聚合：事件合并 → 窗口 GT 对齐 → ROI 因果审计。

用法: python tools/s2_aggregate.py --chunks 5 --chunk-len 120 --offset 300 --duration 600
前提: results/f1_matrix/s2_chunk<i>/tracks.json (+ segment_calib.json)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frisbee_analyzer.events_runner import compute_events  # noqa: E402
from tools.eval_engine_vs_gt import match  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--chunk-len", type=float, default=120.0)
    ap.add_argument("--offset", type=float, default=300.0, help="accept_10min 在原片的起点秒")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--gt", default="results/auto_gt_events.json")
    ap.add_argument("--tol", type=float, default=90.0)
    args = ap.parse_args()

    all_events = []
    calib_fail = []
    for i in range(args.chunks):
        d = Path(f"results/f1_matrix/s2_chunk{i}")
        tracks, calib = d / "tracks.json", d / "segment_calib.json"
        if not tracks.exists():
            print(f"chunk{i}: MISSING tracks.json")
            continue
        doc = json.loads(tracks.read_text(encoding="utf-8"))
        if not calib.exists():
            calib_fail.append(i)
            print(f"chunk{i}: no segment_calib.json — 事件跳过")
            continue
        out = compute_events(doc, calib)
        evs = out["events"]
        for e in evs:
            e["t_sec"] = round(e["t_sec"] + i * args.chunk_len, 2)  # → accept 时间轴
            e["chunk"] = i
        all_events.extend(evs)
        print(f"chunk{i}: {len(evs)} events {dict(Counter(e['type'] for e in evs))}")

    types = Counter(e["type"] for e in all_events)
    print(f"\n== 合并事件（accept 时间轴）: {len(all_events)} {dict(types)}")

    # 90s 新颖性链（§13.13 迭代2）：得分（状态机 score 或更早候选）之后 90s 内的
    # 候选 = followup（拉盘者等待/得分后走回伪影），不计为新提案。
    NOVELTY_WINDOW = 90.0
    last_score_t = None
    for e in sorted(all_events, key=lambda x: x["t_sec"]):
        if e["type"] != "score":
            continue
        if e.get("candidate"):
            if last_score_t is not None and e["t_sec"] - last_score_t <= NOVELTY_WINDOW:
                e["novelty"] = "followup"
            else:
                e["novelty"] = "novel"
                last_score_t = e["t_sec"]
        else:
            last_score_t = e["t_sec"]  # 状态机得分同样开启禁排期
    for e in all_events:
        if e["type"] in ("score", "possession_start", "turnover", "transfer"):
            print(f"  chunk{e['chunk']} t={e['t_sec']:7.2f} {e['type']:16s} "
                  f"team={e.get('team')} cand={e.get('candidate', False)} "
                  f"novelty={e.get('novelty', '-')} {str(e.get('detail',''))[:60]}")

    # 窗口 GT 对齐（accept→原片 = +offset）
    gt_doc = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    gt_win = [g for g in gt_doc["events"]
              if args.offset <= g.get("t_sec", 0) <= args.offset + args.duration]
    gt_shift = [dict(g, t_sec=g["t_sec"] - args.offset) for g in gt_win]
    scores = [e for e in all_events if e["type"] == "score" and e.get("novelty") != "followup"]
    # 全局最小 dt 一对一匹配（贪心按 dt 升序取两边都空闲的）——比按引擎时间
    # 顺序的局部贪心更稳（避免先到者以大 dt 抢占，§13.13 迭代3 实证）。
    pairs = []
    for ei, e in enumerate(scores):
        for gi, g in enumerate(gt_shift):
            dt = abs(g["t_sec"] - e["t_sec"])
            if dt <= args.tol:
                pairs.append((dt, ei, gi))
    pairs.sort()
    used_e, used_g, matched = set(), set(), []
    for dt, ei, gi in pairs:
        if ei in used_e or gi in used_g:
            continue
        used_e.add(ei)
        used_g.add(gi)
        matched.append({"engine": scores[ei], "gt": gt_shift[gi], "dt_sec": round(dt, 1)})
    e_only = [e for ei, e in enumerate(scores) if ei not in used_e]
    g_only = [g for gi, g in enumerate(gt_shift) if gi not in used_g]
    n_follow = sum(1 for e in all_events if e.get("novelty") == "followup")
    print(f"\n== 窗口 GT: {len(gt_win)} 个得分 | 引擎 novel score 提案: {len(scores)} "
          f"(+{n_follow} followup) | matched: {len(matched)}")
    print(f"   P={len(matched)/len(scores):.3f}  R={len(matched)/len(gt_win):.3f}" if scores else "   引擎无得分事件")
    for m in matched:
        print(f"   MATCH engine t={m['engine']['t_sec']:.1f} ({'cand' if m['engine'].get('candidate') else 'state'}"
              f", chunk{m['engine'].get('chunk')}) <-> GT t={m['gt']['t_sec']+args.offset:.0f}s "
              f"({m['gt'].get('team')} via {','.join(m['gt'].get('sources', []))}) dt={m['dt_sec']:.1f}s")
    for g in g_only:
        print(f"   MISS  GT t={g['t_sec']+args.offset:.0f}s {g.get('team')} via {','.join(g.get('sources', []))}")
    for e in e_only:
        print(f"   FALSE engine t={e['t_sec']:.1f} chunk{e.get('chunk')} "
              f"{'cand' if e.get('candidate') else 'state'} {str(e.get('detail',''))[:60]}")

    out = Path("results/f1_matrix/s2_aggregate.json")
    out.write_text(json.dumps({
        "events": all_events, "gt_in_window": gt_win,
        "matched": len(matched), "engine_scores": len(scores), "gt_scores": len(gt_win),
        "calib_fail_chunks": calib_fail,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
