#!/usr/bin/env python3
"""窗口级 GT 对齐评测：引擎得分事件 vs E6 自动 GT（GT 分母只计素材窗口内）。

解决 §13.11 "GT 全场 15 事件口径下 R=1/15 是分母假象"——只把素材时间窗
内的 GT 事件计入分母，引擎时间戳按 --offset 平移回原片时间轴再对齐。

用法:
  python tools/eval_window_gt.py TRACKS_JSON --offset 300 --duration 600 [--tol 90]
  # offset=素材起点在原片的秒数; 引擎 t_sec + offset = 原片时间
匹配复用 eval_engine_vs_gt.match（同类型 ±tol 一对一贪心）。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_engine_vs_gt import match  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_events", help="tracks.json（取 events 字段）或事件数组 json")
    ap.add_argument("--gt", default="results/auto_gt_events.json")
    ap.add_argument("--offset", type=float, required=True, help="素材起点在原片的秒数")
    ap.add_argument("--duration", type=float, required=True, help="素材时长秒")
    ap.add_argument("--tol", type=float, default=90.0)
    ap.add_argument("--types", default="score", help="参与对齐的事件类型（逗号分隔）")
    args = ap.parse_args()

    d = json.loads(Path(args.engine_events).read_text(encoding="utf-8"))
    engine = d.get("events", d) if isinstance(d, dict) else d
    engine = [e for e in engine if e.get("type") in set(args.types.split(","))]

    gt_doc = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    gt_all = gt_doc.get("events", gt_doc) if isinstance(gt_doc, dict) else gt_doc
    t0, t1 = args.offset, args.offset + args.duration
    gt_win = [g for g in gt_all
              if g.get("type") in set(args.types.split(","))
              and t0 <= g.get("t_sec", 0) <= t1]
    # 平移回素材时间轴（match 按同轴比较）
    gt_shift = [dict(g, t_sec=g["t_sec"] - t0) for g in gt_win]

    matched, e_only, g_only = match(engine, gt_shift, args.tol)
    n_e, n_g = len(engine), len(gt_win)
    summary = {
        "window": [t0, t1],
        "engine_score_events": n_e,
        "gt_in_window": n_g,
        "matched": len(matched),
        "precision": round(len(matched) / n_e, 3) if n_e else None,
        "recall": round(len(matched) / n_g, 3) if n_g else None,
        "tolerance_sec": args.tol,
        "matched_detail": [{"engine_t": e.get("t_sec"), "gt_t_orig": m["gt"]["t_sec"] + t0,
                            "dt": m["dt_sec"], "engine": e.get("type"),
                            "candidate": e.get("candidate", False),
                            "gt_team": m["gt"].get("team"), "gt_src": m["gt"].get("sources")}
                           for e, m in [(me["engine"], me) for me in matched]],
        "engine_only": [{"t": e.get("t_sec"), "type": e.get("type"),
                         "candidate": e.get("candidate", False),
                         "detail": str(e.get("detail", ""))[:80]} for e in e_only],
        "gt_missed": [{"t_orig": g["t_sec"] + t0, "team": g.get("team"),
                       "sources": g.get("sources")} for g in g_only],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out = Path(args.engine_events).parent / "window_gt_eval.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
