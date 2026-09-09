"""事件引擎 vs E6 自动 GT 的对齐评测（纯 CPU）。
匹配规则：同类型事件 ±90s 容差（依据 E6 调研：三路 GT 时间精度约分钟级）。

用法:
  python tools/eval_engine_vs_gt.py <engine_events.json> [--gt results/auto_gt_events.json] [--tol 90]
engine_events.json 可为：
  - worker 产物 tracks.json（取其中 "events" 字段）
  - 直接的事件数组
"""
import argparse
import json
from pathlib import Path


def load_events(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, dict):
        return d.get("events", [])
    return d


def t_of(e):
    return e.get("t_sec", e.get("t_start", 0)) or 0


def match(engine, gt, tol=90.0):
    """贪心一对一匹配，返回 (matched, engine_only, gt_only)"""
    used = [False] * len(gt)
    matched = []
    engine_only = []
    for e in sorted(engine, key=t_of):
        te = t_of(e)
        best_j, best_d = None, None
        for j, g in enumerate(gt):
            if used[j] or g.get("type") != e.get("type"):
                continue
            d = abs(t_of(g) - te)
            if d <= tol and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j is not None:
            used[best_j] = True
            matched.append({"engine": e, "gt": gt[best_j], "dt_sec": round(best_d, 1)})
        else:
            engine_only.append(e)
    gt_only = [g for j, g in enumerate(gt) if not used[j]]
    return matched, engine_only, gt_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_events")
    ap.add_argument("--gt", default="results/auto_gt_events.json")
    ap.add_argument("--tol", type=float, default=90.0)
    args = ap.parse_args()

    eng = load_events(Path(args.engine_events))
    gt_doc = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    gt = gt_doc.get("events", gt_doc) if isinstance(gt_doc, dict) else gt_doc

    matched, e_only, g_only = match(eng, gt, args.tol)
    n_e, n_g = len(eng), len(gt)
    out = {
        "engine_events": n_e,
        "gt_events": n_g,
        "tolerance_sec": args.tol,
        "matched": len(matched),
        "engine_precision": round(len(matched) / n_e, 3) if n_e else None,
        "engine_recall": round(len(matched) / n_g, 3) if n_g else None,
        "f1": round(2 * len(matched) / (n_e + n_g), 3) if (n_e + n_g) else None,
        "matched_detail": matched[:20],
        "engine_only_count": len(e_only),
        "gt_only_count": len(g_only),
        "engine_only": e_only[:10],
        "gt_only": g_only[:10],
    }
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("matched_detail", "engine_only", "gt_only")},
                     ensure_ascii=False, indent=1))
    Path("results/engine_vs_gt.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved -> results/engine_vs_gt.json")


if __name__ == "__main__":
    main()
