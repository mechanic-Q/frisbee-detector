"""players_e4 自评：对比零样本基线与微调权重的检出结构，并给出量化指标。

用法（训练完成后）：
    python3 tools/eval_players.py --old runs/gui_analysis/accept_10min/tracks.json \
        --new runs/gui_analysis/accept_10min_v1/tracks.json
指标：
  - dets/frame（预期下降：观众框被类别排除）
  - 队伍框占比（cls 0/1 应接近 1:1 或真实轮换比）
  - 裁判框数量（零样本基线为 0，微调后应 >0）
  - 未分配占比（cls 路径下 = 观众/忽略类被丢弃后的 None 残留，应 ≈0）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer import protocol  # noqa: E402


def load(p: str) -> dict:
    return json.loads(Path(protocol.win_to_wsl(p)).read_text(encoding="utf-8"))


def describe(doc: dict, label: str) -> dict:
    dets = [d for v in doc.get("frames", {}).values() for d in v]
    per_frame = round(len(dets) / max(1, len(doc.get("frames", {}))), 2)
    teams = Counter(d.get("team_id") for d in dets)
    clss = Counter(d.get("cls") for d in dets)
    assigned = teams.get(0, 0) + teams.get(1, 0) + teams.get(2, 0)
    return {
        "label": label,
        "frames": len(doc.get("frames", {})),
        "dets": len(dets),
        "dets_per_frame": per_frame,
        "team_dist": {str(k): v for k, v in sorted(teams.items(), key=lambda kv: str(kv[0]))},
        "assigned_ratio": round(assigned / len(dets), 3) if dets else None,
        "cls_dist": {str(k): v for k, v in sorted(clss.items(), key=lambda kv: str(kv[0]))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="players_e4 自评对比")
    parser.add_argument("--old", required=True, help="零样本基线 tracks.json")
    parser.add_argument("--new", required=True, help="微调权重 tracks.json")
    args = parser.parse_args()

    rows = [describe(load(args.old), "baseline-zeroshot"), describe(load(args.new), "players_e4_v1")]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    base, new = rows
    delta = {
        "dets_per_frame_delta": round(new["dets_per_frame"] - base["dets_per_frame"], 2),
        "assigned_ratio_delta": round((new["assigned_ratio"] or 0) - (base["assigned_ratio"] or 0), 3),
        "referee_dets_new": new["team_dist"].get("2", 0),
    }
    print(json.dumps(delta, ensure_ascii=False, indent=2))
    out = Path(args.new).parent / "eval_compare.json"
    out.write_text(json.dumps({"rows": rows, "delta": delta}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    main()
