"""诊断：深色衫素材上 HSV 路由标色的诚实性（team 平均 rf 是否与其 colors 标签一致）。"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer import protocol
from frisbee_analyzer.team import collect_crops, jersey_fraction

doc = json.loads(Path(protocol.win_to_wsl(sys.argv[1])).read_text(encoding="utf-8"))
# 先从原始 doc 提取 team 归属，再重置（顺序不能反）
team_of = {}
for dets in doc["frames"].values():
    for d in dets:
        if d.get("team_id") in (0, 1) and d["track_id"] not in team_of:
            team_of[d["track_id"]] = d["team_id"]
frames = {k: [d for d in v] for k, v in doc["frames"].items()}
for dets in frames.values():
    for d in dets:
        d["team_id"] = None
track_ids, crops, _xs = collect_crops(frames, protocol.win_to_wsl(doc["video"]))
per_track = defaultdict(list)
for tid, crop in zip(track_ids, crops):
    per_track[tid].append(jersey_fraction(crop))
feats = {t: (float(np.mean([f[0] for f in fs])), float(np.mean([f[1] for f in fs])))
         for t, fs in per_track.items()}
by_team = defaultdict(list)
for tid, (rf, _bf) in feats.items():
    if tid in team_of:
        by_team[team_of[tid]].append(rf)
for t, rfs in sorted(by_team.items()):
    print(f"team{t}: mean rf={np.mean(rfs):.3f} (n={len(rfs)} tracks)")
print("team_colors:", doc.get("team_colors"))
