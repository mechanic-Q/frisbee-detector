"""auto_calib vs 启发式过滤的独立验证（观众排除 + 每帧密度）。"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer.filters import FIELD_POLYGON_M, point_in_polygon  # noqa: E402
from utils.homography import load_calibration, pixel_to_world  # noqa: E402

raw = json.loads(Path("runs/gui_analysis/raw300/tracks.json").read_text(encoding="utf-8"))
auto = json.loads(Path("runs/gui_analysis/raw300/auto_calib.json").read_text(encoding="utf-8"))
m = np.array(auto["matrix"])
H = raw.get("height") or 1080

total = kept_auto = stands = stands_in = 0
pf = []
cls_in = Counter()
for v in raw["frames"].values():
    k = 0
    for d in v:
        x1, y1, x2, y2 = d["bbox"]
        total += 1
        wx, wy = pixel_to_world(m, (x1 + x2) / 2, y2)
        inside = point_in_polygon(wx, wy, FIELD_POLYGON_M)
        if inside:
            k += 1
        if y2 < 220:  # 观众席带（远侧边线以上，目视边界）
            stands += 1
            if inside:
                stands_in += 1
    kept_auto += k
    pf.append(k)
print(f"auto 多边形过滤: {kept_auto}/{total} ({kept_auto/total:.0%}) = {kept_auto/300:.1f}/帧")
print(f"观众席带(y2<220): {stands} 框，其中误保留 {stands_in} ({stands_in/max(1,stands):.0%})")
print(f"每帧: 中位 {statistics.median(pf)}")
# 对照组：启发式 min_y_frac 0.30
heur = sum(1 for v in raw["frames"].values() for d in v
           if (d["bbox"][3] - d["bbox"][1]) >= H * 0.083 and d["bbox"][3] >= H * 0.30)
print(f"启发式(0.30): {heur}/{total} ({heur/total:.0%}) = {heur/300:.1f}/帧")
