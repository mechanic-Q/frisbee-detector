"""迭代对比：v2 vs v3（同 300 帧）检出密度与 HSV 一致性并排报告。

用法：
    python3 tools/compare_iters.py --a runs/gui_analysis/accept10_v2_gpu300/tracks.json \
        --b runs/gui_analysis/accept10_v3_gpu300/tracks.json --video <accept_10min.mp4>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frisbee_analyzer import protocol  # noqa: E402


def density(doc: dict) -> dict:
    dets = [d for v in doc["frames"].values() for d in v]
    cls = Counter(d.get("cls") for d in dets)
    n = max(1, len(doc["frames"]))
    return {"dets": len(dets), "per_frame": round(len(dets) / n, 1),
            "per_frame_cls": {k: round(v / n, 1) for k, v in sorted(cls.items())}}


def color_check(tracks: str, video: str) -> dict:
    tool = Path(__file__).parent / "team_color_check.py"
    out = Path(protocol.win_to_wsl(tracks)).parent / "team_color_check.json"
    subprocess.run(["python3", str(tool), "--tracks", tracks, "--video", video,
                    "--sample-interval", "60"], check=True,
                   stdout=subprocess.DEVNULL)
    r = json.loads(out.read_text(encoding="utf-8"))
    return {"crop": r.get("crop_consistency"), "track": r.get("track_consistency"),
            "decisive": r.get("crops_decisive"), "distinct": r.get("cross_team_distinct")}


def main() -> int:
    parser = argparse.ArgumentParser(description="迭代间对比报告")
    parser.add_argument("--a", required=True, help="v2 tracks.json")
    parser.add_argument("--b", required=True, help="v3 tracks.json")
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    report = {}
    for name, path in (("v2", args.a), ("v3", args.b)):
        doc = json.loads(Path(protocol.win_to_wsl(path)).read_text(encoding="utf-8"))
        report[name] = {"density": density(doc), "color": color_check(path, args.video)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = Path(protocol.win_to_wsl(args.b)).parent / "iter_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    main()
