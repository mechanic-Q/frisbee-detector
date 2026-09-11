#!/usr/bin/env python3
"""对 worker 产物 tracks.json 计算事件并汇总（端区持盘候选/状态机事件分布）。

用法: python tools/summarize_events.py TRACKS_JSON [CALIB_JSON]
  CALIB_JSON 缺省取 tracks.json 同目录的 segment_calib.json（--auto-calibrate 产物）。

输出: 状态机事件计数 + score/possession 类事件明细 + 端区得分候选列表。
纯 CPU。
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frisbee_analyzer.events_runner import compute_events  # noqa: E402


def main() -> int:
    tracks = Path(sys.argv[1])
    calib = Path(sys.argv[2]) if len(sys.argv) > 2 else tracks.parent / "segment_calib.json"
    doc = json.loads(tracks.read_text(encoding="utf-8"))
    out = compute_events(doc, calib)

    events = out["events"]
    counts = Counter(e["type"] for e in events)
    print(f"== {tracks.name} (calib={calib.name})")
    print(f"events: {len(events)}  {dict(counts)}  score={out['score']}")
    for e in events:
        if e["type"] in ("score", "possession_start", "possession_end", "transfer", "turnover", "pull"):
            print(f"  f{e['frame']:6d} t={e['t_sec']:7.2f} {e['type']:16s} "
                  f"team={e.get('team')} {e.get('detail', '')[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
